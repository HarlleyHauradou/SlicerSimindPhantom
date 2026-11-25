"""
Slicer-independent phantom pipeline.

The Slicer logic only converts MRML nodes into SimpleITK images / numpy masks
and calls build_phantom() + export_*(). Keeping the pipeline here allows an
end-to-end test without Slicer (tests/test_pipeline.py).
"""
import datetime
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import SimpleITK as sitk

from . import core
from . import zubal_table as zt


@dataclass
class StructureRow:
    key: str                 # segment id, or "bg:<class>" for HU background classes
    name: str                # segment name / class label shown in the table
    role: str                # core.ROLE_*
    zubal_index: int
    activity: int = 0
    known: bool = True       # False if no mapping rule matched the name
    voxels_hint: int = 0     # approximate voxel count (paint order), 0 = unknown
    voxels: int = 0          # voxel count at CT resolution after painting
    volume_cm3: float = 0.0  # volume at CT resolution after painting


def rows_from_segments(segments, rules, zubal_table=None, include_background=True):
    """segments: list of (segment_id, segment_name, voxel_count_hint)."""
    rows = []
    for seg_id, name, hint in segments:
        rule = core.match_rule(name, rules)
        if rule is None:
            rows.append(StructureRow(seg_id, name, core.ROLE_ORGAN, zt.UNKNOWN_DEFAULT_INDEX,
                                     0, known=False, voxels_hint=int(hint or 0)))
        else:
            rows.append(StructureRow(seg_id, name, rule.role, rule.zubal_index, rule.activity,
                                     True, int(hint or 0)))
    if include_background:
        rows.extend(background_rows())
    return rows


def background_rows():
    return [StructureRow("bg:" + key, label, core.ROLE_BACKGROUND, idx, 0)
            for key, label, idx in core.BACKGROUND_CLASSES]


@dataclass
class PhantomOptions:
    fill_background: bool = True
    add_skin: bool = True
    thresholds: dict = field(default_factory=lambda: dict(core.DEFAULT_THRESHOLDS))
    body_hu_threshold: float = -500.0
    spacing: tuple = None          # mm (x,y,z); None = CT spacing
    matrix: tuple = None           # (nx,ny,nz); None = auto
    crop_to_body: bool = True
    margin_mm: float = 10.0
    reference_grid: core.Grid = None   # exact grid (e.g. SPECT); overrides spacing/matrix
    first_slice: str = "head"
    flip_x: bool = False
    flip_y: bool = False
    compute_density: bool = True
    curve_path: str = None
    density_outside_body_zero: bool = True


@dataclass
class PhantomResult:
    grid: core.Grid
    struct: np.ndarray        # (z,y,x) structure ids on output grid, LPS
    zubal: np.ndarray         # uint8-range Zubal indices on output grid
    activity: np.ndarray      # int activity on output grid
    density: np.ndarray       # float32 g/cm3 on output grid (or None)
    rows: list
    options: PhantomOptions
    curve: tuple
    warnings: list
    checks: dict
    input_info: dict


def validate_rows(rows):
    errors = []
    for r in rows:
        if r.role in (core.ROLE_BODY, core.ROLE_IGNORE):
            continue
        if not (0 <= int(r.zubal_index) <= core.UINT8_MAX):
            errors.append(f"'{r.name}': Zubal index {r.zubal_index} outside 0..255.")
        if not (0 <= int(r.activity) <= core.INT16_MAX):
            errors.append(f"'{r.name}': activity {r.activity} outside 0..{core.INT16_MAX}.")
    if errors:
        raise core.PhantomError("\n".join(errors))


def build_phantom(ct_img, get_mask, rows, options, progress=None):
    """
    ct_img   : SimpleITK image in HU (any orientation).
    get_mask : callable(key) -> bool array (z,y,x) on the ct_img voxel grid.
    rows     : list of StructureRow (all rows, including background classes).
    progress : optional callable(fraction, message).
    """
    def report(f, msg):
        if progress:
            progress(f, msg)

    validate_rows(rows)
    warnings = []
    ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)
    hu = sitk.GetArrayFromImage(ct_img)
    checks, w = core.sanity_check_hu(hu)
    warnings += w

    ids = {r.key: i + 1 for i, r in enumerate(rows)}
    n_ids = len(rows) + 1
    if n_ids > np.iinfo(np.uint16).max:
        raise core.PhantomError("Too many structures.")
    struct = np.zeros(hu.shape, dtype=np.uint16)

    # 1) paint organs and lesions (large first, small later, lesions last)
    paintable = [r for r in rows if r.role in (core.ROLE_ORGAN, core.ROLE_LESION)]
    hints = {}
    for r in paintable:
        hints[r.key] = r.voxels_hint
    order = core.paint_order([(r.key, r.role, hints[r.key]) for r in paintable])
    body_from_segments = np.zeros(hu.shape, dtype=bool)
    for k, key in enumerate(order):
        report(0.6 * k / max(len(order), 1), f"Painting {key}")
        m = get_mask(key)
        if m.shape != hu.shape:
            raise core.PhantomError(f"Mask of '{key}' has shape {m.shape}, CT has {hu.shape}.")
        struct[m.astype(bool)] = ids[key]

    # 2) body mask
    body_rows = [r for r in rows if r.role == core.ROLE_BODY]
    for r in body_rows:
        body_from_segments |= get_mask(r.key).astype(bool)
    if body_rows and body_from_segments.any():
        body = body_from_segments
    else:
        report(0.62, "Body mask from HU")
        body = core.body_mask_from_hu(hu, options.body_hu_threshold)
    body |= struct > 0

    # 3) HU background
    if options.fill_background:
        report(0.65, "Background fill by HU")
        class_ids = {r.key[3:]: ids[r.key] for r in rows if r.role == core.ROLE_BACKGROUND}
        core.fill_background(struct, hu, body, class_ids, options.thresholds)

    # voxel counts / volumes at CT resolution
    counts = np.bincount(struct.ravel(), minlength=n_ids)
    vox_cm3 = float(np.prod(ct_img.GetSpacing())) / 1000.0
    for r in rows:
        r.voxels = int(counts[ids[r.key]])
        r.volume_cm3 = r.voxels * vox_cm3

    # 4) orientation -> LPS
    report(0.7, "Resampling")
    struct_img = sitk.DICOMOrient(core.image_like(struct, ct_img), "LPS")
    body_img = sitk.DICOMOrient(core.image_like(body.astype(np.uint8), ct_img), "LPS")
    ct_lps = sitk.DICOMOrient(ct_img, "LPS")

    # 5) output grid
    if options.reference_grid is not None:
        grid = options.reference_grid
    else:
        center = sitk.GetArrayFromImage(body_img) > 0 if options.crop_to_body else None
        grid = core.make_grid(ct_lps, options.spacing, options.matrix, center,
                              options.margin_mm if options.crop_to_body else 0.0)
    if options.matrix is not None and options.reference_grid is None and options.crop_to_body:
        # warn if the body does not fit in the requested matrix
        bb = sitk.GetArrayFromImage(body_img) > 0
        zyx = np.argwhere(bb)
        if zyx.size:
            ext = (zyx.max(0) - zyx.min(0) + 1)[::-1] * np.array(ct_lps.GetSpacing())
            fov = np.array(grid.size) * np.array(grid.spacing)
            if np.any(ext > fov + 1e-6):
                warnings.append(f"Body extent {np.round(ext, 1).tolist()} mm exceeds the output "
                                f"FOV {np.round(fov, 1).tolist()} mm: the phantom is truncated.")

    struct_out = sitk.GetArrayFromImage(core.resample(struct_img, grid, sitk.sitkNearestNeighbor, 0))
    body_out = sitk.GetArrayFromImage(core.resample(body_img, grid, sitk.sitkNearestNeighbor, 0)) > 0
    body_out |= struct_out > 0

    # 6) skin on the output grid (thickness = 1 output voxel)
    skin_rows = [r for r in rows if r.key == "bg:skin"]
    if options.add_skin and skin_rows:
        lesion_ids = [ids[r.key] for r in rows if r.role == core.ROLE_LESION]
        core.add_skin(struct_out, ids["bg:skin"], lesion_ids, body=body_out)

    zlut = np.zeros(n_ids, dtype=np.int32)
    alut = np.zeros(n_ids, dtype=np.int32)
    for r in rows:
        if r.role in (core.ROLE_BODY, core.ROLE_IGNORE):
            continue
        zlut[ids[r.key]] = int(r.zubal_index)
        alut[ids[r.key]] = int(r.activity)
    zubal = core.lookup(struct_out, zlut, np.int32)
    activity = core.lookup(struct_out, alut, np.int32)

    # 7) density from the CT (resampled as density, not HU)
    density, curve = None, None
    if options.compute_density:
        report(0.85, "Density map")
        curve = core.load_curve(options.curve_path)
        rho = core.hu_to_density(sitk.GetArrayFromImage(ct_lps), curve)
        default = core.RHO_AIR
        if options.density_outside_body_zero:
            rho[sitk.GetArrayFromImage(body_img) == 0] = 0.0
            default = 0.0
        rho_img = core.image_like(rho, ct_lps)
        density = sitk.GetArrayFromImage(
            core.resample(rho_img, grid, sitk.sitkLinear, default, smooth=True)).astype(np.float32)
        if options.density_outside_body_zero:
            density[~body_out] = 0.0

    report(1.0, "Done")
    info = {"input_size_xyz": list(ct_img.GetSize()),
            "input_spacing_mm_xyz": [round(v, 4) for v in ct_img.GetSpacing()]}
    return PhantomResult(grid, struct_out, zubal, activity, density, rows, options, curve,
                         warnings, checks, info)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def _ordered(result, arr):
    o = result.options
    return core.reorder_for_simind(arr, o.first_slice, o.flip_x, o.flip_y)


def _geometry(result):
    g, o = result.grid, result.options
    return {
        "matrix_xyz": list(g.size),
        "voxel_mm_xyz": [round(v, 4) for v in g.spacing],
        "voxel_cm_xyz": [round(v / 10.0, 5) for v in g.spacing],
        "origin_lps_mm": [round(v, 3) for v in g.origin],
        "byte_order": "little-endian",
        "voxel_order": "x fastest, then y, then z",
        "orientation": (f"LPS; first slice = {'cranial (head)' if o.first_slice == 'head' else 'caudal (feet)'}; "
                        f"flip_x={o.flip_x}; flip_y={o.flip_y}"),
    }


def _rows_json(rows):
    return [{k: v for k, v in asdict(r).items() if k != "voxels_hint"} for r in rows]


def _write_info(path, result, files, extra):
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "SimindPhantom (3D Slicer module)",
        "files": files,
        **_geometry(result),
        **extra,
        "structures": _rows_json(result.rows),
        "options": {k: v for k, v in asdict(result.options).items() if k != "reference_grid"},
        "reference_grid": asdict(result.options.reference_grid) if result.options.reference_grid else None,
        "ct_checks": result.checks,
        "input": result.input_info,
        "warnings": result.warnings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


def zub_entries(result, table):
    """One .zub entry per Zubal index present in the phantom."""
    present = sorted(int(v) for v in np.unique(result.zubal))
    act = {}
    for r in result.rows:
        if r.role in (core.ROLE_BODY, core.ROLE_IGNORE) or r.voxels == 0 and r.role != core.ROLE_BACKGROUND:
            continue
        act.setdefault(int(r.zubal_index), set()).add(int(r.activity))
    entries, warnings = [], []
    for idx in present:
        acts = act.get(idx, {0})
        if len(acts) > 1:
            warnings.append(f"Zubal index {idx} has different activities {sorted(acts)}; "
                            f"the .zub uses the maximum. Use the .smi workflow for per-structure activity.")
        e = table.get(idx)
        name = e.name if e else f"structure {idx}"
        dens = e.density if e else 1000
        if e is None:
            warnings.append(f"Zubal index {idx} is not in the Zubal table; density 1000 assumed.")
        entries.append(core.ZubalEntry(idx, name, dens, max(acts)))
    return entries, warnings


def export_zubal(result, out_dir, base, table):
    os.makedirs(out_dir, exist_ok=True)
    if result.zubal.size and result.zubal.max() > core.UINT8_MAX:
        raise core.PhantomError("A Zubal index is larger than 255: cannot write an 8-bit .dat.")
    dat = os.path.join(out_dir, base + ".dat")
    zub = os.path.join(out_dir, base + ".zub")
    info = os.path.join(out_dir, base + "_zubal_info.json")
    sha = core.write_raw(dat, _ordered(result, result.zubal), np.uint8)
    entries, warns = zub_entries(result, table)
    with open(zub, "w", encoding="utf-8", newline="\n") as f:
        f.write(core.format_zub(entries, base))
    result.warnings.extend(warns)
    _write_info(info, result, {"dat": {"path": os.path.abspath(dat), "dtype": "uint8", "sha256": sha},
                               "zub": os.path.abspath(zub)},
                {"zubal_indices_present": [e.index for e in entries]})
    return {"dat": dat, "zub": zub, "info": info}


def export_density_activity(result, out_dir, base):
    if result.density is None:
        raise core.PhantomError("The density map was not computed.")
    os.makedirs(out_dir, exist_ok=True)
    dmi = os.path.join(out_dir, base + ".dmi")
    smi = os.path.join(out_dir, base + ".smi")
    info = os.path.join(out_dir, base + "_dmi_smi_info.json")
    dens_i = np.rint(result.density * 1000.0).clip(0, core.INT16_MAX).astype(np.int32)
    sha_d = core.write_raw(dmi, _ordered(result, dens_i), np.int16)
    sha_s = core.write_raw(smi, _ordered(result, result.activity), np.int16)
    if not result.activity.any():
        result.warnings.append("All activities are 0: the .smi map is empty.")
    hu, rho, src = result.curve
    _write_info(info, result,
                {"dmi": {"path": os.path.abspath(dmi), "dtype": "int16", "value": "density[g/cm3]*1000",
                         "sha256": sha_d},
                 "smi": {"path": os.path.abspath(smi), "dtype": "int16", "value": "relative activity",
                         "sha256": sha_s}},
                {"density_range_g_cm3": [round(float(dens_i.min()) / 1000, 3),
                                         round(float(dens_i.max()) / 1000, 3)],
                 "activity_range": [int(result.activity.min()), int(result.activity.max())],
                 "calibration_curve_source": src,
                 "calibration_curve": [{"hu": float(h), "rho": float(r)} for h, r in zip(hu, rho)]})
    return {"dmi": dmi, "smi": smi, "info": info}


def summary_text(result, files):
    g = result.grid
    lines = [f"Matrix (x,y,z): {list(g.size)}",
             f"Voxel: {[round(v, 3) for v in g.spacing]} mm = {[round(v / 10, 4) for v in g.spacing]} cm",
             f"Slice order: {'head first' if result.options.first_slice == 'head' else 'feet first'}"]
    if result.density is not None:
        lines.append(f"Density range: {result.density.min():.3f} - {result.density.max():.3f} g/cm3")
    lines.append(f"Activity range: {int(result.activity.min())} - {int(result.activity.max())}")
    lines += [f"Written: {p}" for p in files.values()]
    lines += [f"WARNING: {w}" for w in result.warnings]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Activity presets (structure name -> activity)
# --------------------------------------------------------------------------
def save_activities(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "activity"])
        for r in rows:
            if r.role not in (core.ROLE_BODY, core.ROLE_IGNORE):
                wr.writerow([r.name, int(r.activity)])


def load_activities(path, rows):
    """Apply activities by structure name (case/underscore-insensitive).
    Returns the number of rows updated."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or {"name", "activity"} - {c.strip() for c in rd.fieldnames}:
            raise core.PhantomError("Activity CSV must have columns 'name' and 'activity'.")
        values = {}
        for r in rd:
            a = int(float(r["activity"]))
            if not 0 <= a <= core.INT16_MAX:
                raise core.PhantomError(f"Activity {a} for '{r['name']}' outside 0..{core.INT16_MAX}.")
            values[core.normalize_name(r["name"])] = a
    n = 0
    for row in rows:
        key = core.normalize_name(row.name)
        if key in values:
            row.activity = values[key]
            n += 1
    return n
