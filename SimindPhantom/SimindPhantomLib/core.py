"""
Core (Slicer-independent) functions for building SIMIND phantoms.

Only numpy and SimpleITK are used here (both ship with 3D Slicer), so this
module can be unit-tested outside Slicer.

Array convention: numpy arrays are indexed (z, y, x), as returned by
sitk.GetArrayFromImage / slicer.util.arrayFromVolume. Images used for
geometry are SimpleITK images in LPS physical space.
"""
import csv
import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

INT16_MAX = 32767
UINT8_MAX = 255
RHO_AIR = 0.001  # g/cm3

# Generic HU -> density curve (~120 kVp), simplified Schneider 2000 values.
# Same points as GENERIC_120KVP in the course script ct2dmi.py.
GENERIC_120KVP = [
    (-1000, 0.001), (-800, 0.20), (-100, 0.93), (0, 1.00), (100, 1.07),
    (300, 1.20), (1000, 1.60), (1800, 1.85), (3071, 1.92),
]

# Roles of a structure row
ROLE_ORGAN = "organ"
ROLE_LESION = "lesion"
ROLE_BODY = "body"          # used only as body mask (TotalSegmentator "body" task)
ROLE_IGNORE = "ignore"      # not painted
ROLE_BACKGROUND = "background"  # HU-based fill class (fat, muscle, ...)

# HU-based background classes: key -> (label, default Zubal index)
BACKGROUND_CLASSES = [
    ("skin", "Skin (outer body shell)", 1),
    ("gas", "Gas (unsegmented, HU < gas_max)", 24),
    ("fat", "Fat (unsegmented)", 22),
    ("soft", "Muscle / soft tissue (unsegmented)", 9),
    ("bone", "Bone (unsegmented, HU > soft_max)", 8),
]

DEFAULT_THRESHOLDS = {"gas_max": -200.0, "fat_max": -30.0, "soft_max": 150.0}


class PhantomError(Exception):
    """Error with a message meant for the user."""


# --------------------------------------------------------------------------
# Names and mapping rules
# --------------------------------------------------------------------------
def normalize_name(name):
    """'kidney_right' / 'Right Kidney' -> 'kidney right' / 'right kidney'."""
    s = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)


@dataclass
class MappingRule:
    pattern: str
    zubal_index: int
    activity: int = 0
    role: str = ROLE_ORGAN
    comment: str = ""

    def matches(self, name):
        return re.search(self.pattern, normalize_name(name)) is not None


def load_mapping_csv(path):
    """CSV columns: pattern, zubal_index, activity, role[, comment]. First match wins."""
    rules = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        need = {"pattern", "zubal_index"}
        if not rd.fieldnames or not need.issubset({c.strip() for c in rd.fieldnames}):
            raise PhantomError(f"Mapping CSV must have columns {sorted(need)}: {path}")
        for r in rd:
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            try:
                re.compile(r["pattern"])
            except re.error as e:
                raise PhantomError(f"Invalid regex '{r['pattern']}' in {path}: {e}")
            rules.append(MappingRule(
                pattern=r["pattern"],
                zubal_index=int(r["zubal_index"]),
                activity=int(float(r.get("activity") or 0)),
                role=r.get("role") or ROLE_ORGAN,
                comment=r.get("comment", ""),
            ))
    return rules


def save_mapping_csv(path, rules):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "zubal_index", "activity", "role", "comment"])
        for r in rules:
            w.writerow([r.pattern, r.zubal_index, r.activity, r.role, r.comment])


def match_rule(name, rules):
    """Return the first rule matching `name`, or None."""
    for r in rules:
        if r.matches(name):
            return r
    return None


# --------------------------------------------------------------------------
# Zubal index table (.zub)
# --------------------------------------------------------------------------
@dataclass
class ZubalEntry:
    index: int
    name: str
    density: int          # mg/cm3 (= g/cm3 * 1000), as in vox_man.zub
    activity: int = 0


def load_zubal_csv(path):
    """CSV columns: index, name, density_mg_cm3[, activity]."""
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for r in rd:
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            e = ZubalEntry(int(r["index"]), r["name"], int(float(r["density_mg_cm3"])),
                           int(float(r.get("activity") or 0)))
            table[e.index] = e
    return table


_ZUB_LINE = re.compile(r"^\s*(?P<name>.*?\S)\s+(?P<idx>\d+)\s+(?P<dens>\d+)\s+(?P<act>\d+)\s*$")


def parse_zub(text):
    """Parse a SIMIND .zub file ('name index density activity' per line).
    Lines that do not match (headers such as '== V4.3 Code Section ...') are skipped."""
    table = {}
    for line in text.splitlines():
        if line.lstrip().startswith("=="):
            continue
        m = _ZUB_LINE.match(line)
        if m:
            e = ZubalEntry(int(m["idx"]), m["name"], int(m["dens"]), int(m["act"]))
            table[e.index] = e
    if not table:
        raise PhantomError("No 'name index density activity' lines found in the .zub file.")
    return table


def format_zub(entries, title):
    """Text of a .zub file, sorted by name like vox_man.zub."""
    lines = [f"== V4.3 Code Section 1 {title}"]
    for e in sorted(entries, key=lambda e: e.name.lower()):
        lines.append(f"{e.name:<28s}{e.index:5d}{e.density:6d}{e.activity:6d}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# HU -> density calibration (ported from ct2dmi.py)
# --------------------------------------------------------------------------
def load_curve(path=None):
    """Return (hu_points, rho_points, description). CSV with header hu,rho."""
    if not path:
        pts, src = list(GENERIC_120KVP), "generic_120kVp (simplified Schneider)"
    else:
        if not os.path.isfile(path):
            raise PhantomError(f"Calibration curve file not found: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
            cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
            if "hu" not in cols or "rho" not in cols:
                raise PhantomError("Calibration CSV must have columns 'hu' and 'rho'.")
            pts = [(float(r[cols["hu"]]), float(r[cols["rho"]])) for r in rd]
        src = os.path.abspath(path)
    pts.sort()
    if len(pts) < 2:
        raise PhantomError("The calibration curve needs at least 2 points.")
    hu, rho = (np.array(v, dtype=np.float64) for v in zip(*pts))
    if np.any(np.diff(hu) <= 0):
        raise PhantomError("Repeated HU values in the calibration curve.")
    if np.any(np.diff(rho) < 0):
        raise PhantomError("The calibration curve must be monotonic non-decreasing.")
    if rho.min() < 0:
        raise PhantomError("Negative densities in the calibration curve.")
    if rho.max() > INT16_MAX / 1000.0:
        raise PhantomError("Maximum density exceeds the int16/1000 range.")
    return hu, rho, src


def hu_to_density(hu, curve):
    """Piecewise-linear interpolation; saturates outside the curve range."""
    return np.interp(hu, curve[0], curve[1]).astype(np.float32)


def sanity_check_hu(hu):
    """HU-only checks (from ct2dmi.py). Returns (checks dict, warnings list)."""
    out, warn = {}, []
    low = hu[hu < -950]
    if low.size > 1000:
        air = float(np.median(low))
        out["air_median_HU"] = air
        if abs(air + 1000) > 30:
            warn.append(f"Air (voxels < -950 HU) median is {air:.0f} HU (expected ~ -1000): "
                        "check RescaleIntercept / scanner offset.")
    soft = hu[(hu > -200) & (hu < 200)]
    if soft.size > 1000:
        h, e = np.histogram(soft, bins=np.arange(-200, 201, 4))
        c = 0.5 * (e[:-1] + e[1:])
        fat_m, mus_m = (c < -40), (c > -10)
        out["soft_peak_HU"] = float(c[mus_m][np.argmax(h[mus_m])])
        if h[fat_m].max() >= 0.02 * soft.size:
            out["fat_peak_HU"] = float(c[fat_m][np.argmax(h[fat_m])])
            if not (-130 <= out["fat_peak_HU"] <= -60):
                warn.append(f"Fat peak at {out['fat_peak_HU']:.0f} HU (expected ~ -100).")
    out["max_HU"] = float(hu.max())
    if hu.max() > 3071:
        warn.append(f"Max HU {hu.max():.0f}: probable metal artifacts; density saturates.")
    return out, warn


# --------------------------------------------------------------------------
# Structure map (one id per structure row, at CT resolution)
# --------------------------------------------------------------------------
def paint_order(items):
    """items: list of (key, role, voxel_count). Returns keys in painting order.
    Later painting overwrites earlier, so: large organs first, small organs later,
    lesions always last (smaller lesions after larger ones)."""
    organs = [i for i in items if i[1] == ROLE_ORGAN]
    lesions = [i for i in items if i[1] == ROLE_LESION]
    organs.sort(key=lambda i: -i[2])
    lesions.sort(key=lambda i: -i[2])
    return [i[0] for i in organs + lesions]


def body_mask_from_hu(hu, threshold=-500.0):
    """Body mask: HU > threshold, largest connected component, holes filled per axial slice."""
    m = (hu > threshold).astype(np.uint8)
    if not m.any():
        return m.astype(bool)
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(m))
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    body = sitk.GetArrayFromImage(cc) == 1
    out = np.empty_like(body)
    for z in range(body.shape[0]):
        sl = sitk.GetImageFromArray(body[z].astype(np.uint8))
        out[z] = sitk.GetArrayFromImage(sitk.BinaryFillhole(sl)) > 0
    return out


def fill_background(struct, hu, body, class_ids, thresholds=None):
    """Assign HU classes to body voxels not covered by any structure (struct == 0).
    class_ids: dict with keys 'gas','fat','soft','bone' -> structure id (0 = skip)."""
    t = dict(DEFAULT_THRESHOLDS)
    t.update(thresholds or {})
    if not (t["gas_max"] < t["fat_max"] < t["soft_max"]):
        raise PhantomError("Background thresholds must satisfy gas_max < fat_max < soft_max.")
    free = body & (struct == 0)
    classes = [
        ("gas", hu < t["gas_max"]),
        ("fat", (hu >= t["gas_max"]) & (hu < t["fat_max"])),
        ("soft", (hu >= t["fat_max"]) & (hu <= t["soft_max"])),
        ("bone", hu > t["soft_max"]),
    ]
    for key, cond in classes:
        sid = class_ids.get(key, 0)
        if sid:
            struct[free & cond] = sid
    return struct


def outer_shell(mask):
    """1-voxel outer shell of a 3D boolean mask (6-connected erosion)."""
    m = mask.astype(bool)
    er = m.copy()
    er[1:] &= m[:-1]; er[:-1] &= m[1:]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    er[:, :, 1:] &= m[:, :, :-1]; er[:, :, :-1] &= m[:, :, 1:]
    # voxels on the array border are always shell
    er[0] = er[-1] = False
    er[:, 0] = er[:, -1] = False
    er[:, :, 0] = er[:, :, -1] = False
    return m & ~er


def add_skin(struct, skin_id, protected_ids=(), body=None):
    """Put skin_id on the outer shell of the body mask (default: struct > 0),
    except on protected ids (e.g. lesions)."""
    shell = outer_shell(struct > 0 if body is None else body)
    if protected_ids:
        shell &= ~np.isin(struct, list(protected_ids))
    struct[shell] = skin_id
    return struct


def lookup(struct, lut, dtype):
    """Map structure ids through a lookup table (numpy array indexed by id)."""
    lut = np.asarray(lut)
    if struct.size and struct.max() >= lut.size:
        raise PhantomError("Structure id outside lookup table.")
    return lut.astype(dtype)[struct]


# --------------------------------------------------------------------------
# Output grid and resampling (ported from ct2dmi.py)
# --------------------------------------------------------------------------
@dataclass
class Grid:
    size: tuple          # (nx, ny, nz)
    spacing: tuple       # mm
    origin: tuple        # LPS mm
    direction: tuple     # 9 values

    def reference_image(self):
        img = sitk.Image([int(v) for v in self.size], sitk.sitkUInt8)
        img.SetSpacing([float(v) for v in self.spacing])
        img.SetOrigin([float(v) for v in self.origin])
        img.SetDirection([float(v) for v in self.direction])
        return img


def parse_triplet(text, cast, name):
    parts = [p for p in str(text).replace(";", ",").replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise PhantomError(f"{name} must be one value or three (x,y,z).")
    vals = tuple(cast(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise PhantomError(f"{name} values must be positive.")
    return vals


def grid_from_image(img):
    return Grid(tuple(img.GetSize()), tuple(img.GetSpacing()), tuple(img.GetOrigin()),
                tuple(img.GetDirection()))


def make_grid(img, spacing=None, matrix=None, center_mask=None, margin_mm=0.0):
    """Output grid with `spacing` (mm) and optional `matrix` (nx,ny,nz), on the
    direction of `img`.  Centered on the image center, or on the bounding box of
    `center_mask` (z,y,x bool array on img grid) when given.  Without matrix the
    grid covers the image FOV (or the mask bounding box + margin)."""
    old_sp = np.array(img.GetSpacing(), dtype=float)
    old_size = np.array(img.GetSize(), dtype=float)
    new_sp = old_sp.copy() if spacing is None else np.array(spacing, dtype=float)
    D = np.array(img.GetDirection()).reshape(3, 3)
    org = np.array(img.GetOrigin())

    if center_mask is not None and np.any(center_mask):
        zyx = np.argwhere(center_mask)
        lo = zyx.min(0)[::-1].astype(float)   # -> x,y,z index
        hi = zyx.max(0)[::-1].astype(float)
        c_idx = (lo + hi) / 2.0
        extent = (hi - lo + 1) * old_sp + 2 * margin_mm
    else:
        c_idx = (old_size - 1) / 2.0
        extent = old_size * old_sp
    centre = org + D @ (c_idx * old_sp)

    if matrix is None:
        new_size = np.maximum(np.ceil(extent / new_sp - 1e-6).astype(int), 1)
    else:
        new_size = np.array(matrix, dtype=int)
    new_org = centre - D @ ((new_size - 1) * new_sp / 2.0)
    return Grid(tuple(int(v) for v in new_size), tuple(float(v) for v in new_sp),
                tuple(float(v) for v in new_org), tuple(img.GetDirection()))


def antialias(img, new_spacing):
    """Per-axis Gaussian smoothing before down-sampling (sigma in mm)."""
    old_sp = np.array(img.GetSpacing())
    new_sp = np.array(new_spacing, dtype=float)
    sigma = np.where(new_sp > old_sp * 1.01,
                     0.5 * np.sqrt(np.maximum(new_sp ** 2 - old_sp ** 2, 0)), 0.0)
    if np.any(sigma > 0):
        g = sitk.RecursiveGaussianImageFilter()
        g.SetOrder(0)
        for ax in range(3):
            if sigma[ax] > 0:
                g.SetSigma(float(sigma[ax]))
                g.SetDirection(ax)
                img = g.Execute(img)
    return img


def resample(img, grid, interpolator=sitk.sitkNearestNeighbor, default=0.0, smooth=False):
    if smooth:
        img = antialias(sitk.Cast(img, sitk.sitkFloat32), grid.spacing)
    return sitk.Resample(img, grid.reference_image(), sitk.Transform(), interpolator,
                         float(default), img.GetPixelID())


def image_like(arr, ref):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref)
    return img


# --------------------------------------------------------------------------
# Writing SIMIND raw files
# --------------------------------------------------------------------------
def reorder_for_simind(arr_zyx, first_slice="head", flip_x=False, flip_y=False):
    """Input on an LPS grid (z index grows toward the head). SIMIND files are
    written slice by slice; 'head' puts the cranial slice first."""
    a = arr_zyx
    if first_slice == "head":
        a = a[::-1, :, :]
    if flip_x:
        a = a[:, :, ::-1]
    if flip_y:
        a = a[:, ::-1, :]
    return np.ascontiguousarray(a)


def write_raw(path, arr, dtype):
    """Write raw little-endian, x fastest, then y, then z. Verifies by re-reading.
    Returns sha256 of the file."""
    dt = np.dtype(dtype).newbyteorder("<")
    info = np.iinfo(dt)
    if arr.size and (arr.min() < info.min or arr.max() > info.max):
        raise PhantomError(f"Values {arr.min()}..{arr.max()} do not fit in {dt.name} "
                           f"({info.min}..{info.max}); nothing was written to {path}.")
    out = np.ascontiguousarray(arr.astype(dt))
    out.tofile(path)
    back = np.fromfile(path, dtype=dt)
    if back.size != out.size or not np.array_equal(back.reshape(out.shape), out):
        raise PhantomError(f"Read-back verification failed for {path}.")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Lesion helpers
# --------------------------------------------------------------------------
def split_components(mask, min_voxels=1):
    """Split a boolean mask into connected components (26-connectivity),
    largest first, dropping components smaller than min_voxels."""
    if not np.any(mask):
        return []
    f = sitk.ConnectedComponentImageFilter()
    f.SetFullyConnected(True)
    cc = f.Execute(sitk.GetImageFromArray(mask.astype(np.uint8)))
    cc = sitk.RelabelComponent(cc, minimumObjectSize=int(max(min_voxels, 1)),
                               sortByObjectSize=True)
    lab = sitk.GetArrayFromImage(cc)
    return [lab == i for i in range(1, int(lab.max()) + 1)]


def threshold_lesions(func, roi, fraction=0.4, min_voxels=1, split=True):
    """Hot-spot lesions on a functional image (SPECT/PET) already on the CT grid:
    voxels >= fraction * max(func within roi), inside roi."""
    if not 0 < fraction <= 1:
        raise PhantomError("Threshold fraction must be in (0, 1].")
    roi = roi.astype(bool)
    if not roi.any():
        raise PhantomError("The region of interest for lesion search is empty.")
    vmax = float(func[roi].max())
    if vmax <= 0:
        raise PhantomError("The functional image has no positive values inside the region.")
    m = roi & (func >= fraction * vmax)
    if split:
        return split_components(m, min_voxels)
    return [m] if m.sum() >= min_voxels else []


def ellipsoid_mask(shape, center_zyx, radius_mm, spacing_zyx):
    """Boolean ball of radius_mm (physical) around center_zyx on an anisotropic grid."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [float(v) for v in center_zyx]
    s = [float(v) for v in spacing_zyx]
    d2 = ((z - c[0]) * s[0]) ** 2 + ((y - c[1]) * s[1]) ** 2 + ((x - c[2]) * s[2]) ** 2
    return d2 <= float(radius_mm) ** 2


def seeded_region(hu, seed_zyx, tolerance=50.0, roi=None, radius_vox=3):
    """Region grown from a seed: connected voxels whose HU is within
    mean(seed neighbourhood) +/- tolerance, restricted to roi."""
    z, y, x = (int(round(v)) for v in seed_zyx)
    if not (0 <= z < hu.shape[0] and 0 <= y < hu.shape[1] and 0 <= x < hu.shape[2]):
        raise PhantomError("Seed point is outside the CT volume.")
    r = int(radius_vox)
    nb = hu[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    mean = float(np.mean(nb))
    work = hu.astype(np.float32).copy()
    if roi is not None:
        work[~roi.astype(bool)] = -1e6  # never inside the HU window
    seg = sitk.ConnectedThreshold(sitk.GetImageFromArray(work), seedList=[(x, y, z)],
                                  lower=mean - tolerance, upper=mean + tolerance)
    return sitk.GetArrayFromImage(seg) > 0


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
PIP_HELP = """What to try:
 1. In the Slicer Python console, install GDCM from a pre-built wheel and then dicom2nifti:
      slicer.util.pip_install("--only-binary=python-gdcm python-gdcm")
      slicer.util.pip_install("dicom2nifti<=2.5.1")
 2. Check internet access / proxy (HTTPS_PROXY) and that the antivirus allows PythonSlicer.exe.
 3. In the TotalSegmentator module: Advanced > "Force reinstall".
Then click "Install / check TotalSegmentator Python packages" again."""


def _as_text(v):
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def format_pip_error(cmd, output=None, stderr=None, max_lines=30):
    """User message for a failed pip command (subprocess.CalledProcessError fields)."""
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(c) for c in cmd)
    text = "\n".join(t for t in (_as_text(output).strip(), _as_text(stderr).strip()) if t)
    lines = text.splitlines()
    msg = [f"Installing the TotalSegmentator Python packages failed.\nCommand: {cmd}"]
    if lines:
        tail = lines[-max_lines:]
        msg.append(f"Last {len(tail)} line(s) of pip output:\n" + "\n".join(tail))
    else:
        msg.append("pip printed no output here; see View > Error Log for the full pip messages.")
    joined = text.lower()
    if "python-gdcm" in joined or "python_gdcm" in joined:
        msg.append("Cause: pip tried to compile python-gdcm from source (no wheel for this Python).")
    msg.append(PIP_HELP)
    return "\n\n".join(msg)
