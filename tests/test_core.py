"""Tests for the Slicer-independent part (run: python -m pytest tests)."""
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SimindPhantom"))

from SimindPhantomLib import core, pipeline  # noqa: E402
from SimindPhantomLib import zubal_table as zt  # noqa: E402


# ---------------------------------------------------------------- helpers
def ball(shape, center, radius):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius ** 2


def synthetic_ct(shape=(40, 60, 60), spacing=(2.0, 2.0, 2.5)):
    """Cylinder 'body' of soft tissue with a fat layer, a 'liver' ball, a 'kidney'
    ball, a lesion inside the liver, a gas pocket, and air outside."""
    hu = np.full(shape, -1000.0, dtype=np.float32)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt((y - 30) ** 2 + (x - 30) ** 2)
    body = (r <= 25) & (z >= 2) & (z < shape[0] - 2)
    hu[body] = 40.0
    hu[body & (r > 22)] = -100.0                    # fat ring
    liver = ball(shape, (20, 25, 20), 9)
    kidney = ball(shape, (20, 38, 40), 5)
    lesion = ball(shape, (20, 25, 18), 3)
    gas = ball(shape, (10, 35, 25), 2)
    bone = ball(shape, (30, 40, 30), 3)
    hu[liver] = 60.0
    hu[kidney] = 30.0
    hu[gas] = -800.0
    hu[bone] = 700.0
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing(spacing)
    masks = {"seg_liver": liver, "seg_kidney": kidney, "seg_lesion": lesion}
    return img, masks, body


def default_rows(masks):
    rules = zt.default_rules()
    names = {"seg_liver": "liver", "seg_kidney": "kidney_right", "seg_lesion": "liver lesion"}
    segs = [(k, names[k], int(m.sum())) for k, m in masks.items()]
    return pipeline.rows_from_segments(segs, rules)


# ---------------------------------------------------------------- mapping
@pytest.mark.parametrize("name,index,role", [
    ("liver", 12, "organ"), ("Liver", 12, "organ"), ("kidney_right", 14, "organ"),
    ("right kidney", 14, "organ"), ("left adrenal gland", 21, "organ"),
    ("adrenal_gland_left", 21, "organ"), ("kidney_cyst_left", 14, "organ"),
    ("superior lobe of left lung", 10, "organ"), ("lung_upper_lobe_left", 10, "organ"),
    ("gallbladder", 13, "organ"), ("urinary_bladder", 40, "organ"),
    ("vertebrae_L1", 5, "organ"), ("L5 vertebra", 5, "organ"), ("sacrum", 5, "organ"),
    ("rib_left_7", 6, "organ"), ("left rib 7", 6, "organ"), ("sternum", 6, "organ"),
    ("costal_cartilages", 30, "organ"), ("hip_left", 7, "organ"), ("femur_right", 8, "organ"),
    ("heart", 11, "organ"), ("aorta", 23, "organ"), ("inferior_vena_cava", 23, "organ"),
    ("portal vein and splenic vein", 23, "organ"), ("spleen", 31, "organ"),
    ("small_bowel", 18, "organ"), ("duodenum", 18, "organ"), ("thyroid_gland", 28, "organ"),
    ("liver lesion", 63, "lesion"), ("liver_tumor", 63, "lesion"), ("lung_nodules", 63, "lesion"),
    ("body", 0, "body"), ("body_trunc", 0, "body"), ("gluteus_maximus_left", 9, "organ"),
    ("autochthon_right", 9, "organ"), ("spinal_cord", 3, "organ"), ("brain", 2, "organ"),
])
def test_default_mapping(name, index, role):
    rule = core.match_rule(name, zt.default_rules())
    assert rule is not None, name
    assert (rule.zubal_index, rule.role) == (index, role)


def test_unknown_name_flagged():
    rows = pipeline.rows_from_segments([("s1", "mystery thing", 10)], zt.default_rules(),
                                       include_background=False)
    assert rows[0].known is False and rows[0].zubal_index == zt.UNKNOWN_DEFAULT_INDEX


def test_mapping_csv_roundtrip(tmp_path):
    rules = zt.default_rules()
    p = tmp_path / "m.csv"
    core.save_mapping_csv(p, rules)
    back = core.load_mapping_csv(p)
    assert [(r.pattern, r.zubal_index, r.role) for r in back] == \
           [(r.pattern, r.zubal_index, r.role) for r in rules]


# ---------------------------------------------------------------- .zub
def test_zub_roundtrip():
    table = zt.default_zubal_table()
    text = core.format_zub(table.values(), "test")
    back = core.parse_zub(text)
    assert set(back) == set(table)
    assert back[12].name == "liver" and back[12].density == 1060


def test_parse_zub_like_slide():
    text = """== V4.3 Code Section 1 vox_man
adrenals                 21  1025    0
blood pool               23  1060  358
liver                    12  1060  808
outside phantom           0     0    0
"""
    t = core.parse_zub(text)
    assert t[12].activity == 808 and t[23].name == "blood pool" and t[0].density == 0


# ---------------------------------------------------------------- curve
def test_curve_validation(tmp_path):
    hu, rho, _ = core.load_curve(None)
    assert core.hu_to_density(np.array([0.0]), (hu, rho))[0] == pytest.approx(1.0)
    assert core.hu_to_density(np.array([5000.0]), (hu, rho))[0] == pytest.approx(1.92)
    bad = tmp_path / "bad.csv"
    bad.write_text("hu,rho\n0,1.0\n100,0.9\n")
    with pytest.raises(core.PhantomError):
        core.load_curve(str(bad))
    assert core.load_curve(zt.DEFAULT_CURVE_CSV)[1].tolist() == rho.tolist()


# ---------------------------------------------------------------- painting
def test_paint_order_lesions_last_small_last():
    items = [("big", "organ", 100), ("les_big", "lesion", 9), ("small", "organ", 10),
             ("les_small", "lesion", 2)]
    assert core.paint_order(items) == ["big", "small", "les_big", "les_small"]


def test_fill_background_classes():
    hu = np.array([[[-900, -100, 40, 500, 40]]], dtype=np.float32)
    struct = np.array([[[0, 0, 0, 0, 7]]], dtype=np.uint16)
    body = np.ones_like(struct, dtype=bool)
    core.fill_background(struct, hu, body, {"gas": 1, "fat": 2, "soft": 3, "bone": 4})
    assert struct.ravel().tolist() == [1, 2, 3, 4, 7]


def test_outer_shell():
    m = np.zeros((7, 7, 7), bool)
    m[1:6, 1:6, 1:6] = True
    s = core.outer_shell(m)
    assert s.sum() == 125 - 27 and not s[3, 3, 3]


def test_write_raw_guard_and_order(tmp_path):
    p = tmp_path / "a.dat"
    with pytest.raises(core.PhantomError):
        core.write_raw(p, np.array([[[256]]]), np.uint8)
    assert not p.exists()
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)   # z,y,x
    core.write_raw(p, arr, np.int16)
    raw = np.fromfile(p, dtype="<i2")
    assert raw[:4].tolist() == [0, 1, 2, 3]            # x fastest
    assert open(p, "rb").read(4) == b"\x00\x00\x01\x00"  # little endian


def test_reorder_head_first():
    a = np.arange(3)[:, None, None] * np.ones((3, 2, 2))
    assert core.reorder_for_simind(a, "head")[0, 0, 0] == 2
    assert core.reorder_for_simind(a, "feet")[0, 0, 0] == 0


# ---------------------------------------------------------------- lesions
def test_threshold_lesions_two_spheres():
    shape = (30, 30, 30)
    func = np.ones(shape, np.float32)
    func[ball(shape, (10, 10, 10), 3)] = 100
    func[ball(shape, (20, 20, 20), 2)] = 80
    roi = np.ones(shape, bool)
    les = core.threshold_lesions(func, roi, 0.4, min_voxels=5)
    assert len(les) == 2
    assert les[0].sum() == ball(shape, (10, 10, 10), 3).sum()
    assert les[1].sum() == ball(shape, (20, 20, 20), 2).sum()
    # min size removes the small one
    assert len(core.threshold_lesions(func, roi, 0.4, min_voxels=50)) == 1


def test_seeded_region():
    shape = (30, 30, 30)
    hu = np.full(shape, 50.0, np.float32)
    les = ball(shape, (15, 15, 15), 4)
    hu[les] = 150.0
    m = core.seeded_region(hu, (15, 15, 15), tolerance=30, radius_vox=1)
    assert np.array_equal(m, les)


# ---------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path):
    ct, masks, body = synthetic_ct()
    rows = default_rows(masks)
    by_key = {r.key: r for r in rows}
    by_key["seg_liver"].activity = 100
    by_key["seg_lesion"].activity = 800
    by_key["seg_kidney"].activity = 50
    by_key["bg:soft"].activity = 5
    opts = pipeline.PhantomOptions(spacing=(4.0, 4.0, 5.0), matrix=(28, 28, 22),
                                   crop_to_body=True, margin_mm=0)
    res = pipeline.build_phantom(ct, lambda k: masks[k], rows, opts)

    assert res.zubal.shape == (22, 28, 28) == res.activity.shape == res.density.shape
    vals = set(np.unique(res.zubal).tolist())
    assert {0, 12, 14, 63, 1, 9, 22}.issubset(vals), vals
    # lesion wins over liver, and keeps its activity
    assert by_key["seg_lesion"].voxels == masks["seg_lesion"].sum()
    assert by_key["seg_liver"].voxels == masks["seg_liver"].sum() - masks["seg_lesion"].sum()
    assert np.all(res.activity[res.zubal == 63] == 800)
    assert np.all(res.activity[res.zubal == 12] == 100)
    assert np.all(res.density[res.zubal == 0] == 0)
    liver_rho = res.density[res.zubal == 12]
    assert 1.0 < np.median(liver_rho) < 1.1

    files = pipeline.export_zubal(res, str(tmp_path), "ph", zt.default_zubal_table())
    dat = np.fromfile(files["dat"], dtype=np.uint8)
    assert dat.size == 28 * 28 * 22
    # head first: first slice in file is the last LPS slice
    assert np.array_equal(dat.reshape(22, 28, 28)[0], res.zubal[-1])
    zub = core.parse_zub(open(files["zub"]).read())
    assert set(zub) == vals and zub[63].activity == 800

    f2 = pipeline.export_density_activity(res, str(tmp_path), "ph")
    dmi = np.fromfile(f2["dmi"], dtype="<i2").reshape(22, 28, 28)
    smi = np.fromfile(f2["smi"], dtype="<i2").reshape(22, 28, 28)
    assert dmi.shape == smi.shape
    assert np.array_equal(smi, res.activity[::-1])
    assert np.array_equal(dmi > 0, res.zubal[::-1] > 0)
    assert os.path.isfile(f2["info"]) and os.path.isfile(files["info"])


def test_pipeline_reference_grid_and_truncation_warning():
    ct, masks, _ = synthetic_ct()
    rows = default_rows(masks)
    grid = core.Grid((10, 10, 8), (3.0, 3.0, 3.0), (20.0, 20.0, 20.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    res = pipeline.build_phantom(ct, lambda k: masks[k], rows,
                                 pipeline.PhantomOptions(reference_grid=grid))
    assert res.zubal.shape == (8, 10, 10)
    res2 = pipeline.build_phantom(ct, lambda k: masks[k], default_rows(masks),
                                  pipeline.PhantomOptions(spacing=(4, 4, 4), matrix=(4, 4, 4)))
    assert any("truncated" in w for w in res2.warnings)


def test_invalid_activity_rejected():
    ct, masks, _ = synthetic_ct()
    rows = default_rows(masks)
    rows[0].activity = 40000
    with pytest.raises(core.PhantomError):
        pipeline.build_phantom(ct, lambda k: masks[k], rows, pipeline.PhantomOptions())


def test_ellipsoid_mask_anisotropic():
    m = core.ellipsoid_mask((21, 21, 21), (10, 10, 10), 4.0, (2.0, 1.0, 1.0))
    assert m[10, 10, 14] and not m[10, 10, 15]
    assert m[12, 10, 10] and not m[13, 10, 10]


def test_activity_presets_roundtrip(tmp_path):
    _, masks, _ = synthetic_ct()
    rows = default_rows(masks)
    rows[0].activity = 123
    p = tmp_path / "act.csv"
    pipeline.save_activities(p, rows)
    fresh = default_rows(masks)
    assert pipeline.load_activities(p, fresh) == len(fresh)
    assert fresh[0].activity == 123


def test_orientation_independent():
    """Same anatomy stored with a flipped x/y direction must give the same phantom."""
    ct, masks, _ = synthetic_ct()
    opts = dict(spacing=(4.0, 4.0, 5.0), matrix=(28, 28, 22), crop_to_body=False)
    ref = pipeline.build_phantom(ct, lambda k: masks[k], default_rows(masks),
                                 pipeline.PhantomOptions(**opts))
    arr = sitk.GetArrayFromImage(ct)[:, ::-1, ::-1].copy()
    flipped = sitk.GetImageFromArray(arr)
    flipped.SetSpacing(ct.GetSpacing())
    size = np.array(ct.GetSize(), float)
    sp = np.array(ct.GetSpacing())
    flipped.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    flipped.SetOrigin(((size[0] - 1) * sp[0], (size[1] - 1) * sp[1], 0.0))
    fm = {k: m[:, ::-1, ::-1].copy() for k, m in masks.items()}
    res = pipeline.build_phantom(flipped, lambda k: fm[k], default_rows(fm),
                                 pipeline.PhantomOptions(**opts))
    assert np.array_equal(ref.zubal, res.zubal)


def test_format_pip_error():
    out = "\n".join(f"line {i}" for i in range(50)) + "\nUsing cached python_gdcm-3.2.6.tar.gz"
    msg = core.format_pip_error(["PythonSlicer.exe", "-m", "pip", "install", "dicom2nifti<=2.5.1"],
                                out.encode(), None, max_lines=5)
    assert "PythonSlicer.exe -m pip install dicom2nifti<=2.5.1" in msg
    assert "line 46" in msg and "line 45" not in msg
    assert "compile python-gdcm" in msg and "--only-binary=python-gdcm" in msg
    assert "Error Log" in core.format_pip_error("pip install x")


def test_ts_process_kwargs_signatures():
    import inspect

    class NewTS:  # Slicer >= 5.10 TotalSegmentator extension
        def process(self, inputVolume, outputSegmentation, quality=None, cpu=False, task=None,
                    subset=None, interactive=False, sequenceBrowserNode=None):
            return quality, cpu, task, subset

    class OldTS:
        def process(self, inputVolume, outputSegmentation, fast=True, cpu=False, task=None,
                    subset=None, interactive=False, sequenceBrowserNode=None):
            return fast, cpu, task, subset

    class OldestTS:
        def process(self, inputVolume, outputSegmentation, fast=True, task=None):
            return fast, task

    for cls, expect in ((NewTS, ("fast", True, "total", None)),
                        (OldTS, (True, True, "total", None)),
                        (OldestTS, (True, "total"))):
        obj = cls()
        kw = core.ts_process_kwargs(inspect.signature(obj.process).parameters, "fast", True, "total")
        assert obj.process(None, None, **kw) == expect
    kw = core.ts_process_kwargs(inspect.signature(NewTS().process).parameters, "normal")
    assert kw == {"task": "total", "quality": "normal", "cpu": False, "subset": None}
    with pytest.raises(core.PhantomError):
        core.ts_process_kwargs(["quality"], "ultra")
