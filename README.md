# SlicerSimindPhantom

3D Slicer extension that builds patient-specific voxel phantoms for the **SIMIND** Monte Carlo
code, entirely inside Slicer:

```
CT (HU) ──► TotalSegmentator ──► optional lesions ──► structure table (Zubal index + activity)
                                                        ├─► Zubal-type phantom  .dat (uint8) + .zub
                                                        └─► density .dmi (int16) + activity .smi (int16)
```

It replaces the manual workflow of the course *Herramientas Computacionales para Dosimetría
Interna* (FCE-UNLP, Módulo 4). That workflow used console scripts to find the segment numbers,
a hand-typed relabel dictionary, Fiji for the 8-bit conversion, and `ct2dmi.py` for the density
map. The module improves on it in these ways:

- **Mapping by segment name, not by segment order.** `liver` gets index 12 whatever position the
  segment has. The rules are regular expressions in an editable CSV, and the rows are editable in
  the table.
- **No rescaling surprises.** Raw files are written with numpy (little-endian, x fastest) and
  verified by reading them back. Values that do not fit the data type raise an error instead of
  being clipped.
- **`.dmi` and `.smi` always share the same grid.** The activity map is derived from the same
  resampled structure map used for the density map.
- **Unsegmented body voxels are filled by HU class:** gas 24, fat 22, soft tissue 9, bone 8. An
  optional 1-voxel skin shell gets index 1.
- **Lesions** can be added automatically (see below). They always win over the organ that
  contains them and get their own activity.

## Installation

1. Install the **TotalSegmentator** extension (Extensions Manager). The module also offers an
   install button.
2. Clone this repository.
3. In Slicer, open *Edit → Application Settings → Modules → Additional module paths*, add the
   `SimindPhantom` folder of the clone, and restart Slicer.
4. The module appears under **Nuclear Medicine → SIMIND Phantom**.

The UI lives in `SimindPhantom/Resources/UI/SimindPhantom.ui` and can be edited with Qt Designer
(`Slicer --designer`).

## Workflow (module sections)

| # | Section | What it does |
|---|---------|--------------|
| 1 | Input CT | Select the CT. DICOM checks: modality, kVp, contrast, gantry tilt. |
| 2 | Automatic segmentation | Runs TotalSegmentator task `total` (quality normal / fast / faster) and optionally `body`. The body segments are merged into a single `body` segment with all other segments subtracted, so `body` holds only the remaining tissue and nothing overlaps (button **Subtract all segments from 'body'** to redo it after editing). Opens the Segment Editor for manual edits. |
| 3 | Lesions (optional) | See the lesion methods below. |
| 4 | Body background | HU thresholds for the gas / fat / soft tissue / bone fill, the skin shell, and the body mask threshold. |
| 5 | Output grid | Voxel size, matrix, centering on the body, slice order, and flips. Alternatively, an exact reference grid (e.g. the reconstructed SPECT). |
| 6 | Structures | One row per segment plus the background classes. Edit the **Zubal index** and **Activity** columns. Yellow rows had no mapping rule. Load/save the mapping, load your own `vox_man.zub`, and load/save activity presets. |
| 7 | Export | Tab **Zubal (.dat)** writes `.dat` and `.zub`. Tab **Density + Activity (.dmi/.smi)** writes `.dmi` and `.smi` using the HU→ρ curve (generic 120 kVp or your own CSV `hu,rho`). Every export also writes a `_info.json` (geometry, mapping, sha256, warnings) and can load the result back into the scene to check orientation. |

### Lesion methods (`SimindPhantomLib/lesion.py`)

- **TotalSegmentator sub-task**
  - `liver_vessels` gives liver tumor.
  - `lung_nodules` gives lung nodules.
  - Kidney cysts from `total` can be converted into lesions.
- **From another segmentation.** Copies segments whose name matches a regex. Use this for MONAI
  Auto3DSeg results or any manual or external segmentation. A button opens MONAI Auto3DSeg if it
  is installed.
- **Functional image threshold (SPECT/PET).** Finds voxels at or above *x %* of the maximum
  inside an organ (or the body) of a functional image registered to the CT. Each connected island
  becomes one lesion.
- **Seeded region growing.** You place one markup point per lesion. The region grows within the
  seed's mean HU ± a tolerance, inside an organ and a maximum radius.

New lesions are named `lesion_<n> (<source>)`. They therefore get Zubal index 63 and role
`lesion`. To add a new method, subclass `LesionMethod` and append it to `METHODS`.

## Output formats

| File | Type | Content |
|------|------|---------|
| `<base>.dat` | uint8 raw | Zubal indices, x fastest, then y, then z, first slice = head (default) |
| `<base>.zub` | text | `name index density(mg/cm3) activity`, one line per index present |
| `<base>.dmi` | int16 LE raw | density [g/cm³] × 1000 |
| `<base>.smi` | int16 LE raw | relative activity per voxel (0–32767) |
| `<base>_*_info.json` | JSON | matrix, voxel (mm and cm for the SIMIND `.smc`), orientation, structures, curve, warnings |

## Default tables (`SimindPhantom/Resources/Data`)

- `ts_to_zubal.csv`: regex → Zubal index rules. The first match wins. They work with both
  TotalSegmentator ids (`kidney_right`) and standard names (`right kidney`).
- `zubal_indices.csv`: index, name, and density. Rows marked `slide` come from the course's
  `vox_man.zub`. Rows marked `assumed` are typical values. **Check them against your own
  `vox_man.zub`**, or load it with *Load .zub table*.
- `generic_120kVp_curve.csv`: simplified Schneider curve, the same one used by `ct2dmi.py`. Use
  the curve measured on your own scanner when possible.

## Troubleshooting

### `pip install dicom2nifti` fails (python-gdcm / CMake error)

On Slicer 5.8 for Windows (Python 3.9), pip can pick the newest **source** release of
`python-gdcm`. That release has no pre-built wheel for this Python, so pip tries to compile it
and fails with `CMake Error: ... does not appear to contain CMakeLists.txt`.

The **Install / check TotalSegmentator Python packages** button tells pip to prefer pre-built
wheels, which avoids this. To fix it by hand, run in the Slicer Python console:

```python
slicer.util.pip_install("--only-binary=python-gdcm python-gdcm")
slicer.util.pip_install("dicom2nifti<=2.5.1")
```

Then click the button again. If something still fails, the status box shows the end of the pip
output, and the full messages are in *View → Error Log*.

## Development

```
SimindPhantom/SimindPhantom.py          module + widget (loads the .ui, connects signals) + self-test
SimindPhantom/SimindPhantomLib/logic.py Slicer/MRML logic (no widgets)
SimindPhantom/SimindPhantomLib/lesion.py lesion methods
SimindPhantom/SimindPhantomLib/pipeline.py phantom pipeline (Slicer-independent)
SimindPhantom/SimindPhantomLib/core.py  numpy/SimpleITK functions (Slicer-independent)
tests/                                  pytest for core + pipeline
```

To run the tests outside Slicer (they need numpy, SimpleITK, and pytest):

```bash
python -m pytest tests
```

To run the self-test inside Slicer, which uses synthetic data and does not need TotalSegmentator,
enable *Developer mode* and click **Reload and Test** in the module.
