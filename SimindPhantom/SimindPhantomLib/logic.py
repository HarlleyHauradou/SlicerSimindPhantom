"""
SimindPhantomLogic: everything that touches Slicer / MRML (no widgets here).

The numerical work is done by the Slicer-independent modules core.py and
pipeline.py; this class converts MRML nodes to arrays / SimpleITK images and back.
"""
import colorsys
import importlib
import inspect
import logging
import os
import subprocess

import numpy as np
import SimpleITK as sitk
import sitkUtils
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from . import core, pipeline
from . import zubal_table as zt

COLOR_TABLE_NAME = "SimindPhantom Zubal colors"


class SimindPhantomLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.rules = zt.default_rules()
        self.zubalTable = zt.default_zubal_table()
        self.zubalTableSource = zt.DEFAULT_ZUBAL_CSV
        self.lastResult = None
        self.tsRequirementsOk = False

    # ------------------------------------------------------------------ tables
    def loadMapping(self, path):
        self.rules = core.load_mapping_csv(path)

    def resetMapping(self):
        self.rules = zt.default_rules()

    def loadZubalTable(self, path):
        """Accepts a SIMIND .zub file (e.g. vox_man.zub) or a CSV like zubal_indices.csv."""
        if path.lower().endswith(".csv"):
            self.zubalTable = core.load_zubal_csv(path)
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                self.zubalTable = core.parse_zub(f.read())
        self.zubalTableSource = path

    def zubalName(self, index):
        return zt.zubal_name(self.zubalTable, index)

    # ------------------------------------------------------------------ checks
    @staticmethod
    def checkNoTransform(*nodes):
        for n in nodes:
            if n is not None and n.GetParentTransformNode() is not None:
                raise core.PhantomError(
                    f"'{n.GetName()}' is under a transform. Harden the transform first "
                    "(Data module, right-click > Harden transform).")

    @staticmethod
    def ctDicomInfo(ctNode):
        """Read a few DICOM tags of the first instance (if the CT came from the DICOM
        database) and return (info dict, warnings) - checks ported from ct2dmi.py."""
        info, warnings = {}, []
        uids = ctNode.GetAttribute("DICOM.instanceUIDs") if ctNode else None
        db = getattr(slicer, "dicomDatabase", None)
        if not uids or db is None:
            return info, warnings
        try:
            fileName = db.fileForInstance(uids.split()[0])
            tags = {"modality": "0008,0060", "manufacturer": "0008,0070", "model": "0008,1090",
                    "kvp": "0018,0060", "kernel": "0018,1210", "contrast_agent": "0018,0010",
                    "gantry_tilt": "0018,1120", "slice_thickness": "0018,0050"}
            for k, t in tags.items():
                info[k] = (db.fileValue(fileName, t) or "").strip()
        except Exception as e:  # never block the workflow because of metadata
            logging.warning(f"Could not read DICOM tags: {e}")
            return info, warnings
        if info.get("modality") and info["modality"] != "CT":
            warnings.append(f"Modality = {info['modality']} (CT expected).")
        if info.get("gantry_tilt") not in (None, "", "0", "0.0", "0.00"):
            warnings.append(f"Gantry tilt = {info['gantry_tilt']} deg: correct it before use.")
        if info.get("contrast_agent"):
            warnings.append(f"Contrast agent declared ({info['contrast_agent']}): soft-tissue HU "
                            "and density will be over-estimated where contrast is present.")
        try:
            kvp = float(info.get("kvp") or "nan")
            if not np.isnan(kvp) and not (100 <= kvp <= 140):
                warnings.append(f"kVp = {kvp:g}: the generic curve is for ~120 kVp; "
                                "use your scanner's calibration curve.")
        except ValueError:
            pass
        return info, warnings

    # ------------------------------------------------------------------ TotalSegmentator
    @staticmethod
    def isTotalSegmentatorAvailable():
        return hasattr(slicer.modules, "totalsegmentator")

    @staticmethod
    def isMonaiAuto3DSegAvailable():
        return hasattr(slicer.modules, "monaiauto3dseg")

    @staticmethod
    def installExtension(name):
        """Download and install an extension (Slicer >= 5.4). Returns True if a restart is needed."""
        em = slicer.app.extensionsManagerModel()
        if em.isExtensionInstalled(name):
            return False
        if not em.downloadAndInstallExtensionByName(name, True, True):
            raise core.PhantomError(f"Could not install the {name} extension "
                                    "(use View > Extensions Manager).")
        return True

    @staticmethod
    def _newTotalSegmentatorLogic():
        if not SimindPhantomLogic.isTotalSegmentatorAvailable():
            raise core.PhantomError("The TotalSegmentator extension is not installed.")
        from TotalSegmentator import TotalSegmentatorLogic
        tsLogic = TotalSegmentatorLogic()
        if hasattr(tsLogic, "useStandardSegmentNames"):
            tsLogic.useStandardSegmentNames = True
        tsLogic.logCallback = lambda msg: logging.info(f"[TotalSegmentator] {msg}")
        return tsLogic

    @staticmethod
    def _isImportable(module):
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            return False

    def ensureTotalSegmentatorRequirements(self, force=False):
        """Install the Python packages TotalSegmentator needs (once per session).

        pip is told to prefer pre-built wheels (PIP_PREFER_BINARY=1): otherwise it may pick
        the newest *source* release of a dependency (e.g. python-gdcm 3.2.6 for Python 3.9 on
        Windows) and fail trying to compile it with CMake."""
        if self.tsRequirementsOk and not force:
            return
        tsLogic = self._newTotalSegmentatorLogic()
        previous = os.environ.get("PIP_PREFER_BINARY")
        os.environ["PIP_PREFER_BINARY"] = "1"
        try:
            if not self._isImportable("dicom2nifti") and not self._isImportable("gdcm"):
                slicer.util.pip_install("--only-binary=python-gdcm python-gdcm")
            tsLogic.setupPythonRequirements()
        except subprocess.CalledProcessError as e:
            raise core.PhantomError(core.format_pip_error(e.cmd, e.output, e.stderr)) from e
        finally:
            if previous is None:
                os.environ.pop("PIP_PREFER_BINARY", None)
            else:
                os.environ["PIP_PREFER_BINARY"] = previous
        self.tsRequirementsOk = True

    def availableTotalSegmentatorTasks(self):
        """{task: {"title": str, "requiresLicense": bool}} offered by the installed extension."""
        try:
            tasks = getattr(self._newTotalSegmentatorLogic(), "tasks", {}) or {}
        except Exception:
            return {}
        return {k: {"title": v.get("title", k), "requiresLicense": bool(v.get("requiresLicense"))}
                for k, v in tasks.items()}

    def runTotalSegmentator(self, ctNode, segNode, task="total", quality="normal", cpu=False,
                            subset=None):
        """Run TotalSegmentator through the logic of the installed extension
        (works with the old `fast=` and the new `quality=` process() signatures)."""
        self.ensureTotalSegmentatorRequirements()
        tsLogic = self._newTotalSegmentatorLogic()
        params = inspect.signature(tsLogic.process).parameters
        kwargs = core.ts_process_kwargs(params, quality, cpu, task, subset)
        try:
            tsLogic.process(ctNode, segNode, **kwargs)
        except subprocess.CalledProcessError as e:
            raise core.PhantomError(
                f"TotalSegmentator ({task}) failed; see View > Error Log for details.\n"
                f"Command: {e.cmd if isinstance(e.cmd, str) else ' '.join(map(str, e.cmd))}") from e

    def runTotalSegmentatorInto(self, ctNode, segNode, task, quality="normal", cpu=False,
                                keep=None, rename=None):
        """Run a TotalSegmentator task into a temporary segmentation and copy its
        segments (optionally filtered by `keep(name)` and renamed by `rename(name)`)
        into segNode. Returns the list of new segment ids."""
        tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"_tmp_TS_{task}")
        try:
            self.runTotalSegmentator(ctNode, tmp, task=task, quality=quality, cpu=cpu)
            return self.copySegments(tmp, segNode, keep, rename)
        finally:
            slicer.mrmlScene.RemoveNode(tmp)

    @staticmethod
    def copySegments(srcNode, dstNode, keep=None, rename=None):
        src, dst = srcNode.GetSegmentation(), dstNode.GetSegmentation()
        newIds = []
        for i in range(src.GetNumberOfSegments()):
            sid = src.GetNthSegmentID(i)
            name = src.GetSegment(sid).GetName()
            if keep and not keep(name):
                continue
            before = set(dst.GetSegmentIDs())
            dst.CopySegmentFromSegmentation(src, sid, False)
            added = [s for s in dst.GetSegmentIDs() if s not in before]
            for s in added:
                if rename:
                    dst.GetSegment(s).SetName(rename(name))
                newIds.append(s)
        return newIds

    # ------------------------------------------------------------------ segments
    @staticmethod
    def segmentVoxelCount(segNode, segId):
        """Fast voxel count from the internal (possibly shared) labelmap."""
        try:
            seg = segNode.GetSegmentation()
            arr = slicer.util.arrayFromSegmentInternalBinaryLabelmap(segNode, segId)
            lv = seg.GetSegment(segId).GetLabelValue()
            lm = seg.GetSegment(segId).GetRepresentation(
                slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName())
            sp = lm.GetSpacing() if lm is not None else (1.0, 1.0, 1.0)
            n = int(np.count_nonzero(arr == lv)) if arr is not None else 0
            return n, n * float(np.prod(sp)) / 1000.0
        except Exception:
            return 0, 0.0

    def listSegments(self, segNode):
        """[(segment id, name, voxel count, volume cm3)]"""
        out = []
        if segNode is None:
            return out
        seg = segNode.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            n, cm3 = self.segmentVoxelCount(segNode, sid)
            out.append((sid, seg.GetSegment(sid).GetName(), n, cm3))
        return out

    def structureRows(self, segNode, includeBackground=True):
        segs = self.listSegments(segNode)
        rows = pipeline.rows_from_segments([(s[0], s[1], s[2]) for s in segs], self.rules,
                                           self.zubalTable, includeBackground)
        vols = {s[0]: s[3] for s in segs}
        for r in rows:
            r.volume_cm3 = vols.get(r.key, 0.0)
        return rows

    @staticmethod
    def segmentMask(segNode, segId, ctNode):
        return slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segId, ctNode) > 0

    @staticmethod
    def addSegmentFromMask(segNode, ctNode, mask, name, color=(1.0, 0.1, 0.1)):
        sid = segNode.GetSegmentation().AddEmptySegment("", name, list(color))
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), segNode, sid, ctNode)
        return sid

    @staticmethod
    def nextLesionNumber(segNode):
        seg = segNode.GetSegmentation()
        n = 0
        for i in range(seg.GetNumberOfSegments()):
            name = core.normalize_name(seg.GetSegment(seg.GetNthSegmentID(i)).GetName())
            if name.startswith("lesion"):
                n += 1
        return n + 1

    def roiMask(self, ctNode, segNode, segId=None):
        """Mask of a segment on the CT grid, or the body (HU) when segId is empty."""
        if segId:
            return self.segmentMask(segNode, segId, ctNode)
        return core.body_mask_from_hu(slicer.util.arrayFromVolume(ctNode).astype(np.float32))

    @staticmethod
    def volumeOnCtGrid(volumeNode, ctNode):
        """Resample any scalar volume (e.g. SPECT/PET) onto the CT voxel grid (physical space)."""
        img = sitkUtils.PullVolumeFromSlicer(volumeNode)
        ref = sitkUtils.PullVolumeFromSlicer(ctNode)
        out = sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), ref, sitk.Transform(),
                            sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(out)

    # ------------------------------------------------------------------ phantom
    def buildPhantom(self, ctNode, segNode, rows, options, referenceVolumeNode=None,
                     progress=None):
        if ctNode is None:
            raise core.PhantomError("Select the input CT volume.")
        if segNode is None:
            raise core.PhantomError("Select (or create with TotalSegmentator) a segmentation.")
        self.checkNoTransform(ctNode, segNode, referenceVolumeNode)
        if referenceVolumeNode is not None:
            ref = sitk.DICOMOrient(sitkUtils.PullVolumeFromSlicer(referenceVolumeNode), "LPS")
            options.reference_grid = core.grid_from_image(ref)
        else:
            options.reference_grid = None
        ctImg = sitkUtils.PullVolumeFromSlicer(ctNode)
        result = pipeline.build_phantom(ctImg, lambda key: self.segmentMask(segNode, key, ctNode),
                                        rows, options, progress)
        info, warns = self.ctDicomInfo(ctNode)
        result.input_info.update({"ct_node": ctNode.GetName(), "segmentation_node": segNode.GetName(),
                                  "dicom": info, "zubal_table": self.zubalTableSource})
        result.warnings[:0] = warns
        self.lastResult = result
        return result

    def exportZubal(self, result, outDir, baseName):
        return pipeline.export_zubal(result, outDir, baseName, self.zubalTable)

    def exportDensityActivity(self, result, outDir, baseName):
        return pipeline.export_density_activity(result, outDir, baseName)

    # ------------------------------------------------------------------ scene
    def zubalColorTable(self):
        node = slicer.mrmlScene.GetFirstNodeByName(COLOR_TABLE_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", COLOR_TABLE_NAME)
        node.SetTypeToUser()
        node.SetNumberOfColors(256)
        node.SetNamesInitialised(True)
        for idx in range(256):
            if idx == 0:
                node.SetColor(0, "outside phantom", 0.0, 0.0, 0.0, 0.0)
                continue
            r, g, b = colorsys.hsv_to_rgb((idx * 0.61803398875) % 1.0, 0.65, 0.95)
            name = self.zubalTable[idx].name if idx in self.zubalTable else f"index {idx}"
            node.SetColor(idx, name, r, g, b, 1.0)
        return node

    @staticmethod
    def _pushArray(arr, grid, name, className):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(grid.spacing)
        img.SetOrigin(grid.origin)
        img.SetDirection(grid.direction)
        target = slicer.mrmlScene.GetFirstNodeByName(name)
        if target is not None and not target.IsA(className):
            target = None
        return sitkUtils.PushVolumeToSlicer(img, targetNode=target, name=name, className=className)

    def showZubal(self, result, baseName):
        node = self._pushArray(result.zubal.astype(np.int16), result.grid, f"{baseName}_zubal",
                               "vtkMRMLLabelMapVolumeNode")
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetAndObserveColorNodeID(self.zubalColorTable().GetID())
        slicer.util.setSliceViewerLayers(label=node)
        return node

    def showDensityActivity(self, result, baseName):
        nodes = []
        if result.density is not None:
            d = self._pushArray(result.density, result.grid, f"{baseName}_density",
                                "vtkMRMLScalarVolumeNode")
            nodes.append(d)
        a = self._pushArray(result.activity.astype(np.float32), result.grid, f"{baseName}_activity",
                            "vtkMRMLScalarVolumeNode")
        nodes.append(a)
        a.CreateDefaultDisplayNodes()
        a.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileHotToColdRainbow.txt")
        if result.density is not None:
            slicer.util.setSliceViewerLayers(background=nodes[0], foreground=a, foregroundOpacity=0.5)
        return nodes
