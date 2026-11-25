"""
Automatic / semi-automatic lesion segmentation methods.

Every method has the same interface so new ones can be added later:

    method.available(logic) -> (bool, reason)
    method.run(logic, ctNode, segNode, params) -> list of new segment ids

New segments are named "lesion_<n> (<source>)" so the default mapping rule
gives them Zubal index 63, role "lesion" (painted last) and their own activity row.
"""
import re

import numpy as np
import slicer
import vtk

from . import core

LESION_COLOR = (1.0, 0.85, 0.0)


def _min_voxels(ctNode, min_ml):
    vox_mm3 = float(np.prod(ctNode.GetSpacing()))
    return max(1, int(round(float(min_ml) * 1000.0 / vox_mm3)))


def _add_lesions(logic, ctNode, segNode, masks, source):
    ids = []
    n = logic.nextLesionNumber(segNode)
    for m in masks:
        if not np.any(m):
            continue
        ids.append(logic.addSegmentFromMask(segNode, ctNode, m, f"lesion_{n} ({source})", LESION_COLOR))
        n += 1
    return ids


def _split_or_whole(mask, split, min_vox):
    if split:
        return core.split_components(mask, min_vox)
    return [mask] if mask.sum() >= min_vox else []


class LesionMethod:
    key = ""
    title = ""

    def available(self, logic):
        return True, ""

    def run(self, logic, ctNode, segNode, params):
        raise NotImplementedError


class TotalSegmentatorLesions(LesionMethod):
    """TotalSegmentator sub-tasks that contain lesions."""
    key = "ts"
    title = "TotalSegmentator sub-task"
    # task -> (regex of the lesion segments in the task output, label)
    TASKS = {
        "liver_vessels": (r"tumou?r", "liver tumor"),
        "lung_nodules": (r"nodule", "lung nodules"),
        "kidney_cysts_from_total": (r"cyst", "kidney cysts (already in 'total')"),
    }

    def available(self, logic):
        if not logic.isTotalSegmentatorAvailable():
            return False, "TotalSegmentator extension is not installed."
        return True, ""

    def run(self, logic, ctNode, segNode, params):
        task = params.get("task", "liver_vessels")
        if task not in self.TASKS:
            raise core.PhantomError(f"Unknown TotalSegmentator lesion task '{task}'.")
        pattern, label = self.TASKS[task]
        split = params.get("split", True)
        min_vox = _min_voxels(ctNode, params.get("min_ml", 0.1))

        if task == "kidney_cysts_from_total":
            source = [(segNode, sid) for sid in segNode.GetSegmentation().GetSegmentIDs()
                      if re.search(pattern, core.normalize_name(
                          segNode.GetSegmentation().GetSegment(sid).GetName()))]
            masks = []
            for node, sid in source:
                masks += _split_or_whole(logic.segmentMask(node, sid, ctNode), split, min_vox)
            ids = _add_lesions(logic, ctNode, segNode, masks, "kidney cyst")
            if params.get("remove_source", True):
                for _, sid in source:
                    segNode.GetSegmentation().RemoveSegment(sid)
            return ids

        tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"_tmp_TS_{task}")
        try:
            logic.runTotalSegmentator(ctNode, tmp, task=task, fast=params.get("fast", False),
                                      cpu=params.get("cpu", False))
            seg = tmp.GetSegmentation()
            masks = []
            for sid in seg.GetSegmentIDs():
                if re.search(pattern, core.normalize_name(seg.GetSegment(sid).GetName())):
                    masks += _split_or_whole(logic.segmentMask(tmp, sid, ctNode), split, min_vox)
        finally:
            slicer.mrmlScene.RemoveNode(tmp)
        if not masks:
            raise core.PhantomError(f"TotalSegmentator task '{task}' found no {label}.")
        return _add_lesions(logic, ctNode, segNode, masks, label)


class ImportFromSegmentation(LesionMethod):
    """Lesions produced by another tool (MONAI Auto3DSeg, a manual segmentation, ...):
    segments whose name matches a regex are copied as lesions."""
    key = "import"
    title = "From another segmentation (MONAI Auto3DSeg, ...)"

    def run(self, logic, ctNode, segNode, params):
        src = params.get("source")
        if src is None:
            raise core.PhantomError("Select the segmentation that contains the lesions.")
        if src is segNode:
            raise core.PhantomError("Source and phantom segmentations must be different.")
        regex = params.get("regex") or r"tumou?r|lesion|nodule|metasta|cancer|mass"
        try:
            re.compile(regex)
        except re.error as e:
            raise core.PhantomError(f"Invalid name filter: {e}")
        split = params.get("split", True)
        min_vox = _min_voxels(ctNode, params.get("min_ml", 0.1))
        seg = src.GetSegmentation()
        masks = []
        for sid in seg.GetSegmentIDs():
            if re.search(regex, core.normalize_name(seg.GetSegment(sid).GetName())):
                masks += _split_or_whole(logic.segmentMask(src, sid, ctNode), split, min_vox)
        if not masks:
            raise core.PhantomError(f"No segment of '{src.GetName()}' matches '{regex}'.")
        return _add_lesions(logic, ctNode, segNode, masks, src.GetName())


class FunctionalThreshold(LesionMethod):
    """Hot spots of a SPECT/PET registered to the CT: >= fraction * max inside a ROI."""
    key = "functional"
    title = "Functional image threshold (SPECT/PET)"

    def run(self, logic, ctNode, segNode, params):
        func = params.get("functional")
        if func is None:
            raise core.PhantomError("Select the functional (SPECT/PET) volume.")
        logic.checkNoTransform(func, ctNode)
        roi = logic.roiMask(ctNode, segNode, params.get("roi_segment"))
        arr = logic.volumeOnCtGrid(func, ctNode)
        masks = core.threshold_lesions(arr, roi, float(params.get("fraction", 0.4)),
                                       _min_voxels(ctNode, params.get("min_ml", 0.5)),
                                       split=params.get("split", True))
        if not masks:
            raise core.PhantomError("No hot spot above the threshold and minimum volume.")
        return _add_lesions(logic, ctNode, segNode, masks,
                            f"{int(round(100 * float(params.get('fraction', 0.4))))}% {func.GetName()}")


class SeededRegion(LesionMethod):
    """Semi-automatic: one region per markup point, grown on the CT within
    mean(seed) +/- tolerance HU, inside a ROI and a maximum radius."""
    key = "seeded"
    title = "Seeded region growing (markup points)"

    def run(self, logic, ctNode, segNode, params):
        markups = params.get("markups")
        if markups is None or markups.GetNumberOfControlPoints() == 0:
            raise core.PhantomError("Place at least one markup point inside each lesion.")
        logic.checkNoTransform(ctNode)
        hu = slicer.util.arrayFromVolume(ctNode).astype(np.float32)
        roi = logic.roiMask(ctNode, segNode, params.get("roi_segment"))
        spacing_zyx = ctNode.GetSpacing()[::-1]
        rasToIjk = vtk.vtkMatrix4x4()
        ctNode.GetRASToIJKMatrix(rasToIjk)
        masks = []
        for i in range(markups.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            markups.GetNthControlPointPositionWorld(i, ras)
            ijk = rasToIjk.MultiplyPoint(list(ras) + [1.0])[:3]
            zyx = (ijk[2], ijk[1], ijk[0])
            m = core.seeded_region(hu, zyx, float(params.get("tolerance", 50.0)), roi, radius_vox=1)
            m &= core.ellipsoid_mask(hu.shape, zyx, float(params.get("max_radius_mm", 40.0)),
                                     spacing_zyx)
            if m.any():
                masks.append(m)
        if not masks:
            raise core.PhantomError("Region growing produced no voxels (seed outside ROI?).")
        return _add_lesions(logic, ctNode, segNode, masks, "seeded")


METHODS = [TotalSegmentatorLesions(), ImportFromSegmentation(), FunctionalThreshold(), SeededRegion()]
