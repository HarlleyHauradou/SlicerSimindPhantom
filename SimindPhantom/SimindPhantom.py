"""
SimindPhantom - 3D Slicer module.

CT -> TotalSegmentator -> (lesions) -> Zubal-type phantom (.dat + .zub)
                                    -> density (.dmi) + activity (.smi) maps for SIMIND.

This file only contains the module description, the widget (which loads
Resources/UI/SimindPhantom.ui and connects it to the logic) and the self-test.
All computations are in SimindPhantomLib (logic.py, lesion.py, pipeline.py, core.py).
"""
import copy
import os
import re
import sys

import ctk
import numpy as np
import qt
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleTest,
                                           ScriptedLoadableModuleWidget)
from slicer.util import VTKObservationMixin

from SimindPhantomLib import core, lesion, pipeline
from SimindPhantomLib.logic import SimindPhantomLogic

COL_SEGMENT, COL_ROLE, COL_INDEX, COL_ZNAME, COL_ACTIVITY, COL_VOLUME = range(6)
HEADERS = ["Structure", "Role", "Zubal index", "Zubal name", "Activity", "Volume (cm3)"]


# ----------------------------------------------------------------------------
class SimindPhantom(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SIMIND Phantom"
        self.parent.categories = ["Nuclear Medicine"]
        self.parent.dependencies = []
        self.parent.contributors = ["SimindPhantom contributors"]
        self.parent.helpText = """
Builds patient-specific voxel phantoms for the SIMIND Monte Carlo code, entirely inside Slicer:
CT -> TotalSegmentator -> optional lesions -> Zubal-type phantom (.dat + .zub) or
density (.dmi) + activity (.smi) maps on the same grid.
Structures are mapped to Zubal indices by segment NAME (editable table), not by segment order.
"""
        self.parent.acknowledgementText = """
Based on the workflow of the course "Herramientas Computacionales para Dosimetria Interna"
(Facultad de Ciencias Exactas, UNLP). Uses TotalSegmentator (Wasserthal et al., 2023).
"""


# ----------------------------------------------------------------------------
class SimindPhantomWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.rows = []
        self._updatingTable = False
        self._refreshing = False

    # ------------------------------------------------------------------ setup
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimindPhantom.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimindPhantomLogic()

        # widgets whose configuration is easier in code
        self.ui.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.outputDirSelector.settingKey = "SimindPhantom/OutputDir"
        self.ui.curvePathLineEdit.filters = ctk.ctkPathLineEdit.Files
        self.ui.curvePathLineEdit.nameFilters = ["CSV (*.csv)", "All files (*)"]
        self.ui.curvePathLineEdit.settingKey = "SimindPhantom/CurvePath"
        self.ui.progressBar.visible = False

        t = self.ui.structureTable
        t.setColumnCount(len(HEADERS))
        t.setHorizontalHeaderLabels(HEADERS)
        t.horizontalHeader().setSectionResizeMode(COL_SEGMENT, qt.QHeaderView.Stretch)
        t.verticalHeader().visible = False

        for m in lesion.METHODS:
            self.ui.lesionMethodComboBox.addItem(m.title)
        for q in core.TS_QUALITIES:
            self.ui.tsQualityComboBox.addItem(q, q)
        self.fillLesionTaskCombo()

        # connections
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.ui.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
        self.ui.runTotalSegmentatorButton.connect("clicked()", self.onRunTotalSegmentator)
        self.ui.installTotalSegmentatorButton.connect("clicked()", self.onInstallTotalSegmentator)
        self.ui.tsRequirementsButton.connect("clicked()", self.onTotalSegmentatorRequirements)
        self.ui.openSegmentEditorButton.connect("clicked()", self.onOpenSegmentEditor)
        self.ui.lesionMethodComboBox.connect("currentIndexChanged(int)", self.onLesionMethodChanged)
        self.ui.openMonaiButton.connect("clicked()", self.onOpenMonai)
        self.ui.detectLesionsButton.connect("clicked()", self.onDetectLesions)
        self.ui.refreshTableButton.connect("clicked()", lambda: self.refreshTable(keepZubal=True))
        self.ui.loadMappingButton.connect("clicked()", self.onLoadMapping)
        self.ui.saveMappingButton.connect("clicked()", self.onSaveMapping)
        self.ui.resetMappingButton.connect("clicked()", self.onResetMapping)
        self.ui.loadZubButton.connect("clicked()", self.onLoadZub)
        self.ui.loadActivityButton.connect("clicked()", self.onLoadActivities)
        self.ui.saveActivityButton.connect("clicked()", self.onSaveActivities)
        self.ui.structureTable.connect("itemChanged(QTableWidgetItem*)", self.onTableItemChanged)
        self.ui.exportZubalButton.connect("clicked()", self.onExportZubal)
        self.ui.exportDmiSmiButton.connect("clicked()", self.onExportDmiSmi)
        self.ui.referenceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGuiState)
        self.ui.bodyMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onBodyMaskChanged)
        self.ui.matrixLineEdit.connect("textChanged(QString)", self.updateGuiState)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.onLesionMethodChanged(0)
        self.updateZubalTableLabel()
        self.restoreParameters()
        self.updateGuiState()

    def cleanup(self):
        self.removeObservers()

    def onReload(self):
        """Reload also the SimindPhantomLib helper package: Slicer's Reload only
        re-imports this file, so without this the old logic stays in sys.modules."""
        for name in list(sys.modules):
            if name == "SimindPhantomLib" or name.startswith("SimindPhantomLib."):
                del sys.modules[name]
        ScriptedLoadableModuleWidget.onReload(self)

    def enter(self):
        self.updateGuiState()

    def onSceneEndClose(self, caller, event):
        self.rows = []
        self.fillTable()
        if self.parent.isEntered:
            self.updateGuiState()

    # ------------------------------------------------------------------ parameter node
    def restoreParameters(self):
        p = self.logic.getParameterNode()
        for role, selector in (("InputCT", self.ui.inputVolumeSelector),
                               ("Segmentation", self.ui.segmentationSelector)):
            node = p.GetNodeReference(role)
            if node:
                selector.setCurrentNode(node)
        if p.GetParameter("BaseName"):
            self.ui.baseNameLineEdit.text = p.GetParameter("BaseName")

    def saveParameters(self):
        p = self.logic.getParameterNode()
        ct, seg = self.ui.inputVolumeSelector.currentNode(), self.ui.segmentationSelector.currentNode()
        p.SetNodeReferenceID("InputCT", ct.GetID() if ct else None)
        p.SetNodeReferenceID("Segmentation", seg.GetID() if seg else None)
        p.SetParameter("BaseName", self.ui.baseNameLineEdit.text)

    # ------------------------------------------------------------------ GUI state
    def updateGuiState(self, *args):
        tsOk = self.logic.isTotalSegmentatorAvailable()
        ct = self.ui.inputVolumeSelector.currentNode()
        seg = self.ui.segmentationSelector.currentNode()
        self.ui.installTotalSegmentatorButton.visible = not tsOk
        self.ui.runTotalSegmentatorButton.enabled = tsOk and ct is not None
        self.ui.tsRequirementsButton.enabled = tsOk
        self.ui.runTotalSegmentatorButton.toolTip = (
            "" if tsOk else "Install the TotalSegmentator extension first.")
        self.ui.openSegmentEditorButton.enabled = seg is not None
        self.ui.detectLesionsButton.enabled = ct is not None and seg is not None
        self.ui.openMonaiButton.enabled = self.logic.isMonaiAuto3DSegAvailable()
        canExport = ct is not None and seg is not None and len(self.rows) > 0
        self.ui.exportZubalButton.enabled = canExport
        self.ui.exportDmiSmiButton.enabled = canExport
        useRef = self.ui.referenceVolumeSelector.currentNode() is not None
        for wdg in (self.ui.voxelSizeLineEdit, self.ui.matrixLineEdit, self.ui.cropToBodyCheckBox,
                    self.ui.marginSpinBox):
            wdg.enabled = not useRef

    def onInputChanged(self, node):
        if node is not None:
            info, warns = self.logic.ctDicomInfo(node)
            txt = []
            if info:
                txt.append(f"{info.get('manufacturer', '')} {info.get('model', '')}  "
                           f"kVp={info.get('kvp', '?')}  kernel={info.get('kernel', '?')}")
            txt += [f"WARNING: {w}" for w in warns]
            self.ui.ctInfoLabel.text = "\n".join(txt)
            if not self.ui.baseNameLineEdit.text or self.ui.baseNameLineEdit.text == "phantom":
                self.ui.baseNameLineEdit.text = re.sub(r"[^A-Za-z0-9_.-]+", "_", node.GetName()).strip("_")
        else:
            self.ui.ctInfoLabel.text = ""
        self.saveParameters()
        self.updateGuiState()

    def onSegmentationChanged(self, node):
        self.saveParameters()
        self._setBodyMaskSelector(self.logic.bodyMaskNode(node))
        self.refreshTable(keepZubal=False)

    def _setBodyMaskSelector(self, node):
        wasBlocked = self.ui.bodyMaskSelector.blockSignals(True)
        self.ui.bodyMaskSelector.setCurrentNode(node)
        self.ui.bodyMaskSelector.blockSignals(wasBlocked)

    def onBodyMaskChanged(self, node):
        self.logic.setBodyMaskNode(self.ui.segmentationSelector.currentNode(), node)
        self.refreshTable(keepZubal=True)

    # ------------------------------------------------------------------ helpers
    def setStatus(self, text, append=False):
        if append:
            self.ui.statusTextEdit.appendPlainText(text)
        else:
            self.ui.statusTextEdit.setPlainText(text)
        slicer.app.processEvents()

    def progress(self, fraction, message):
        self.ui.progressBar.visible = True
        self.ui.progressBar.value = int(100 * fraction)
        self.ui.progressBar.format = f"{message} (%p%)"
        slicer.app.processEvents()

    def ensureSegmentation(self):
        seg = self.ui.segmentationSelector.currentNode()
        if seg is None:
            ct = self.ui.inputVolumeSelector.currentNode()
            seg = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", f"{ct.GetName()} phantom segmentation" if ct else "phantom")
            if ct is not None:
                seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
            self.ui.segmentationSelector.setCurrentNode(seg)
        return seg

    # ------------------------------------------------------------------ TotalSegmentator
    def onInstallTotalSegmentator(self):
        with slicer.util.tryWithErrorDisplay("Could not install TotalSegmentator.", waitCursor=True):
            if self.logic.installExtension("TotalSegmentator"):
                if slicer.util.confirmOkCancelDisplay("TotalSegmentator was installed. Restart Slicer now?"):
                    slicer.util.restart()

    def onTotalSegmentatorRequirements(self):
        with slicer.util.tryWithErrorDisplay("TotalSegmentator Python packages could not be installed.",
                                             waitCursor=True):
            self.setStatus("Installing / checking TotalSegmentator Python packages "
                           "(the first time this downloads PyTorch and can take a long time)...")
            self.logic.ensureTotalSegmentatorRequirements(force=True)
            self.setStatus("TotalSegmentator Python packages are installed.", append=True)

    def onRunTotalSegmentator(self):
        ct = self.ui.inputVolumeSelector.currentNode()
        with slicer.util.tryWithErrorDisplay("TotalSegmentator failed.", waitCursor=True):
            seg = self.ensureSegmentation()
            quality, cpu = self.ui.tsQualityComboBox.currentData, self.ui.tsCpuCheckBox.checked
            if not self.logic.tsRequirementsOk:
                self.setStatus("Checking TotalSegmentator Python packages (first run installs them "
                               "and can take a long time)...")
                self.logic.ensureTotalSegmentatorRequirements()
            self.setStatus("Running TotalSegmentator 'total' (this can take several minutes)...")
            self.logic.runTotalSegmentator(ct, seg, task="total", quality=quality, cpu=cpu)
            if self.ui.tsBodyCheckBox.checked:
                self.setStatus("Running TotalSegmentator 'body' (separate, hidden 'body mask' "
                               "segmentation so it does not overlap the organs)...", append=True)
                bodyNode = self.logic.bodyMaskNode(seg, ct, create=True)
                bodyNode.GetSegmentation().RemoveAllSegments()
                self.logic.runTotalSegmentatorInto(ct, bodyNode, "body", quality=quality, cpu=cpu)
                self._setBodyMaskSelector(bodyNode)
            self.setStatus(f"Segmentation done: {seg.GetSegmentation().GetNumberOfSegments()} segments.",
                           append=True)
            self.refreshTable(keepZubal=True)

    def onOpenSegmentEditor(self):
        seg = self.ui.segmentationSelector.currentNode()
        ct = self.ui.inputVolumeSelector.currentNode()
        slicer.util.selectModule("SegmentEditor")
        editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor.setSegmentationNode(seg)
        if ct is not None:
            if hasattr(editor, "setSourceVolumeNode"):
                editor.setSourceVolumeNode(ct)
            else:
                editor.setMasterVolumeNode(ct)

    # ------------------------------------------------------------------ lesions
    def onLesionMethodChanged(self, index):
        self.ui.lesionStackedWidget.setCurrentIndex(index)
        method = lesion.METHODS[index]
        ok, reason = method.available(self.logic)
        self.ui.detectLesionsButton.toolTip = reason
        usesRoi = method.key in ("functional", "seeded")
        self.ui.lesionRoiComboBox.enabled = usesRoi

    def fillLesionTaskCombo(self):
        cb = self.ui.lesionTsTaskComboBox
        cb.clear()
        if not self.logic.isTotalSegmentatorAvailable():
            return
        for task, label, _ in lesion.TotalSegmentatorLesions().task_items(self.logic):
            cb.addItem(f"{task} - {label}", task)

    def fillRoiCombo(self):
        cb = self.ui.lesionRoiComboBox
        current = cb.currentData
        cb.clear()
        cb.addItem("(whole body)", "")
        for r in self.rows:
            if r.role == core.ROLE_ORGAN:
                cb.addItem(r.name, r.key)
        idx = cb.findData(current) if current else 0
        cb.setCurrentIndex(max(idx, 0))

    def onOpenMonai(self):
        slicer.util.selectModule("MONAIAuto3DSeg")

    def onDetectLesions(self):
        ct = self.ui.inputVolumeSelector.currentNode()
        seg = self.ui.segmentationSelector.currentNode()
        method = lesion.METHODS[self.ui.lesionMethodComboBox.currentIndex]
        params = {
            "task": self.ui.lesionTsTaskComboBox.currentData,
            "quality": self.ui.tsQualityComboBox.currentData,
            "cpu": self.ui.tsCpuCheckBox.checked,
            "source": self.ui.lesionSourceSelector.currentNode(),
            "regex": self.ui.lesionRegexLineEdit.text.strip(),
            "functional": self.ui.functionalVolumeSelector.currentNode(),
            "fraction": self.ui.functionalThresholdSpinBox.value / 100.0,
            "markups": self.ui.seedMarkupsSelector.currentNode(),
            "tolerance": self.ui.seedToleranceSpinBox.value,
            "max_radius_mm": self.ui.seedMaxRadiusSpinBox.value,
            "roi_segment": self.ui.lesionRoiComboBox.currentData or None,
            "min_ml": self.ui.lesionMinVolumeSpinBox.value,
            "split": self.ui.lesionSplitCheckBox.checked,
        }
        with slicer.util.tryWithErrorDisplay("Lesion detection failed.", waitCursor=True):
            ok, reason = method.available(self.logic)
            if not ok:
                raise core.PhantomError(reason)
            self.setStatus(f"Detecting lesions: {method.title}...")
            ids = method.run(self.logic, ct, seg, params)
            names = [seg.GetSegmentation().GetSegment(i).GetName() for i in ids]
            self.setStatus(f"{len(ids)} lesion segment(s) added:\n" + "\n".join(names), append=True)
            self.refreshTable(keepZubal=True)

    # ------------------------------------------------------------------ structure table
    def refreshTable(self, keepZubal=True):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._refreshTable(keepZubal)
        finally:
            self._refreshing = False

    def _refreshTable(self, keepZubal):
        seg = self.ui.segmentationSelector.currentNode()
        old = {r.key: r for r in self.rows}
        with slicer.util.tryWithErrorDisplay("Could not read the segmentation.", waitCursor=True):
            bodyNode = self.ui.bodyMaskSelector.currentNode()
            if seg is not None and any(self.logic.isBodySegmentName(seg.GetSegmentation().GetSegment(sid).GetName())
                                       for sid in seg.GetSegmentation().GetSegmentIDs()):
                if bodyNode is None or bodyNode is seg:
                    bodyNode = self.logic.bodyMaskNode(seg, self.ui.inputVolumeSelector.currentNode(),
                                                       create=True)
                n = self.logic.moveBodySegments(seg, bodyNode)
                self.logic.setBodyMaskNode(seg, bodyNode)
                self._setBodyMaskSelector(bodyNode)
                self.setStatus(f"{n} body segment(s) moved to '{bodyNode.GetName()}' so they no longer "
                               "overlap the organs (used only as body mask).")
            self.rows = self.logic.structureRows(seg, bodyNode) if seg is not None else []
            for r in self.rows:
                o = old.get(r.key)
                if o is not None:
                    r.activity = o.activity
                    if keepZubal:
                        r.zubal_index, r.role = o.zubal_index, o.role
        self.fillTable()
        self.fillRoiCombo()
        self.updateGuiState()

    def fillTable(self):
        t = self.ui.structureTable
        self._updatingTable = True
        t.setRowCount(len(self.rows))
        unknown = 0
        for i, r in enumerate(self.rows):
            values = [r.name, r.role, str(r.zubal_index), self.logic.zubalName(r.zubal_index),
                      str(r.activity), f"{r.volume_cm3:.2f}" if r.volume_cm3 else ""]
            for c, v in enumerate(values):
                item = qt.QTableWidgetItem(v)
                editable = c in (COL_INDEX, COL_ACTIVITY) and r.role not in (core.ROLE_BODY, core.ROLE_IGNORE)
                flags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
                if editable:
                    flags |= qt.Qt.ItemIsEditable
                item.setFlags(flags)
                if not r.known:
                    item.setBackground(qt.QBrush(qt.QColor(255, 235, 150)))
                    item.setToolTip("No mapping rule matched this name: default soft tissue. Check it.")
                elif r.role == core.ROLE_LESION:
                    item.setBackground(qt.QBrush(qt.QColor(255, 210, 210)))
                elif r.role in (core.ROLE_BODY, core.ROLE_BACKGROUND):
                    item.setBackground(qt.QBrush(qt.QColor(225, 235, 250)))
                t.setItem(i, c, item)
            unknown += 0 if r.known else 1
        self._updatingTable = False
        if unknown:
            self.setStatus(f"{unknown} structure(s) without mapping rule (yellow rows): check their Zubal index.")

    def onTableItemChanged(self, item):
        if self._updatingTable:
            return
        r = self.rows[item.row()]
        col = item.column()
        try:
            v = int(float(item.text()))
            if col == COL_INDEX:
                if not 0 <= v <= core.UINT8_MAX:
                    raise ValueError("Zubal index must be 0..255")
                r.zubal_index = v
            elif col == COL_ACTIVITY:
                if not 0 <= v <= core.INT16_MAX:
                    raise ValueError(f"Activity must be 0..{core.INT16_MAX}")
                r.activity = v
            else:
                return
        except ValueError as e:
            slicer.util.errorDisplay(f"'{item.text()}': {e}")
        self._updatingTable = True
        item.setText(str(r.zubal_index if col == COL_INDEX else r.activity))
        if col == COL_INDEX:
            self.ui.structureTable.item(item.row(), COL_ZNAME).setText(self.logic.zubalName(r.zubal_index))
        self._updatingTable = False

    def _fileDialog(self, save, title, filt):
        f = (qt.QFileDialog.getSaveFileName if save else qt.QFileDialog.getOpenFileName)(
            slicer.util.mainWindow(), title, self.ui.outputDirSelector.currentPath or "", filt)
        return f or None

    def onLoadMapping(self):
        path = self._fileDialog(False, "Load mapping", "CSV (*.csv)")
        if path:
            with slicer.util.tryWithErrorDisplay("Could not load the mapping."):
                self.logic.loadMapping(path)
                self.refreshTable(keepZubal=False)

    def onSaveMapping(self):
        path = self._fileDialog(True, "Save mapping rules", "CSV (*.csv)")
        if path:
            with slicer.util.tryWithErrorDisplay("Could not save the mapping."):
                core.save_mapping_csv(path, self.logic.rules)

    def onResetMapping(self):
        self.logic.resetMapping()
        self.refreshTable(keepZubal=False)

    def onLoadZub(self):
        path = self._fileDialog(False, "Load Zubal table", "Zubal table (*.zub *.csv);;All files (*)")
        if path:
            with slicer.util.tryWithErrorDisplay("Could not read the Zubal table."):
                self.logic.loadZubalTable(path)
                self.updateZubalTableLabel()
                self.fillTable()

    def updateZubalTableLabel(self):
        self.ui.zubalTableLabel.text = (f"Zubal table: {os.path.basename(self.logic.zubalTableSource)} "
                                        f"({len(self.logic.zubalTable)} indices)")

    def onLoadActivities(self):
        path = self._fileDialog(False, "Load activities", "CSV (*.csv)")
        if path:
            with slicer.util.tryWithErrorDisplay("Could not load activities."):
                n = pipeline.load_activities(path, self.rows)
                self.fillTable()
                self.setStatus(f"Activities applied to {n} structure(s).")

    def onSaveActivities(self):
        path = self._fileDialog(True, "Save activities", "CSV (*.csv)")
        if path:
            with slicer.util.tryWithErrorDisplay("Could not save activities."):
                pipeline.save_activities(path, self.rows)

    # ------------------------------------------------------------------ export
    def phantomOptions(self, computeDensity):
        o = pipeline.PhantomOptions()
        o.fill_background = self.ui.fillBackgroundCheckBox.checked
        o.add_skin = self.ui.addSkinCheckBox.checked
        o.thresholds = {"gas_max": self.ui.gasMaxSpinBox.value, "fat_max": self.ui.fatMaxSpinBox.value,
                        "soft_max": self.ui.softMaxSpinBox.value}
        o.body_hu_threshold = self.ui.bodyThresholdSpinBox.value
        o.spacing = core.parse_triplet(self.ui.voxelSizeLineEdit.text, float, "Voxel size") \
            if self.ui.voxelSizeLineEdit.text.strip() else None
        o.matrix = core.parse_triplet(self.ui.matrixLineEdit.text, int, "Matrix") \
            if self.ui.matrixLineEdit.text.strip() else None
        o.crop_to_body = self.ui.cropToBodyCheckBox.checked
        o.margin_mm = self.ui.marginSpinBox.value
        o.first_slice = "head" if self.ui.firstSliceComboBox.currentIndex == 0 else "feet"
        o.flip_x = self.ui.flipXCheckBox.checked
        o.flip_y = self.ui.flipYCheckBox.checked
        o.compute_density = computeDensity
        o.curve_path = self.ui.curvePathLineEdit.currentPath.strip() or None
        o.density_outside_body_zero = self.ui.densityOutsideZeroCheckBox.checked
        return o

    def _export(self, computeDensity):
        ct = self.ui.inputVolumeSelector.currentNode()
        seg = self.ui.segmentationSelector.currentNode()
        outDir = self.ui.outputDirSelector.currentPath
        base = self.ui.baseNameLineEdit.text.strip()
        if not outDir:
            raise core.PhantomError("Choose an output folder.")
        if not base:
            raise core.PhantomError("Choose a base name.")
        self.saveParameters()
        self.ui.outputDirSelector.addCurrentPathToHistory()
        rows = copy.deepcopy(self.rows)
        self.setStatus("Building phantom...")
        result = self.logic.buildPhantom(ct, seg, rows, self.phantomOptions(computeDensity),
                                         self.ui.referenceVolumeSelector.currentNode(), self.progress,
                                         bodyNode=self.ui.bodyMaskSelector.currentNode())
        # show the volumes computed at CT resolution in the table
        vols = {r.key: r.volume_cm3 for r in result.rows}
        for r in self.rows:
            r.volume_cm3 = vols.get(r.key, r.volume_cm3)
        self.fillTable()
        return result, outDir, base

    def onExportZubal(self):
        try:
            with slicer.util.tryWithErrorDisplay("Export failed.", waitCursor=True):
                result, outDir, base = self._export(computeDensity=False)
                files = self.logic.exportZubal(result, outDir, base)
                if self.ui.showResultCheckBox.checked:
                    self.logic.showZubal(result, base)
                self.setStatus(pipeline.summary_text(result, files))
        finally:
            self.ui.progressBar.visible = False

    def onExportDmiSmi(self):
        try:
            with slicer.util.tryWithErrorDisplay("Export failed.", waitCursor=True):
                result, outDir, base = self._export(computeDensity=True)
                files = self.logic.exportDensityActivity(result, outDir, base)
                if self.ui.showResultCheckBox.checked:
                    self.logic.showDensityActivity(result, base)
                self.setStatus(pipeline.summary_text(result, files))
        finally:
            self.ui.progressBar.visible = False


# ----------------------------------------------------------------------------
class SimindPhantomTest(ScriptedLoadableModuleTest):
    """Self-test with synthetic data (does not need TotalSegmentator)."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_export_both_workflows()
        self.setUp()
        self.test_functional_threshold_lesions()

    @staticmethod
    def _ball(shape, c, r):
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r

    def _synthetic(self):
        shape = (40, 60, 60)
        hu = np.full(shape, -1000, np.int16)
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        body = ((y - 30) ** 2 + (x - 30) ** 2 <= 25 ** 2) & (z >= 2) & (z < 38)
        hu[body] = 40
        liver, kidney, les = (self._ball(shape, (20, 25, 20), 9), self._ball(shape, (20, 38, 40), 5),
                              self._ball(shape, (20, 25, 18), 3))
        hu[liver] = 60
        ct = slicer.util.addVolumeFromArray(hu, name="SyntheticCT")
        ct.SetSpacing(2.0, 2.0, 2.5)
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SyntheticSeg")
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        logic = SimindPhantomLogic()
        for name, m in (("liver", liver), ("right kidney", kidney), ("lesion_1 (test)", les)):
            logic.addSegmentFromMask(seg, ct, m, name)
        return logic, ct, seg, {"liver": liver, "lesion": les}

    def test_export_both_workflows(self):
        self.delayDisplay("Export Zubal and density/activity from synthetic data")
        logic, ct, seg, masks = self._synthetic()
        rows = logic.structureRows(seg)
        byName = {r.name: r for r in rows}
        self.assertEqual(byName["liver"].zubal_index, 12)
        self.assertEqual(byName["right kidney"].zubal_index, 14)
        self.assertEqual(byName["lesion_1 (test)"].role, core.ROLE_LESION)
        byName["liver"].activity = 100
        byName["lesion_1 (test)"].activity = 900

        opts = pipeline.PhantomOptions(spacing=(4.0, 4.0, 5.0), matrix=(32, 32, 24))
        result = logic.buildPhantom(ct, seg, rows, opts)
        outDir = os.path.join(slicer.app.temporaryPath, "SimindPhantomTest")
        f1 = logic.exportZubal(result, outDir, "test")
        f2 = logic.exportDensityActivity(result, outDir, "test")

        dat = np.fromfile(f1["dat"], np.uint8).reshape(24, 32, 32)
        smi = np.fromfile(f2["smi"], "<i2").reshape(24, 32, 32)
        dmi = np.fromfile(f2["dmi"], "<i2").reshape(24, 32, 32)
        self.assertTrue({0, 12, 14, 63}.issubset(set(np.unique(dat).tolist())))
        self.assertTrue(np.all(smi[dat == 63] == 900))
        self.assertTrue(np.all(smi[dat == 12] == 100))
        self.assertTrue(np.all(dmi[dat == 0] == 0))
        self.assertGreater(int(np.median(dmi[dat == 12])), 1000)
        self.assertEqual(byName["lesion_1 (test)"].voxels, int(masks["lesion"].sum()))
        self.test_body_mask_moved(logic, ct, seg, dat, outDir)
        logic.showZubal(result, "test")
        logic.showDensityActivity(result, "test")
        self.delayDisplay("Export test passed")

    def test_body_mask_moved(self, logic, ct, seg, datBefore, outDir):
        self.delayDisplay("Body segment is moved out of the phantom segmentation")
        z, y, x = np.ogrid[:40, :60, :60]
        body = ((y - 30) ** 2 + (x - 30) ** 2 <= 25 ** 2) & (z >= 2) & (z < 38)
        logic.addSegmentFromMask(seg, ct, body, "body")
        bodyNode = logic.bodyMaskNode(seg, ct, create=True)
        self.assertEqual(logic.moveBodySegments(seg, bodyNode), 1)
        names = [seg.GetSegmentation().GetSegment(s).GetName() for s in seg.GetSegmentation().GetSegmentIDs()]
        self.assertNotIn("body", names)
        self.assertEqual(bodyNode.GetSegmentation().GetNumberOfSegments(), 1)
        self.assertIs(logic.bodyMaskNode(seg), bodyNode)
        rows = logic.structureRows(seg, bodyNode)
        self.assertEqual(sum(r.role == core.ROLE_BODY for r in rows), 1)
        byName = {r.name: r for r in rows}
        byName["liver"].activity = 100
        byName["lesion_1 (test)"].activity = 900
        result = logic.buildPhantom(ct, seg, rows, pipeline.PhantomOptions(spacing=(4.0, 4.0, 5.0),
                                    matrix=(32, 32, 24)), bodyNode=bodyNode)
        f = logic.exportZubal(result, outDir, "test_body")
        dat = np.fromfile(f["dat"], np.uint8).reshape(24, 32, 32)
        self.assertTrue(np.array_equal(dat, datBefore))

    def test_functional_threshold_lesions(self):
        self.delayDisplay("Functional threshold lesion detection")
        logic, ct, seg, _ = self._synthetic()
        shape = (40, 60, 60)
        spect = np.ones(shape, np.float32)
        spect[self._ball(shape, (20, 25, 20), 3)] = 100
        spect[self._ball(shape, (20, 38, 40), 2)] = 70
        node = slicer.util.addVolumeFromArray(spect, name="SyntheticSPECT")
        node.SetSpacing(2.0, 2.0, 2.5)
        method = lesion.FunctionalThreshold()
        ids = method.run(logic, ct, seg, {"functional": node, "fraction": 0.4, "min_ml": 0.01,
                                          "split": True, "roi_segment": None})
        self.assertEqual(len(ids), 2)
        rows = logic.structureRows(seg)
        self.assertEqual(sum(r.role == core.ROLE_LESION for r in rows), 3)
        self.delayDisplay("Functional threshold test passed")
