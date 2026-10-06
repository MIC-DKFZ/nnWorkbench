"""Minimal offscreen checks for the Qt application shell."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication

from nnworkbench.ui.main_window import MainWindow


def test_main_window_can_be_created_offscreen() -> None:
    QStandardPaths.setTestModeEnabled(True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    try:
        assert window.windowTitle().startswith("nnWorkbench")
        assert window.splitter.count() == 3
    finally:
        window.close()
        application.processEvents()


def test_preprocessing_dialog_defaults_to_resenc_l_planner() -> None:
    from nnworkbench.ui.preprocessing_dialog import PreprocessingDialog

    application = QApplication.instance() or QApplication([])
    dialog = PreprocessingDialog(42, ("ExperimentPlanner", "nnUNetPlannerResEncL"))

    try:
        assert dialog.planner.currentText() == "nnUNetPlannerResEncL"
        assert "-pl nnUNetPlannerResEncL" in dialog.command()
    finally:
        dialog.close()
        application.processEvents()
