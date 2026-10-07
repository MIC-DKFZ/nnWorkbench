"""Minimal offscreen checks for the Qt application shell."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.ui.main_window import MainWindow


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def test_main_window_starts_on_the_overview() -> None:
    application = _application()
    window = MainWindow()
    try:
        assert window.windowTitle().startswith("nnWorkbench")
        assert window.pages.currentWidget() is window.overview
    finally:
        window.close()
        application.processEvents()


def test_dataset_page_shows_cases_and_stage_states(tmp_path: Path) -> None:
    from nnworkbench.ui.dataset_state import load_state

    raw = tmp_path / "Dataset007_Cells"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    (raw / "imagesTr" / "a_0000.nii.gz").touch()
    (raw / "imagesTr" / "b_0000.nii.gz").touch()
    (raw / "labelsTr" / "a.nii.gz").touch()
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1, "ignore": 2},
        "numTraining": 1, "file_ending": ".nii.gz"}), encoding="utf-8")
    application = _application()
    window = MainWindow()
    try:
        record = DatasetRecord(7, "Cells", raw_path=raw)
        state = load_state(record)
        window.dataset_page.record = record
        window.dataset_page._apply_state(state)
        assert state.sparse and state.raw.missing_labels == 1
        assert state.data_state() == "warn"
        data = window.dataset_page.pages["data"]
        assert data._table.rowCount() == 2
        badges = [item.text() for item in data.header_badges(state)]
        assert badges[0].startswith("labels incomplete") and badges[-1].startswith("Sparse labels")
        preprocess = window.dataset_page.pages["preprocess"]
        assert not preprocess._run.isEnabled()  # labels are missing
        assert "-pl nnUNetPlannerResEncL" in preprocess._command.command()
    finally:
        window.close()
        application.processEvents()
