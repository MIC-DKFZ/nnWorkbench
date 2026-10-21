"""nnU-Net's region-based labels (a label is a list of IDs, with regions_class_order): the nnU-Net Workflow must run
through; annotating and pushing the class table are refused instead of breaking dataset.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication, QWidget

from nnworkbench import command_runner
from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_json import definition_errors, has_regions

# BraTS-like: voxel IDs 1, 2, 3 in the label files; nnU-Net trains on the overlapping regions
REGION_LABELS = {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": 3}


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _region_dataset(raw_root: Path) -> Path:
    folder = raw_root / "Dataset137_BraTS"
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    for case in ("brats_001", "brats_002"):
        for channel in range(4):
            (folder / "imagesTr" / f"{case}_{channel:04d}.nii.gz").touch()
        (folder / "labelsTr" / f"{case}.nii.gz").touch()
    manifest = {"channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}, "labels": REGION_LABELS,
                "regions_class_order": [1, 2, 3], "numTraining": 2, "file_ending": ".nii.gz"}
    (folder / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "dataset.json").read_text(encoding="utf-8"))


def test_region_labels_are_valid_and_detected(tmp_path: Path) -> None:
    folder = _region_dataset(tmp_path)
    assert definition_errors(_manifest(folder)) == ()
    assert has_regions(_manifest(folder))
    assert not has_regions({"labels": {"background": 0, "cell": 1}})
    assert not has_regions(None)


def test_json_editor_saves_region_labels_unchanged(tmp_path: Path) -> None:
    from nnworkbench.ui.json_editor import DatasetJsonPanel

    application = _application()
    folder = _region_dataset(tmp_path)
    before = _manifest(folder)
    parent = QWidget()
    panel = DatasetJsonPanel(parent)
    try:
        panel.set_manifest(folder, before, True, cases=2)
        panel._start_edit()
        panel._save_edit()
        after = _manifest(folder)
        assert after["labels"] == REGION_LABELS and after["regions_class_order"] == [1, 2, 3]
        assert {key: after[key] for key in before} == before  # the editor only adds its empty standard keys
        assert "Not saved" not in panel._note.text()
    finally:
        parent.close()
        application.processEvents()


def test_pushing_the_class_table_keeps_the_regions(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import push_classes

    folder = _region_dataset(tmp_path)
    before = _manifest(folder)
    with pytest.raises(ValueError, match="region-based labels"):
        push_classes(folder, [(0, "background"), (1, "tumor")])
    assert _manifest(folder) == before


def test_data_page_has_no_problems_and_no_annotation_for_regions(tmp_path: Path) -> None:
    from nnworkbench.ui.dataset_state import load_state
    from nnworkbench.ui.stages.common import label_mode_badge

    application = _application()
    folder = _region_dataset(tmp_path)
    state = load_state(DatasetRecord(137, "BraTS", raw_path=folder))
    assert state.regions and state.data_problems() == []
    badge = label_mode_badge(state)
    try:
        assert badge.text() == "Region-based labels"
    finally:
        badge.deleteLater()
        application.processEvents()


def test_cli_status_and_suggest_run_and_annotate_is_refused(tmp_path, monkeypatch, capsys) -> None:
    for name in ("raw", "preprocessed", "results"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {
        f"nnUNet_{name}": str(tmp_path / name) for name in ("raw", "preprocessed", "results")})
    _region_dataset(tmp_path / "raw")

    assert command_runner._run_builtin_command(["status", "-d", "137", "--json", "--no-device"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["stages"]["data"]["state"] != "error" and status["cases"] == 2
    assert command_runner._run_builtin_command(["suggest", "-d", "137", "--step", "preprocess", "--json",
                                                "--no-device"]) == 0
    assert "nnUNetv2_plan_and_preprocess" in json.loads(capsys.readouterr().out)["command"]
    with pytest.raises(ValueError, match="region-based labels"):
        command_runner._run_builtin_command(["annotate", "-d", "137", "--case", "brats_001"])
