"""Minimal offscreen checks for the Qt application shell."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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
        assert state.data_state()[0] == "error"
        data = window.dataset_page.pages["data"]
        assert data._table.rowCount() == 2
        badges = [item.text() for item in data.header_badges(state)]
        assert badges[0].startswith("labels incomplete") and badges[-1].startswith("Sparse labels")
        preprocess = window.dataset_page.pages["preprocess"]
        preprocess._run_dialog.open()
        assert not preprocess._run.isEnabled()  # labels are missing
        preprocess._run_dialog.close()
        assert "-pl nnUNetPlannerResEncL" in preprocess._command.command()
    finally:
        window.close()
        application.processEvents()


def _results_dataset(tmp_path: Path):
    """A dataset with one trained model: folds 0 and 1 validated, fold all finished, fold 2 still training."""
    from nnworkbench.ui.dataset_state import load_state

    raw = tmp_path / "raw" / "Dataset008_Cells"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    (raw / "imagesTr" / "a_0000.nii.gz").touch()
    (raw / "labelsTr" / "a.nii.gz").touch()
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1},
        "numTraining": 1, "file_ending": ".nii.gz"}), encoding="utf-8")
    model = tmp_path / "results" / "Dataset008_Cells" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    for fold, dice in (("0", 0.8), ("1", 0.6), ("all", 0.95)):
        folder = model / f"fold_{fold}"
        (folder / "validation").mkdir(parents=True)
        (folder / "checkpoint_final.pth").touch()
        (folder / "training_log_1.txt").write_text("x: Epoch 999 \n", encoding="utf-8")
        case = {"reference_file": "/gt/a.nii.gz", "prediction_file": "/p/a.nii.gz", "metrics": {"1": {"Dice": dice}}}
        (folder / "validation" / "summary.json").write_text(json.dumps({
            "foreground_mean": {"Dice": dice}, "mean": {"1": {"Dice": dice}}, "metric_per_case": [case]}),
            encoding="utf-8")
    (model / "fold_2").mkdir()
    (model / "fold_2" / "training_log_1.txt").write_text("x: Epoch 12 \n", encoding="utf-8")
    preprocessed = tmp_path / "preprocessed" / "Dataset008_Cells"
    (preprocessed / "nnUNetPlans_3d_fullres").mkdir(parents=True)
    (preprocessed / "nnUNetPlans_3d_fullres" / "a.b2nd").touch()
    (preprocessed / "nnUNetPlans.json").write_text(json.dumps({
        "plans_name": "nnUNetPlans", "configurations": {"3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres"}}}),
        encoding="utf-8")
    record = DatasetRecord(8, "Cells", raw_path=raw, preprocessed_path=preprocessed, results_path=model.parent)
    return record, load_state(record)


def _pump(application, seconds: float = 0.5) -> None:
    import time

    end = time.time() + seconds
    while time.time() < end:
        application.processEvents()
        time.sleep(0.01)


def test_train_validate_predict_pages(tmp_path: Path) -> None:
    application = _application()
    window = MainWindow()
    try:
        record, state = _results_dataset(tmp_path)
        window.dataset_page.record = record
        window.dataset_page._apply_state(state)
        model = state.models[0]
        assert model.mean_dice == pytest.approx(0.7)  # fold all is excluded
        assert state.train_state() == ("todo", "Train · a fold is training")  # fold 2 was just logged
        assert state.validate_state()[0] == "ok" and state.predict_state()[0] == "ok"
        assert state.preprocess_state()[0] == "ok"
        import dataclasses
        from types import SimpleNamespace

        outdated = dataclasses.replace(state, analysis={"files_analyzed": 1}, analysis_outdated=2,
                                       plans=(SimpleNamespace(preprocessed_time=1.0, outdated=True),))
        pages = window.dataset_page.pages
        assert "statistics outdated · 2 labels changed" in [b.text() for b in pages["data"].header_badges(outdated)]
        train_badges = [b.text() for b in pages["train"].header_badges(outdated)]
        assert "preprocessing outdated · new trainings use old labels" in train_badges

        train = window.dataset_page.pages["train"]
        fold_2 = next(fold for fold in model.folds if fold.name == "2")
        train._prefill(model, fold_2, continue_training=True)
        assert train._command.command().endswith("--c") and " 3d_fullres 2 " in train._command.command()
        assert train._source_note.text().startswith("From nnUNetTrainer")
        train._run_dialog.close()
        # ▶ Train… in the header always starts from the defaults, not from the selected model
        train._open_defaults()
        command = train._command.command()
        # defaults = the existing model here (nnUNetTrainer, nnUNetPlans, 3d_fullres), so its first free fold
        assert " 3d_fullres 3 " in command and "-p nnUNetPlans" in command and "--c" not in command
        assert train._source_note.text() == "Defaults for this dataset." and train._trainer_hint.isVisibleTo(train)
        train._run_dialog.close()
        # starting an existing fold from scratch asks first; continuing or a new fold does not
        fold_0 = next(fold for fold in model.folds if fold.name == "0")
        train._prefill(model, fold_0, continue_training=False)
        assert "is finished" in train._overwrite_warning()
        train._prefill(model, fold_2, continue_training=False)
        assert "already has a training" in train._overwrite_warning()
        train._continue.setChecked(True)
        assert train._overwrite_warning() == ""
        train._run_dialog.close()
        # ▶ Next fold… trains the first fold the model does not have yet (0, 1, 2 and all exist)
        assert train._free_fold(model) == "3"
        train._prefill(model, train._free_fold(model), continue_training=False)
        assert " 3d_fullres 3 " in train._command.command() and "nnUNetPlans" in train._command.command()
        assert train._overwrite_warning() == ""
        # several folds: one command per fold, run one after another in one job
        train._fold_choice.set_selected(["3", "4"])
        assert train._command.command().count("nnUNetv2_train") == 2 and " && " in train._command.command()
        assert train._start.text() == "▶ Train 2 folds"
        train._fold_choice.set_selected(["0", "3"])
        assert "Fold 0 of this model is finished" in train._overwrite_warning()
        train._run_dialog.close()

        # without preprocessed data, plans and configuration stay empty (placeholders in the command)
        train.set_state(dataclasses.replace(state, plans=()))
        train._prefill(model, fold_2, continue_training=True)
        assert train._plans.currentText() == "" and " CONFIGURATION 2 " in train._command.command()
        assert "-p PLANS" in train._command.command()
        train._run_dialog.close()
        train.set_state(state)

        validate = window.dataset_page.pages["validate"]
        _pump(application)
        assert validate._fold is None  # opens on the model overview
        fold_0 = next(fold for fold in model.folds if fold.name == "0")
        validate._fold_clicked(model, fold_0)
        _pump(application)
        assert validate._fold is fold_0 and len(validate._cases) == 1
        from nnworkbench.services.workflow_status import ValidationCase

        cases = [ValidationCase(name, "", "", dice, None, {}) for name, dice in (("b", 0.9), ("a", 0.2), ("c", None))]
        assert [case.case_id for case in validate._sorted_cases(cases)] == ["a", "b", "c"]  # by name by default
        validate._cases = tuple(cases)
        validate._sort_cases(1)  # Dice: weakest first, cases without Dice last
        assert [case.case_id for case in validate._cases] == ["a", "b", "c"]
        validate._sort_cases(1)  # again: reversed
        assert [case.case_id for case in validate._cases] == ["b", "a", "c"]
        validate._sort_cases(0)
        assert [case.case_id for case in validate._cases] == ["a", "b", "c"]
        validate._fold_clicked(model, fold_0)  # second click: back to the model
        assert validate._fold is None

        predict = window.dataset_page.pages["predict"]
        assert predict._folds.selected() == ["0", "1"]  # fold all only when it is the only finished fold
        predict._input.setText(str(tmp_path))
        predict._folders_changed()  # the last folders are remembered (one pair, not per dataset)
        other = dataclasses.replace(state, record=DatasetRecord(9, "Other", raw_path=state.record.raw_path))
        predict.set_state(other)
        assert predict._input.text() == str(tmp_path)  # kept when switching datasets
        assert window.context.store.value("predict/input") == str(tmp_path)
        predict._continue.setChecked(True)
        assert predict._command.command().endswith("-f 0 1 -npp 3 -nps 3 --continue_prediction")
        predict._npp.setValue(1); predict._nps.setValue(2)  # fewer processes, e.g. when running out of RAM
        assert "-npp 1 -nps 2" in predict._command.command()
    finally:
        window.close()
        application.processEvents()


def test_about_dialog_shows_version_and_credits() -> None:
    from importlib import metadata

    from qtpy.QtWidgets import QLabel

    from nnworkbench.ui.about_dialog import AboutDialog

    application = _application()  # keep a reference: PyQt5 would delete an unreferenced QApplication
    window = MainWindow()
    dialog = AboutDialog(window)  # owned by the window, as in the app (an orphan dialog crashed at exit)
    try:
        labels = dialog.findChildren(QLabel)
        texts = " ".join(widget.text() for widget in labels)
        assert f"Version {metadata.version('nnworkbench')}" in texts
        assert "Applied Computer Vision Lab" in texts and "Apache License 2.0" in texts
        assert sum(1 for widget in labels if widget.pixmap() is not None and not widget.pixmap().isNull()) >= 2
    finally:
        dialog.close()
        window.close()
        application.processEvents()
