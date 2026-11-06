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


def test_closing_with_a_running_job_asks_first(tmp_path: Path, monkeypatch) -> None:
    import sys

    application = _application()
    window = MainWindow()
    jobs = window.context.jobs
    jobs._log_folder = tmp_path
    asked = []
    try:
        assert jobs._start("Train · fold 0", "Cells", "one fold",
                           [("fold 0", sys.executable, ["-c", "import time; time.sleep(30)"])])
        monkeypatch.setattr(window, "_ask_about_job", lambda job, windows: asked.append(job) or False)
        assert not window._confirm_quit() and jobs.busy  # Cancel: the window and the job stay
        monkeypatch.setattr(window, "_ask_about_job", lambda job, windows: True)
        assert window._confirm_quit() and asked[0].title == "Train · fold 0"  # closeEvent then stops the job
        window.context.shutdown()
        assert not jobs.busy
    finally:
        monkeypatch.undo()
        jobs.shutdown()
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
        # the three folders of this test, not those saved on this computer (none on the CI runners): the run
        # checks below assume they exist
        for name in ("raw", "preprocessed", "results"):
            window.context.roots[name] = tmp_path / name
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
        assert fold_2.status == "training"  # its log was just written: a second run asks, also with Continue
        assert "still training" in train._overwrite_warning()
        train._continue.setChecked(True)
        assert "still training" in train._overwrite_warning()
        train._prefill(model, fold_0, continue_training=True)
        assert train._overwrite_warning() == ""
        train._run_dialog.close()
        # ▶ Next fold… trains the first fold the model does not have yet (0, 1, 2 and all exist)
        assert train._free_fold(model) == "3"
        train._prefill(model, train._free_fold(model), continue_training=False)
        assert " 3d_fullres 3 " in train._command.command() and "nnUNetPlans" in train._command.command()
        assert train._overwrite_warning() == ""
        # several folds: one command per fold, run one after another in one job
        train._fold_choice.set_selected(["3", "4"])
        assert train._command.command().count("nnUNetv2_train") == 2 and "\n" in train._command.command()
        assert train._start.text() == "▶ Train 2 folds"
        train._fold_choice.set_selected(["0", "3"])
        assert "Fold 0 of this model is finished" in train._overwrite_warning()
        train._run_dialog.close()

        # without preprocessed data here, plans and configuration are still filled in (preprocessed on a cluster):
        # the command can be copied, the run stays disabled
        train.set_state(dataclasses.replace(state, plans=()))
        train._prefill(model, fold_2, continue_training=True)
        assert train._plans.currentText() == "nnUNetPlans" and " 3d_fullres 2 " in train._command.command()
        assert not train._start.isEnabled() and "not preprocessed on this computer" in train._busy_hint.text()
        assert train._busy_hint.property("tone") == "orange" and "only with Copy command" in train._plans_hint.text()
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
        from nnworkbench.services.defaults import prediction_processes

        processes = prediction_processes()  # nnU-Net's 3, fewer on a machine with fewer cores (e.g. a CI runner)
        assert predict._command.command().endswith(f"-f 0 1 -npp {processes} -nps {processes} --continue_prediction")
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


def test_new_iteration_button_and_source_dataset(tmp_path: Path) -> None:
    from nnworkbench.ui.new_dataset_dialog import NewDatasetDialog

    application = _application()
    window = MainWindow()
    try:
        overview = window.overview
        assert not overview._iteration.isHidden()

        first = DatasetRecord(3, "Cells", raw_path=tmp_path / "Dataset003_Cells")
        second = DatasetRecord(7, "Lungs", raw_path=tmp_path / "Dataset007_Lungs")
        dialog = NewDatasetDialog(window, title="Create next iteration", sources=[first, second], source=second,
                                  suggest=lambda record: (record.dataset_id + 1, f"{record.name}_iter1"))
        assert dialog.source() is second  # the selected dataset is preselected
        assert (dialog.dataset_id.value(), dialog.name.text()) == (8, "Lungs_iter1")
        dialog.source_box.setCurrentIndex(0)  # switch the source: ID and name follow
        assert dialog.source() is first and (dialog.dataset_id.value(), dialog.name.text()) == (4, "Cells_iter1")
        dialog.close()
    finally:
        window.close()
        application.processEvents()


def test_choice_group_keep_one_never_deselects_the_last_option() -> None:
    from qtpy.QtWidgets import QWidget

    from nnworkbench.ui.widgets import ChoiceGroup

    application = _application()
    parent = QWidget()
    group = ChoiceGroup(exclusive=False, parent=parent, keep_one=True)
    group.set_options(["0", "1", "2"], selected=["0"])
    group._buttons["1"].click()
    assert group.selected() == ["0", "1"]
    group._buttons["0"].click()
    group._buttons["1"].click()  # the last selected one stays selected
    assert group.selected() == ["1"]
    group.set_selected(["2"])
    assert group.selected() == ["2"]
    parent.close()
    parent.deleteLater()
    assert application is not None


def test_labels_editor_adds_ignore_with_the_highest_id_once() -> None:
    from qtpy.QtWidgets import QWidget

    from nnworkbench.ui.json_editor import _PairEditor

    application = _application()
    parent = QWidget()
    editor = _PairEditor({"background": 0, "cell": 1, "nucleus": 2}, value_is_id=True, add_text="＋ add label",
                         add_ignore=True, parent=parent)
    assert editor._add_ignore.isEnabled()
    editor._add_ignore.click()
    assert editor.value() == {"background": 0, "cell": 1, "nucleus": 2, "ignore": 3}
    assert not editor._add_ignore.isEnabled()  # only one ignore label
    editor._remove(editor._rows[-1])
    assert editor._add_ignore.isEnabled()
    parent.close()
    parent.deleteLater()
    assert application is not None


def test_suggest_gives_the_same_commands_as_the_gui(tmp_path: Path) -> None:
    import shlex

    from nnworkbench.cli_tools import suggestion

    application = _application()
    window = MainWindow()
    try:
        record, state = _results_dataset(tmp_path)
        window.dataset_page.record = record
        window.dataset_page._apply_state(state)
        roots = {"raw": tmp_path / "raw", "preprocessed": tmp_path / "preprocessed", "results": tmp_path / "results"}
        pages = window.dataset_page.pages

        pages["train"]._open_defaults()  # ▶ Train… in the header
        gui_train = pages["train"]._command_text()[0]
        pages["train"]._run_dialog.close()
        suggested = suggestion(roots, 8, "train")
        assert suggested["arguments"][1:] == shlex.split(gui_train)

        gui_predict = pages["predict"]._command_text()  # the GUI remembers the last folders
        folders = {"input_directory": pages["predict"]._input.text().strip(),
                   "output_directory": pages["predict"]._output.text().strip()}
        assert suggestion(roots, 8, "predict", **folders)["arguments"][1:] == shlex.split(gui_predict)

        gui_preprocess = pages["preprocess"]._command_text()
        assert suggestion(roots, 8, "preprocess")["arguments"][1:] == shlex.split(gui_preprocess)
    finally:
        window.close()
        window.deleteLater()
    assert application is not None


def test_description_dialog_views_edits_and_saves(tmp_path: Path) -> None:
    from qtpy.QtWidgets import QWidget

    from nnworkbench.ui.description_dialog import DescriptionDialog

    application = _application()
    path = tmp_path / "dataset-description.md"
    path.write_text("# Cells\n\n*Type:* confocal\n", encoding="utf-8")
    parent = QWidget()
    dialog = DescriptionDialog(path, parent)
    assert "Cells" in dialog._viewer.toPlainText()  # shown rendered
    dialog._start_edit()
    assert dialog._editor.toPlainText().startswith("# Cells")
    dialog._editor.setPlainText("# Cells\n\nType: confocal, 40 volumes\n")
    dialog._save_edit()
    assert "40 volumes" in path.read_text(encoding="utf-8") and "40 volumes" in dialog._viewer.toPlainText()
    dialog.close()
    parent.close()
    parent.deleteLater()
    assert application is not None


def test_description_tab_creates_fills_and_saves_the_form(tmp_path: Path) -> None:
    import json

    from qtpy.QtWidgets import QWidget

    from nnworkbench.ui.description_panel import DescriptionPanel

    application = _application()
    folder = tmp_path / "Dataset042_Cells"; (folder / "imagesTr").mkdir(parents=True)
    (folder / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "cell": 1}}), encoding="utf-8")
    parent = QWidget()
    panel = DescriptionPanel(parent)
    panel.set_dataset(folder)
    assert panel._stack.currentIndex() == 0  # no description yet: Create description
    panel._create()
    assert panel.editing and "type" in panel._editors
    sections = {name: panel._class_sections[token] for token, _value, name in panel._class_table.rows()}
    sections["cell"].editors["visual"].text.setPlainText("Bright membrane; touching cells both count.")
    # the class sections follow the table: a new row gets a section, a renamed row keeps its text
    panel._class_table._add_row(2, "nucleus")
    assert len(panel._class_sections) == 3
    row = next(entry for entry in panel._class_table._rows if entry[3].text() == "cell")
    row[3].setText("cells")
    assert panel._form_answers()["class:cells:visual"].text == "Bright membrane; touching cells both count."
    # the form opens folded (an overview); a class added while editing opens; folded sections show a summary
    assert all(card.collapsed for card in panel._cards)
    assert not panel._class_sections[next(token for token, _value, name in panel._class_table.rows()
                                          if name == "nucleus")].collapsed
    panel._set_all_collapsed(True)
    assert all(card.collapsed for card in panel._cards) and "1 of 4 answered" in panel._class_sections[
        next(token for token, _value, name in panel._class_table.rows() if name == "cells")]._summary_label.text()
    panel._set_all_collapsed(False)
    panel._channel_table._add_row(1, "DAPI")
    channels_row, channels_text = panel._sync_rows["channels"]
    assert not channels_row.isHidden() and "channel 1 (DAPI) is only in the description" in channels_text.text()
    assert panel._save_form()
    text = (folder / "nnworkbench" / "dataset-description.md").read_text(encoding="utf-8")
    assert "| 1 | DAPI |" in text
    assert not panel.editing and "touching cells both count" in panel._viewer.toPlainText()
    assert "### 1. cells" in text and "### 2. nucleus" in text and "touching cells both count" in text
    parent.close()
    parent.deleteLater()
    assert application is not None


def test_description_tab_warns_about_class_differences_and_pushes(tmp_path: Path, monkeypatch) -> None:
    import json

    from qtpy.QtWidgets import QMessageBox, QWidget

    from nnworkbench.ui.description_panel import DescriptionPanel

    application = _application()
    folder = tmp_path / "Dataset042_Cells"; (folder / "imagesTr").mkdir(parents=True)
    (folder / "dataset.json").write_text(json.dumps({"channel_names": {"0": "x"}, "labels": {"background": 0, "cell": 1},
                                                     "file_ending": ".nii.gz"}), encoding="utf-8")
    parent = QWidget()
    panel = DescriptionPanel(parent)
    pushed = []
    panel.dataset_json_changed.connect(lambda: pushed.append(True))
    panel.set_dataset(folder)
    panel._create()
    assert panel._sync.isHidden()  # same classes
    panel._class_table._add_row(2, "nucleus")
    assert not panel._sync.isHidden() and "nucleus (2) is only in the description" in panel._sync_text.text()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    panel._push_classes()
    assert json.loads((folder / "dataset.json").read_text(encoding="utf-8"))["labels"]["nucleus"] == 2
    assert pushed == [True] and panel._sync.isHidden() and not panel.editing  # saved with the form
    parent.close()
    parent.deleteLater()
    assert application is not None


def test_start_screen_marks_unset_paths_orange_and_missing_paths_red(tmp_path: Path) -> None:
    from nnworkbench.ui.overview_page import PathsCard

    application = _application()
    card = PathsCard()
    try:
        card.set_paths("", "", "")
        assert [card._marks[key].objectName() for key in ("raw", "preprocessed", "results")] == ["statusUnset"] * 3
        assert card._status.text() == "3 not set" and card._status.property("tone") == "orange"
        card.set_paths(str(tmp_path), "", str(tmp_path / "missing"))
        assert [card._marks[key].objectName() for key in ("raw", "preprocessed", "results")] == [
            "statusYes", "statusUnset", "statusNo"]
        assert card._status.text() == "1 not found · 1 not set" and card._status.property("tone") == "red"
        for name in ("raw", "preprocessed", "results"):
            (tmp_path / name).mkdir()
        card.set_paths(*(str(tmp_path / name) for name in ("raw", "preprocessed", "results")))
        assert card._status.property("tone") == "green"
        card.set_paths(str(tmp_path), str(tmp_path / "preprocessed"), str(tmp_path / "results"))
        assert card._status.text() == "folders overlap" and card._status.property("tone") == "red"
    finally:
        card.close()
        application.processEvents()


def test_empty_dataset_list_says_what_to_do() -> None:
    from qtpy.QtWidgets import QLabel

    application = _application()
    window = MainWindow()
    overview = window.overview
    try:
        overview.set_records([])
        assert overview._table_stack.currentWidget() is overview._empty
        assert "No datasets found" in overview._empty.findChild(QLabel).text()
        overview.set_records([DatasetRecord(42, "Cells")])
        assert overview._table_stack.currentWidget() is overview._table
        overview._search.setText("liver")
        assert "No dataset matches" in overview._empty.findChild(QLabel).text()
    finally:
        window.close()
        application.processEvents()


def test_app_mark_is_drawn_in_the_app_bar_and_about() -> None:
    from qtpy.QtGui import QColor

    from nnworkbench.ui.about_dialog import AboutDialog
    from nnworkbench.ui.widgets import AppMark

    application = _application()
    window = MainWindow()
    about = AboutDialog(window)
    try:
        assert window.findChildren(AppMark) and about.findChildren(AppMark)
        image = AppMark(40, window).grab().toImage()
        colours = [QColor(image.pixel(x, y)) for x in range(40) for y in range(40)]
        assert any(c.red() > 240 and c.green() > 240 and c.blue() > 240 for c in colours)  # the white W
        assert any(c.blue() > 150 and c.red() < 120 for c in colours)  # the blue badge
    finally:
        about.close()
        window.close()
        application.processEvents()
