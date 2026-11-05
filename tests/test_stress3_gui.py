"""GUI after the user stress test (round 3: personas input P, chaos C, expert E, beginner B). Each test fails
without its fix."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtGui import QFocusEvent, QGuiApplication
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget

from nnworkbench import path_setup
from nnworkbench.models.dataset import DatasetRecord

NAME = "Dataset007_Cells"
PLANS = {"plans_name": "nnUNetPlans", "experiment_planner_used": "ExperimentPlanner",
         "configurations": {"3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres", "patch_size": [64, 64, 64],
                                           "spacing": [1.0, 1.0, 1.0], "batch_size": 2}}}


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _dataset(root: Path, dataset_id: int = 7, name: str = "Cells", cases: int = 2) -> DatasetRecord:
    raw = root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{name}"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for index in range(cases):
        (raw / "imagesTr" / f"c{index}_0000.nii.gz").touch()
        (raw / "labelsTr" / f"c{index}.nii.gz").touch()
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1}, "numTraining": cases,
        "file_ending": ".nii.gz"}), encoding="utf-8")
    preprocessed = root / "nnUNet_preprocessed" / raw.name
    (preprocessed / "nnUNetPlans_3d_fullres").mkdir(parents=True)
    (preprocessed / "nnUNetPlans.json").write_text(json.dumps(PLANS), encoding="utf-8")
    for index in range(cases):
        (preprocessed / "nnUNetPlans_3d_fullres" / f"c{index}.b2nd").touch()
    return DatasetRecord(dataset_id, name, raw, preprocessed, None)


class _Env:
    """A dataset page in a parent widget, with jobs recorded instead of started."""

    def __init__(self, root: Path, monkeypatch) -> None:
        from nnworkbench.ui.context import AppContext
        from nnworkbench.ui.dataset_page import DatasetPage
        self.application = _application()
        self.root = root
        self.holder = QWidget()
        self.context = AppContext(self.holder)
        for name in ("raw", "preprocessed", "results"):
            (root / f"nnUNet_{name}").mkdir(parents=True, exist_ok=True)
            self.context.roots[name] = root / f"nnUNet_{name}"
        self.started: list = []
        monkeypatch.setattr(self.context.jobs, "start_nnunet", lambda *args, **kwargs: self.started.append(args)
                            or True)
        self._folders = {key: self.context.store.value(key) for key in ("predict/input", "predict/output")}
        self.page = DatasetPage(self.context, self.holder)
        self.holder.resize(1600, 1000)

    def open(self, record: DatasetRecord) -> None:
        self.page.open(record)
        self.wait()

    def wait(self, seconds: float = 10.0) -> None:
        end = time.monotonic() + seconds
        tasks = self.context.tasks
        while time.monotonic() < end:
            self.application.processEvents()
            if tasks._pool.activeThreadCount() == 0 and not tasks._workers:
                self.application.processEvents()
                return
            time.sleep(0.01)

    def close(self) -> None:
        for key, value in self._folders.items():
            if value is None:
                self.context.store.remove(key)
            else:
                self.context.store.setValue(key, value)
        # as the app on quit: results of background reads still running must not reach the closed page
        self.context.tasks.shutdown()
        self.context.tasks._pool.waitForDone(10_000)
        self.holder.close()
        self.application.processEvents()


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    environment = _Env(tmp_path, monkeypatch)
    yield environment
    environment.close()


# ---------------------------------------------------------------- typing and pasting (P3-P6, P9-P12, P14-P16)

def test_p3_p6_start_screen_takes_pasted_paths_and_keeps_them_absolute(tmp_path, monkeypatch) -> None:
    from nnworkbench.ui.overview_page import PathsCard

    application = _application()
    for name in ("raw", "pre", "res"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    parent = QWidget()
    card = PathsCard(parent)
    try:
        card.set_paths(f'"{tmp_path / "raw"}"', (tmp_path / "pre").as_uri(), "res")
        assert card._status.property("tone") == "green", card._note.text()
        assert card.local_paths() == (tmp_path / "raw", tmp_path / "pre", tmp_path / "res")
    finally:
        card.close()
        parent.close()
        application.processEvents()


def test_b2_b3_start_screen_flags_overlapping_folders_and_a_parent_folder(tmp_path) -> None:
    from nnworkbench.ui.overview_page import PathsCard

    application = _application()
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        (tmp_path / name).mkdir()
    parent = QWidget()
    card = PathsCard(parent)
    try:
        card.set_paths(*(str(tmp_path / "nnUNet_raw"),) * 3)
        assert card._status.text() == "folders overlap"
        card.set_paths(str(tmp_path), str(tmp_path / "nnUNet_preprocessed"), str(tmp_path / "nnUNet_results"))
        assert card._status.text() == "folders overlap"  # nnUNet_preprocessed is inside the chosen raw folder
        other = tmp_path / "other"
        (other / "nnUNet_raw").mkdir(parents=True)
        card.set_paths(str(other), str(tmp_path / "nnUNet_preprocessed"), str(tmp_path / "nnUNet_results"))
        assert card._status.text() == "check folders" and "did you mean" in card._note.text()
    finally:
        card.close()
        parent.close()
        application.processEvents()


def test_p4_predict_resolves_pasted_folders_and_refuses_relative_ones(env, tmp_path) -> None:
    from nnworkbench.ui.stages.predict_stage import _folder

    source = tmp_path / "my input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    record = _dataset(tmp_path)
    model = tmp_path / "nnUNet_results" / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    model.mkdir(parents=True)
    (model / "checkpoint_final.pth").touch()
    env.open(DatasetRecord(7, "Cells", record.raw_path, record.preprocessed_path, model.parent.parent))
    predict = env.page.pages["predict"]
    predict._input.setText(f'"{source}"')  # Windows "Copy as path"
    predict._output.setText("my output")  # would land in the GUI's start folder
    assert _folder(predict._input) == source and not _folder(predict._output).is_absolute()
    predict._folders_changed()
    env.wait()
    predict.update_actions()
    assert not predict._predict.isEnabled() and "full path" in predict._predict.toolTip()
    predict._output.setText(str(tmp_path / "my output"))
    predict.update_actions()
    assert predict._predict.isEnabled(), predict._predict.toolTip()


def test_p9_number_field_shows_the_number_it_uses() -> None:
    from nnworkbench.ui.widgets import NumberField

    application = _application()
    parent = QWidget()
    field = NumberField(1, 64, 3, parent)
    try:
        field.setText("")
        QTest.keyClicks(field, "1,000")  # the comma is refused (in German "1.000" became 1)
        assert field.text() == "1000" and field.value() == 10  # 1000 is out of range: the last valid value
        field.setText("")
        field.focusOutEvent(QFocusEvent(QFocusEvent.Type.FocusOut))  # left empty: shows the number used again
        assert field.text() == "10"
    finally:
        parent.close()
        application.processEvents()


@pytest.mark.parametrize(("text", "reason"), [
    ("nnworkbench-run nnUNetv2_train 7 3d_fullres 0", "whole command"), ("—npz", "—"),
    ("“--npz”", "“"), ("--npz --c", "non-breaking space"), ("-pretrained_weights /a/Ana’s/w.pth", "")])
def test_p10_arguments_copied_from_documents_are_flagged(text: str, reason: str) -> None:
    from nnworkbench.ui.stages.common import arguments_error

    assert (reason in arguments_error(text)) if reason else arguments_error(text) == ""


def test_p11_pasted_trainer_name_without_spaces() -> None:
    from nnworkbench.ui.editable_dropdown import EditableDropdownComboBox

    application = _application()
    parent = QWidget()
    box = EditableDropdownComboBox(parent)
    box.setEditable(True)
    try:
        box.setCurrentText("nnUNetTrainer_250epochs \n")
        assert box.currentText() == "nnUNetTrainer_250epochs"
    finally:
        parent.close()
        application.processEvents()


def test_p12_tables_take_focus_so_ctrl_c_copies_the_cell() -> None:
    from nnworkbench.ui.widgets import make_table, table_item

    application = _application()
    parent = QWidget()
    table = make_table(["Case"])
    table.setParent(parent)
    try:
        assert table.focusPolicy() != Qt.FocusPolicy.NoFocus
        table.setRowCount(1)
        table.setItem(0, 0, table_item("cells_001"))
        parent.show()
        table.setFocus()
        table.setCurrentCell(0, 0)
        application.processEvents()
        QTest.keyClick(table, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        assert QGuiApplication.clipboard().text() == "cells_001"
    finally:
        parent.close()
        application.processEvents()


def test_p15_p16_find_shortcut_and_window_fits_the_screen(tmp_path, monkeypatch) -> None:
    from nnworkbench.ui.main_window import MainWindow

    monkeypatch.setattr(path_setup, "configuration_path", lambda: tmp_path / "paths.json")
    application = _application()
    window = MainWindow()
    try:
        window.show()
        application.processEvents()
        available = QGuiApplication.primaryScreen().availableGeometry()
        assert window.minimumHeight() <= available.height() - 40 and window.minimumWidth() <= available.width()
        window._find()
        application.processEvents()
        assert window.overview._search.hasFocus()
    finally:
        window.switcher.hide()
        window.close()
        application.processEvents()


def test_p1_napari_close_button_names_naparis_close_shortcut() -> None:
    from nnworkbench.services.napari_inspector import _close_keys

    application = _application()
    assert _close_keys() in ("Ctrl+W", "⌘W")  # Ctrl+C closed Napari: copying lost the annotation
    assert application is not None


# ---------------------------------------------------------------- editors and switching (C1, C3, C4, C6, B12)

def _json_panel(folder: Path):
    from nnworkbench.ui.json_editor import DatasetJsonPanel

    parent = QWidget()
    panel = DatasetJsonPanel(parent)
    panel.set_manifest(folder, json.loads((folder / "dataset.json").read_text(encoding="utf-8")), editable=True)
    return parent, panel


def test_c1_c3_json_editor_keeps_edits_on_reload_and_warns_about_changes_on_disk(tmp_path) -> None:
    application = _application()
    folder = _dataset(tmp_path).raw_path
    parent, panel = _json_panel(folder)
    try:
        manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        manifest["labels"]["nucleus"] = 2  # changed by a colleague after the page was loaded
        (folder / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        panel._start_edit()  # reads the file again
        assert "nucleus" in panel._editors["labels"].value()
        panel._editors["name"].setText("Renamed")
        panel.set_manifest(folder, manifest, editable=True)  # a job finished, Refresh
        assert panel._stack.currentIndex() == 1 and panel._editors["name"].text() == "Renamed"
        manifest["description"] = "changed again outside"
        (folder / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        panel._save_edit()
        assert "changed by another program" in panel._note.text()
        assert json.loads((folder / "dataset.json").read_text(encoding="utf-8"))["description"] == "changed again outside"
        panel._save_edit()  # the second Save replaces it
        assert json.loads((folder / "dataset.json").read_text(encoding="utf-8"))["name"] == "Renamed"
    finally:
        parent.close()
        application.processEvents()


def test_b12_json_editor_asks_before_changing_labels_of_existing_label_files(tmp_path, monkeypatch) -> None:
    application = _application()
    folder = _dataset(tmp_path).raw_path
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: questions.append(args[2])
                        or QMessageBox.StandardButton.Cancel)
    parent, panel = _json_panel(folder)
    try:
        panel._start_edit()
        manifest = panel._collect()
        manifest["labels"]["ignore"] = 2
        assert not panel._confirm_label_change(manifest)
        assert "no ignore voxels" in questions[-1]
    finally:
        parent.close()
        application.processEvents()


def test_c4_description_text_dialog_compares_with_the_loaded_text(tmp_path, monkeypatch) -> None:
    from nnworkbench.ui.description_dialog import DescriptionDialog

    application = _application()
    path = tmp_path / "Dataset007_Cells" / "nnworkbench" / "dataset-description.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Description\n", encoding="utf-8")
    parent = QWidget()
    dialog = DescriptionDialog(path, parent)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: asked.append(args[1])
                        or QMessageBox.StandardButton.Cancel)
    try:
        dialog._start_edit()
        path.write_text("# Changed by a colleague\n", encoding="utf-8")
        assert not dialog._has_changes()  # no own edits: Escape does not ask
        dialog._editor.setPlainText("# Mine\n")
        dialog._save_edit()
        assert asked == ["Replace changes?"] and path.read_text(encoding="utf-8") == "# Changed by a colleague\n"
    finally:
        dialog.close()
        parent.close()
        application.processEvents()


def test_c6_switching_dataset_closes_the_run_form_of_the_previous_one(env, tmp_path) -> None:
    first = _dataset(tmp_path, 7, "Cells", cases=5)
    second = _dataset(tmp_path, 8, "Nuclei", cases=5)
    env.open(first)
    train = env.page.pages["train"]
    train._open_defaults()
    assert train._run_dialog.visible
    env.page.open(second)  # still loading
    assert not train._run_dialog.visible and train.state is None
    train._start_training()
    assert env.started == []
    env.wait()


def test_e9_val_typed_as_argument_and_b8_too_few_cases(env, tmp_path) -> None:
    record = _dataset(tmp_path, cases=4)
    fold = tmp_path / "nnUNet_results" / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "checkpoint_final.pth").touch()
    log = fold / "training_log_2026_1_1_00_00_00.txt"
    log.touch()
    os.utime(log, (time.time() - 3600,) * 2)
    env.open(DatasetRecord(7, "Cells", record.raw_path, record.preprocessed_path, fold.parent.parent))
    train = env.page.pages["train"]
    model = train.state.models[0]
    train._prefill(model, model.folds[0], continue_training=False)
    assert "is finished" in train._overwrite_warning()
    train._arguments.setText("--val --npz")
    assert train._overwrite_warning() == ""
    train._arguments.setText("")
    train._fold_choice.set_selected(["1"])
    train.update_actions()
    assert not train._start.isEnabled() and "at least 5" in train._start.toolTip()
    train._fold_choice.set_selected(["all"])
    train.update_actions()
    assert train._start.isEnabled()
    train._run_dialog.close()


def test_b1_png_dataset_preselects_2d(env, tmp_path) -> None:
    record = _dataset(tmp_path)
    manifest = json.loads((record.raw_path / "dataset.json").read_text(encoding="utf-8"))
    (record.raw_path / "dataset.json").write_text(json.dumps({**manifest, "file_ending": ".png"}), encoding="utf-8")
    env.open(record)
    assert env.page.pages["preprocess"]._configurations.selected() == ["2d"]


# ---------------------------------------------------------------- decisions of 2026-10-22

@pytest.mark.parametrize("answer", ["Cancel", "Discard", "Save"])
def test_c2_unsaved_edits_ask_before_another_dataset_or_quit(env, tmp_path, monkeypatch, answer) -> None:
    record = _dataset(tmp_path)
    env.open(record)
    data = env.page.pages["data"]
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_kwargs: asked.append(args[2])
                        or getattr(QMessageBox.StandardButton, answer))
    assert env.page.confirm_leave() and asked == []  # nothing edited
    data._json._start_edit()
    assert env.page.confirm_leave() and asked == []  # opened, not changed
    data._json._editors["name"].setText("Renamed")
    assert env.page.confirm_leave() == (answer != "Cancel")
    assert "dataset.json of Dataset007_Cells" in asked[-1]
    saved = json.loads((record.raw_path / "dataset.json").read_text(encoding="utf-8")).get("name")
    assert (saved == "Renamed") == (answer == "Save")
    assert data._json.has_unsaved_changes() == (answer == "Cancel")


def test_c2_quit_asks_the_dataset_page_first(tmp_path, monkeypatch) -> None:
    from nnworkbench.ui.main_window import MainWindow

    monkeypatch.setattr(path_setup, "configuration_path", lambda: tmp_path / "paths.json")
    application = _application()
    window = MainWindow()
    try:
        monkeypatch.setattr(window.dataset_page, "confirm_leave", lambda: False)
        assert not window._confirm_quit()
        monkeypatch.setattr(window.dataset_page, "confirm_leave", lambda: True)
        assert window._confirm_quit()
    finally:
        window.switcher.hide()
        window.close()
        application.processEvents()


def test_c7_ended_gui_training_job_is_no_longer_training(tmp_path) -> None:
    from nnworkbench.ui.jobs import JobRunner

    application = _application()
    fold = tmp_path / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "training_log_2026_1_1_00_00_00.txt").touch()
    parent = QWidget()
    runner = JobRunner(parent, log_folder=tmp_path / "logs")
    try:
        runner.ended_trainings.append((["nnUNetv2_train", "7", "3d_fullres", "0"], time.time()))
        assert runner.ended_folds(tmp_path) == {fold: runner.ended_trainings[0][1]}
    finally:
        parent.close()
        application.processEvents()


def test_p13_only_the_output_changed_keeps_predict_enabled(env, tmp_path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    record = _dataset(tmp_path)
    model = tmp_path / "nnUNet_results" / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    model.mkdir(parents=True)
    (model / "checkpoint_final.pth").touch()
    env.open(DatasetRecord(7, "Cells", record.raw_path, record.preprocessed_path, model.parent.parent))
    predict = env.page.pages["predict"]
    predict._input.setText(str(source))
    predict._output.setText(str(tmp_path / "out1"))
    predict._folders_changed()
    env.wait()
    predict._output.setText(str(tmp_path / "out2"))
    predict._folders_changed()  # leaving the field: a new listing starts
    predict.update_actions()
    assert predict._predict.isEnabled(), predict._predict.toolTip()  # the click right after it works
    predict._input.setText(str(tmp_path))
    predict._folders_changed()
    predict.update_actions()
    assert not predict._predict.isEnabled()  # a new input folder waits for its listing
    env.wait()


def test_b10_preprocess_waits_for_the_data_problems(env, tmp_path) -> None:
    record = _dataset(tmp_path)
    manifest = json.loads((record.raw_path / "dataset.json").read_text(encoding="utf-8"))
    (record.raw_path / "dataset.json").write_text(json.dumps({**manifest, "numTraining": 9}), encoding="utf-8")
    env.open(record)
    preprocess = env.page.pages["preprocess"]
    preprocess._run_dialog.open()
    try:
        assert not preprocess._run.isEnabled() and "numTraining" in preprocess._run.toolTip()
    finally:
        preprocess._run_dialog.close()


def test_c12_task_runner_reports_running_tasks() -> None:
    import threading

    from nnworkbench.ui.tasks import TaskRunner

    application = _application()
    parent = QWidget()
    runner = TaskRunner(parent)
    release = threading.Event()
    try:
        runner.run("hang", release.wait, lambda _result: None)
        assert runner.active  # quitting now ends the process (app.py) instead of waiting for the share
    finally:
        release.set()
        runner._pool.waitForDone(5000)
        parent.close()
        application.processEvents()
    assert not runner.active


def test_b5_rgb_image_opens_as_one_colour_image(tmp_path) -> None:
    pytest.importorskip("mitk")
    from PIL import Image

    from nnworkbench.services.napari_inspector import _read_image

    Image.new("RGB", (60, 40), (1, 2, 3)).save(tmp_path / "c_0000.png")
    data, affine = _read_image(tmp_path / "c_0000.png")
    assert data.shape == (40, 60, 3) and affine.shape == (3, 3)  # was (3, 40, 60): three slices in Napari


def test_new_iteration_dialog_offers_linked_images(tmp_path) -> None:
    from nnworkbench.ui.new_dataset_dialog import NewDatasetDialog

    application = _application()
    parent = QWidget()
    record = DatasetRecord(42, "Cells", tmp_path)
    iteration = NewDatasetDialog(parent, sources=[record], source=record, suggest=lambda _record: (43, "Cells_iter1"))
    new = NewDatasetDialog(parent)
    try:
        assert iteration.link_images is not None and not iteration.link_images.isChecked()  # copy by default
        assert new.link_images is None and new.description is not None
    finally:
        iteration.close()
        new.close()
        parent.close()
        application.processEvents()


def test_e5_start_screen_badge_when_nnunet_variables_differ(tmp_path, monkeypatch) -> None:
    from nnworkbench.ui.overview_page import PathsCard

    application = _application()
    for name in ("raw", "pre", "res"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("nnUNet_raw", "/elsewhere/nnUNet_raw")
    parent = QWidget()
    card = PathsCard(parent)
    try:
        card.set_paths(str(tmp_path / "raw"), str(tmp_path / "pre"), str(tmp_path / "res"))
        assert card._status.text() == "✓ all found" and card._status.property("tone") == "green"
        assert card._variables.isVisibleTo(card) and "/elsewhere/nnUNet_raw" in card._variables.toolTip()
        assert str(tmp_path / "raw") in card._variables.toolTip()  # which one is used
        monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
        card.set_paths(str(tmp_path / "raw"), str(tmp_path / "pre"), str(tmp_path / "res"))
        assert not card._variables.isVisibleTo(card)
    finally:
        card.close()
        parent.close()
        application.processEvents()
