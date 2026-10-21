"""Robustness of the dataset page and the stages against odd nnU-Net files (GUI stress test T1-T13)."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtWidgets import QApplication, QWidget

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_contents import json_value, model_folds
from nnworkbench.services.workflow_status import (
    mean_class_dice,
    summary_foreground_dice,
    validation_cases,
)
from nnworkbench.ui.dataset_state import load_state

NAME = "Dataset007_Cells"
PLANS = {"plans_name": "nnUNetPlans", "experiment_planner_used": "ExperimentPlanner",
         "configurations": {"3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres", "patch_size": [64, 64, 64],
                                           "spacing": [1.0, 1.0, 1.0], "batch_size": 2}}}
POSIX_PERMISSIONS = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _raw(root: Path, cases=("a", "b")) -> Path:
    raw = root / "nnUNet_raw" / NAME
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for case in cases:
        (raw / "imagesTr" / f"{case}_0000.nii.gz").touch()
        (raw / "labelsTr" / f"{case}.nii.gz").touch()
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1}, "numTraining": len(cases),
        "file_ending": ".nii.gz"}), encoding="utf-8")
    return raw


def _preprocessed(root: Path, plans: object = PLANS, cases: int = 2, plans_text: str | None = None) -> Path:
    preprocessed = root / "nnUNet_preprocessed" / NAME
    folder = preprocessed / "nnUNetPlans_3d_fullres"
    folder.mkdir(parents=True)
    (preprocessed / "nnUNetPlans.json").write_text(plans_text if plans_text is not None else json.dumps(plans),
                                                   encoding="utf-8")
    for index in range(cases):
        (folder / f"c{index}.b2nd").touch()
        (folder / f"c{index}_seg.b2nd").touch()
    return preprocessed


def _summary(dice: object = 0.8) -> dict:
    return {"foreground_mean": {"Dice": dice}, "mean": {"1": {"Dice": dice}},
            "metric_per_case": [{"prediction_file": "/x/validation/a.nii.gz", "reference_file": "/x/gt/a.nii.gz",
                                 "metrics": {"1": {"Dice": dice}}}]}


def _model(root: Path, folds=("0",), summary_text: str | None = None, log: str | None = "",
           debug: str | None = None) -> Path:
    model = root / "nnUNet_results" / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    for fold in folds:
        folder = model / f"fold_{fold}"
        (folder / "validation").mkdir(parents=True)
        (folder / "checkpoint_final.pth").write_bytes(b"")
        if log is not None:
            (folder / "training_log_2026_1_1_00_00_00.txt").write_text(log, encoding="utf-8")
        if debug is not None:
            (folder / "debug.json").write_text(debug, encoding="utf-8")
        (folder / "validation" / "summary.json").write_text(
            summary_text if summary_text is not None else json.dumps(_summary()), encoding="utf-8")
    return model


def _record(root: Path) -> DatasetRecord:
    paths = [root / f"nnUNet_{name}" / NAME for name in ("raw", "preprocessed", "results")]
    return DatasetRecord(7, "Cells", *(path if path.is_dir() else None for path in paths))


class _Env:
    """A dataset page in a parent widget, with jobs recorded instead of started."""

    def __init__(self, root: Path, monkeypatch) -> None:
        from nnworkbench.ui.context import AppContext
        from nnworkbench.ui.dataset_page import DatasetPage
        from nnworkbench.ui.stages import preprocess_stage, train_stage

        monkeypatch.setattr(preprocess_stage, "available_planners", lambda: ("ExperimentPlanner",))
        monkeypatch.setattr(train_stage, "available_trainers", lambda: ("nnUNetTrainer",))
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
        # the last prediction folders are kept in the (shared test) settings: restored when closing
        self._folders = {key: self.context.store.value(key) for key in ("predict/input", "predict/output")}
        self.page = DatasetPage(self.context, self.holder)
        self.holder.resize(1600, 1000)

    def open(self) -> DatasetRecord:
        record = _record(self.root)
        self.page.open(record)
        self.wait()
        return record

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
        self.context.shutdown()
        self.holder.close()
        self.holder.deleteLater()
        self.application.processEvents()


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    environment = _Env(tmp_path, monkeypatch)
    yield environment
    environment.close()


# ---------------------------------------------------------------- T1 encodings

def test_t1_json_that_is_not_utf8_or_nested_too_deeply_is_unavailable(tmp_path: Path) -> None:
    utf16 = tmp_path / "utf16.json"
    utf16.write_bytes(json.dumps({"a": 1}).encode("utf-16"))
    nested = tmp_path / "nested.json"
    nested.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")
    assert json_value(utf16) is None and json_value(nested) is None
    _raw(tmp_path)
    (tmp_path / "nnUNet_raw" / NAME / "dataset.json").write_bytes(
        json.dumps({"channel_names": {"0": "a"}, "labels": {"background": 0, "x": 1}}).encode("utf-16"))
    preprocessed = _preprocessed(tmp_path)
    (preprocessed / "splits_final.json").write_bytes("[{}, {}]".encode("utf-16"))
    (preprocessed / "latin.json").write_bytes(b'{"configurations": {"\xe9": {}}}')
    model = _model(tmp_path, debug=None)
    (model / "fold_0" / "debug.json").write_bytes(json.dumps({"num_epochs": 5}).encode("utf-16"))
    (model / "fold_0" / "validation" / "summary.json").write_bytes(json.dumps(_summary()).encode("utf-16"))
    state = load_state(_record(tmp_path))
    assert "dataset.json cannot be read (no valid JSON object)." in state.manifest_errors
    assert state.fold_count == 5 and len(state.plans) == 1 and state.models[0].folds[0].dice is None


# ---------------------------------------------------------------- T2 broken links, unreadable folders

@pytest.mark.skipif(not POSIX_PERMISSIONS, reason="needs POSIX permissions and symlinks (not as root)")
def test_t2_broken_log_symlink_and_unreadable_folders(tmp_path: Path) -> None:
    _raw(tmp_path)
    preprocessed = _preprocessed(tmp_path)
    model = _model(tmp_path, folds=("0", "1"), log=None)
    os.symlink(tmp_path / "missing.txt", model / "fold_0" / "training_log_2026.txt")
    locked = [model.parent / "locked", model / "fold_1", preprocessed / "nnUNetPlans_3d_fullres"]
    locked[0].mkdir()
    for folder in locked:
        folder.chmod(0)
    try:
        state = load_state(_record(tmp_path))
    finally:
        for folder in locked:
            folder.chmod(0o755)
    folds = {fold.name: fold for fold in state.models[0].folds}
    assert folds["0"].status == "reduced" and folds["0"].dice == pytest.approx(0.8)
    assert folds["1"].status == "stopped"  # unreadable: nothing known about it
    assert state.plans[0].configurations[0].preprocessed_cases == 0


# ---------------------------------------------------------------- T3 odd plans and huge numbers

def test_t3_plans_with_odd_numbers_and_huge_values(env: _Env) -> None:
    _raw(env.root)
    plans = json.loads(json.dumps(PLANS))
    plans["configurations"]["3d_fullres"].update(patch_size=128, spacing=["a", None, 2], batch_size=True)
    _preprocessed(env.root, plans)
    _model(env.root, debug='{"num_epochs": 1e400}', log="x: Epoch " + "9" * 5000 + "\nx: Pseudo dice [1e999, 0.5]\n")
    state = load_state(_record(env.root))
    config = state.plans[0].configurations[0]
    assert config.patch_size == () and config.spacing == ("a", None, 2) and config.batch_size is None
    fold = state.models[0].folds[0]
    assert fold.epoch is None and fold.total_epochs is None and fold.pseudo_dice == pytest.approx(0.5)
    preprocess = env.page.pages["preprocess"]
    env.page.record = state.record
    preprocess.set_state(state)  # raised: format code 'g' for a string
    assert "spacing a × None × 2" in preprocess._plans_table.item(1, 0).text()


# ---------------------------------------------------------------- T4 one failing stage, failed load

def test_t4_failing_stage_does_not_block_the_others_and_load_errors_are_shown(env: _Env, monkeypatch) -> None:
    from nnworkbench.ui import dataset_page

    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root)
    record = _record(env.root)
    state = load_state(record)
    env.page.record = record

    def broken(_state):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(env.page.pages["train"], "set_state", broken)
        env.page._apply_state(state)
    assert env.page.pages["validate"].state is state and env.page.pages["predict"].state is state
    assert "boom" in env.page.pages["train"]._sub.text()

    def unreadable(_record):
        raise OSError("disk gone")

    monkeypatch.setattr(dataset_page, "load_state", unreadable)
    env.page.record = None
    env.page.open(record)
    env.wait()
    for page in env.page.pages.values():
        assert "disk gone" in page._sub.text() and "Refresh" in page._sub.text()


# ---------------------------------------------------------------- T5 additional arguments that do not split

def test_t5_unsplittable_arguments_disable_run_and_show_the_error(env: _Env) -> None:
    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root)
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    preprocess, train, predict = (env.page.pages[key] for key in ("preprocess", "train", "predict"))
    predict._input.setText(str(source))
    predict._output.setText(str(env.root / "output"))
    predict._folders_changed()
    env.wait()
    preprocess._run_dialog.open()
    train._open_defaults()
    try:
        for stage, run in ((preprocess, preprocess._run), (train, train._start), (predict, predict._predict)):
            stage._arguments.setText("--npz")
            assert run.isEnabled() and "--npz" in stage._command.command()
            stage._arguments.setText("-foo 'bar")
            assert stage._command.command() == ""  # no stale command to copy or run
            assert "cannot be read" in stage._command.toPlainText()
            assert not stage._command._copy_buttons[0].isEnabled() and not run.isEnabled()
            run.click()
            {preprocess: preprocess._run_preprocessing, train: train._start_training,
             predict: predict._run_prediction}[stage]()
            stage._arguments.setText("")
        assert env.started == []
    finally:
        preprocess._run_dialog.close()
        train._run_dialog.close()


# ---------------------------------------------------------------- T6 fold folder names

def test_t6_only_nnunet_fold_folders_and_folds_passed_as_a_list(tmp_path: Path) -> None:
    from nnworkbench.services.commands import prediction_command

    model = _model(tmp_path, folds=("0", "all", "it's", "a b", "x"))
    assert [path.name for path in model_folds(model)] == ["fold_0", "fold_all"]
    command = prediction_command(1, "/i", "/o", "P", "2d", folds=["0", "a b"])
    assert "-f 0 'a b' -npp" in command
    assert "-f 0 2 -npp" in prediction_command(1, "/i", "/o", "P", "2d", folds="0 2")


# ---------------------------------------------------------------- T7 foreground_mean that is not an object

def test_t7_validate_with_foreground_mean_null(env: _Env) -> None:
    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root, summary_text=json.dumps({"foreground_mean": None, "mean": [], "metric_per_case": []}))
    env.open()
    validate = env.page.pages["validate"]
    model = validate._model_rows[0]
    fold = model.folds[0]
    validate._fold = fold
    validate._apply_summary(fold, {"foreground_mean": None, "mean": None})  # raised AttributeError
    assert validate._foreground.text() == "—"
    validate._apply_summary(fold, {"foreground_mean": [1]})
    assert validate._foreground.text() == "—"


# ---------------------------------------------------------------- T8 folder gone before it is expanded

def test_t8_expanding_a_removed_folder(env: _Env) -> None:
    import shutil

    _raw(env.root)
    preprocessed = _preprocessed(env.root)
    env.open()
    preprocess = env.page.pages["preprocess"]
    tree = preprocess._tree
    item = next(tree.topLevelItem(index) for index in range(tree.topLevelItemCount())
                if tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole))
    shutil.rmtree(preprocessed / "nnUNetPlans_3d_fullres")
    preprocess._expand_folder(item)  # raised FileNotFoundError
    assert item.childCount() == 1 and item.child(0).text(0).startswith("Cannot list this folder")


# ---------------------------------------------------------------- T9 unreadable input folder

def test_t9_input_folder_that_cannot_be_listed_clears_the_cases(env: _Env, monkeypatch) -> None:
    from nnworkbench.ui.stages import predict_stage

    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root)
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    predict = env.page.pages["predict"]
    predict._input.setText(str(source))
    predict._output.setText(str(env.root / "output"))
    predict._folders_changed()
    env.wait()
    assert predict._table.rowCount() == 1 and predict._predict.isEnabled()

    def denied(_source, _target):
        raise PermissionError(13, "Permission denied", str(source))

    monkeypatch.setattr(predict_stage, "prediction_status", denied)
    predict._folders_changed()
    env.wait()
    assert predict._table.rowCount() == 0 and not predict._predict.isEnabled()
    assert "Permission denied" in predict._images_note.text() and "Permission denied" in predict._busy_hint.text()


# ---------------------------------------------------------------- T10 integers too large for a float

def test_t10_integer_dice_with_400_digits(tmp_path: Path) -> None:
    huge = 10 ** 400
    summary = _summary(huge)
    assert summary_foreground_dice(summary) is None
    assert mean_class_dice(summary) == []
    assert validation_cases(summary)[0].mean_dice is None and validation_cases(summary)[0].weakest is None
    _raw(tmp_path)
    _model(tmp_path, summary_text=json.dumps(summary))
    assert load_state(_record(tmp_path)).models[0].folds[0].dice is None


# ---------------------------------------------------------------- T11 large preprocessed folders

def test_t11_expanding_a_folder_with_3000_cases_stays_responsive(env: _Env) -> None:
    _raw(env.root)
    preprocessed = _preprocessed(env.root, cases=3000)
    for index in range(3000):
        (preprocessed / "nnUNetPlans_3d_fullres" / f"c{index}.pkl").touch()
    env.open()
    preprocess = env.page.pages["preprocess"]
    tree = preprocess._tree
    item = next(tree.topLevelItem(index) for index in range(tree.topLevelItemCount())
                if str(tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)).endswith("nnUNetPlans_3d_fullres"))
    start = time.monotonic()
    item.setExpanded(True)
    while item.childCount() < 9000 and time.monotonic() - start < 1.0:
        env.application.processEvents()
    elapsed = time.monotonic() - start
    assert item.childCount() == 9000 and elapsed < 1.0
    opened = []
    preprocess.context.inspect_preprocessed = opened.append
    preprocess._file_activated(item.child(0))
    assert opened and opened[0].name.endswith(".b2nd")


# ---------------------------------------------------------------- T12 unusable prediction folders

def test_t12_predict_needs_cases_and_an_output_other_than_the_input(env: _Env) -> None:
    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root)
    empty = env.root / "empty"
    empty.mkdir()
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    predict = env.page.pages["predict"]
    for input_folder, output_folder, reason in ((empty, env.root / "out", "no images"),
                                                (source, source, "different from the input")):
        predict._input.setText(str(input_folder))
        predict._output.setText(str(output_folder))
        predict._folders_changed()
        env.wait()
        assert not predict._predict.isEnabled() and reason in predict._busy_hint.text()
    predict._output.setText("~/predictions")
    predict._folders_changed()
    env.wait()
    assert predict._predict.isEnabled()
    arguments = shlex.split(predict._command.command())
    assert arguments[arguments.index("-o") + 1] == str(Path("~/predictions").expanduser())  # no shell expands ~


# ---------------------------------------------------------------- T13 status --json

def test_t13_status_json_has_no_infinity(tmp_path: Path) -> None:
    from nnworkbench.status_report import build_status

    _raw(tmp_path)
    _model(tmp_path, summary_text='{"foreground_mean": {"Dice": Infinity}}', log="x: Pseudo dice [1e999]\n")
    status = build_status(load_state(_record(tmp_path)))
    json.dumps(status, allow_nan=False)  # raised ValueError: Infinity is not valid JSON
    assert status["models"][0]["folds"][0]["dice"] is None
