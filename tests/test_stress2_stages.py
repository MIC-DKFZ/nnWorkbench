"""Stages at scale and at the minimum window size (GUI stress test round 2: R2, R7–R9, R14–R16, R18, R23–R25)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtWidgets import QApplication, QScrollArea, QWidget

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_json import definition_errors
from nnworkbench.services.workflow_status import ignore_label_id, prediction_status

NAME = "Dataset007_Cells"
PLANS = {"plans_name": "nnUNetPlans", "experiment_planner_used": "ExperimentPlanner",
         "configurations": {"3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres", "patch_size": [64, 64, 64],
                                           "spacing": [1.0, 1.0, 1.0], "batch_size": 2}}}
LONG_TRAINER = "nnUNetTrainer_variant07_" + "x" * 40


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _raw(root: Path, classes: int = 1, channels: int = 1, cases=("a", "b"), labels: dict | None = None) -> Path:
    raw = root / "nnUNet_raw" / NAME
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for case in cases:
        for channel in range(channels):
            (raw / "imagesTr" / f"{case}_{channel:04d}.nii.gz").touch()
        (raw / "labelsTr" / f"{case}.nii.gz").touch()
    names = labels or ({"background": 0} | {f"class_{index:03d}_with_a_long_descriptive_name": index
                                            for index in range(1, classes + 1)})
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {str(index): f"channel{index}" for index in range(channels)}, "labels": names,
        "numTraining": len(cases), "file_ending": ".nii.gz"}), encoding="utf-8")
    return raw


def _preprocessed(root: Path, plans_name: str = "nnUNetPlans") -> Path:
    preprocessed = root / "nnUNet_preprocessed" / NAME
    plans = json.loads(json.dumps(PLANS))
    plans["plans_name"] = plans_name
    identifier = f"{plans_name}_3d_fullres"
    plans["configurations"]["3d_fullres"]["data_identifier"] = identifier
    (preprocessed / identifier).mkdir(parents=True)
    (preprocessed / f"{plans_name}.json").write_text(json.dumps(plans), encoding="utf-8")
    for index in range(2):
        (preprocessed / identifier / f"c{index}.b2nd").touch()
    return preprocessed


def _summary(classes: int, cases: list[str], seed: int = 0) -> dict:
    """Dice of class k in case i: a fixed pattern, so the expected values can be computed here."""
    def dice(case: int, label: int) -> float:
        return ((case * 7 + label * 13 + seed) % 97) / 100

    per_case = [{"prediction_file": f"/cluster/validation/{case}.nii.gz", "reference_file": f"/cluster/gt/{case}.nii.gz",
                 "metrics": {str(label): {"Dice": dice(index, label)} for label in range(1, classes + 1)}}
                for index, case in enumerate(cases)]
    mean = {str(label): {"Dice": sum(dice(index, label) for index in range(len(cases))) / len(cases)}
            for label in range(1, classes + 1)}
    foreground = sum(entry["Dice"] for entry in mean.values()) / classes
    return {"foreground_mean": {"Dice": foreground}, "mean": mean, "metric_per_case": per_case}


def _model(root: Path, classes: int, folds=("0", "1", "all"), trainer: str = "nnUNetTrainer",
           plans: str = "nnUNetPlans", cases_per_fold: int = 4) -> Path:
    model = root / "nnUNet_results" / NAME / f"{trainer}__{plans}__3d_fullres"
    for number, fold in enumerate(folds):
        folder = model / f"fold_{fold}"
        (folder / "validation").mkdir(parents=True)
        (folder / "checkpoint_final.pth").write_bytes(b"")
        (folder / "training_log_2026_1_1_00_00_00.txt").write_text("", encoding="utf-8")
        # fold all validates on training cases: the same case names as the other folds
        start = 0 if fold == "all" else number * cases_per_fold
        cases = [f"case_{index:03d}" for index in range(start, start + cases_per_fold)]
        (folder / "validation" / "summary.json").write_text(json.dumps(_summary(classes, cases, number)),
                                                           encoding="utf-8")
    return model


def _record(root: Path) -> DatasetRecord:
    paths = [root / f"nnUNet_{name}" / NAME for name in ("raw", "preprocessed", "results")]
    return DatasetRecord(7, "Cells", *(path if path.is_dir() else None for path in paths))


class _Env:
    """A dataset page in a shown parent widget, with jobs recorded instead of started."""

    def __init__(self, root: Path, monkeypatch, size=(1200, 640)) -> None:
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
        self._folders = {key: self.context.store.value(key) for key in ("predict/input", "predict/output")}
        self._howto = self.context.howto_expanded
        self.context.set_howto_expanded(True)
        self.page = DatasetPage(self.context, self.holder)
        self.holder.resize(*size)
        self.page.resize(*size)
        self.holder.show()

    def open(self, stage: str = "") -> QWidget | None:
        self.page.open(_record(self.root))
        self.wait()
        if stage:
            self.page.show_stage(stage)
            self.wait()
            return self.page.pages[stage]
        return None

    def wait(self, seconds: float = 20.0) -> None:
        end = time.monotonic() + seconds
        tasks = self.context.tasks
        while time.monotonic() < end:
            self.application.processEvents()
            if tasks._pool.activeThreadCount() == 0 and not tasks._workers:
                self.application.processEvents()
                return
            time.sleep(0.01)

    def resize(self, width: int, height: int) -> None:
        self.holder.resize(width, height)
        self.page.resize(width, height)
        for _ in range(5):
            self.application.processEvents()

    def close(self) -> None:
        for key, value in self._folders.items():
            if value is None:
                self.context.store.remove(key)
            else:
                self.context.store.setValue(key, value)
        self.context.set_howto_expanded(self._howto)
        self.context.shutdown()
        # as the app on quit: results of background reads still running must not reach the closed page
        self.context.tasks.shutdown()
        self.context.tasks._pool.waitForDone(10_000)
        self.holder.close()
        self.holder.deleteLater()
        self.application.processEvents()


# ---------------------------------------------------------------- R2 / R14 / R23 Validate with 500 classes

def test_r2_validate_500_classes_fast_correct_and_scrollable(tmp_path: Path, monkeypatch) -> None:
    classes = 500
    _raw(tmp_path, classes=classes)
    _model(tmp_path, classes)
    env = _Env(tmp_path, monkeypatch)
    try:
        stage = env.open("validate")
        names = env.page.pages["validate"].state.names
        model = stage._model_rows[0]
        fold = next(item for item in model.folds if item.name == "0")
        start = time.perf_counter()
        stage._fold_clicked(model, fold)
        env.wait()
        fold_seconds = time.perf_counter() - start
        start = time.perf_counter()
        stage._select_case(0)
        env.application.processEvents()
        case_seconds = time.perf_counter() - start
        # one widget row per class took 13.5 s per click with 500 classes
        assert fold_seconds < 3.0 and case_seconds < 2.0, (fold_seconds, case_seconds)
        summary = json.loads((fold.path / "validation" / "summary.json").read_text(encoding="utf-8"))
        entry = next(item for item in summary["metric_per_case"] if stage._cases[0].case_id in item["reference_file"])
        expected = {names[int(key)]: value["Dice"] for key, value in entry["metrics"].items()}
        assert dict(stage._class_rows) == expected  # every class with its name and value
        shown = stage._dice_bars.rows
        weakest = sorted(expected.items(), key=lambda item: item[1])[:len(shown)]
        assert 0 < len(shown) <= 20 and [row[0] for row in shown] == [name for name, _dice in weakest]
        assert all(row[2] == f"{dice:.2f}" and name in row[3] for row, (name, dice) in zip(shown, weakest))
        stage._show_all.click()
        assert [row[0] for row in stage._dice_bars.rows] == list(expected)  # all, in label order
        assert stage._dice_bars.height() == classes * stage._dice_bars.ROW_HEIGHT
        # the panel scrolls instead of drawing the classes over the details (R14)
        area = stage.right.widget(0)
        assert isinstance(area, QScrollArea) and stage._details.isVisibleTo(area.widget())
        # weakest class column: the Dice first, so a long name never hides it (R23)
        case = stage._cases[0]
        assert stage._cases_table.item(0, 2).text() == f"{case.weakest[1]:.2f} · {case.weakest[0]}"
        # the subject names the plans too (R24)
        stage._show_model(model)
        assert "nnUNetPlans" in stage._panel_subject.text().replace("\u200b", "")
    finally:
        env.close()


# ---------------------------------------------------------------- R7 the ignore key is exactly "ignore"

def test_r7_ignore_key_must_be_exact() -> None:
    assert ignore_label_id({"labels": {"background": 0, "cell": 1, "ignore": 2}}) == 2
    for key in ("Ignore", " ignore ", "IGNORE"):
        manifest = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "cell": 1, key: 2}}
        assert ignore_label_id(manifest) is None  # a normal class for nnU-Net
        errors = definition_errors(manifest)
        assert any(f'"{key}"' in error and '"ignore"' in error for error in errors), errors
    assert definition_errors({"channel_names": {"0": "CT"}, "labels": {"background": 0, "ignore": 1}}) == ()


def test_r7_misspelled_ignore_is_a_dense_dataset_problem(tmp_path: Path) -> None:
    from nnworkbench.ui.dataset_state import load_state

    _raw(tmp_path, labels={"background": 0, "cell": 1, "Ignore": 2})
    state = load_state(_record(tmp_path))
    assert state.ignore_id is None and not state.sparse
    assert any('"Ignore"' in problem for problem in state.data_problems())


# ---------------------------------------------------------------- R8 validation CLI

def test_r8_validation_report_uses_cross_validation_folds_only(tmp_path: Path, capsys) -> None:
    from nnworkbench.cli_tools import run_validation, validation_report

    _raw(tmp_path, classes=30)
    _model(tmp_path, 30)  # folds 0 and 1 plus fold all with the cases of fold 0 again
    roots = {name: tmp_path / f"nnUNet_{name}" for name in ("raw", "preprocessed", "results")}
    [model] = validation_report(roots, 7)["models"]
    folds = [case["fold"] for case in model["cases"]]
    cases = [case["case"] for case in model["cases"]]
    assert "all" not in folds and len(cases) == len(set(cases)) == 8  # every case once
    assert set(model["folds"]) == {"0", "1", "all"}  # the Dice of fold all is still listed
    summaries = [json.loads((tmp_path / "nnUNet_results" / NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres"
                             / f"fold_{fold}" / "validation" / "summary.json").read_text(encoding="utf-8"))
                 for fold in ("0", "1")]
    first = next(name for name in model["classes"])
    assert abs(model["classes"][first] - sum(item["mean"]["1"]["Dice"] for item in summaries) / 2) < 1e-12
    [only_all] = validation_report(roots, 7, fold="all")["models"]
    assert {case["fold"] for case in only_all["cases"]} == {"all"} and len(only_all["cases"]) == 4
    run_validation(roots, 7, limit=3)
    text = capsys.readouterr().out
    class_lines = [line for line in text.splitlines() if line.startswith(("      0.", "      1."))]
    assert "weakest classes (20 of 30)" in text and len(class_lines) == 20  # one class per line
    values = [float(line.split()[0]) for line in class_lines]
    assert values == sorted(values)  # weakest first
    case_lines = [line for line in text.splitlines() if line.startswith("      case_")]
    assert "fold all excluded" in text and len(case_lines) == 3 and not any("fold all" in line for line in case_lines)


# ---------------------------------------------------------------- R9 prediction input needs every channel

def test_r9_prediction_status_needs_every_channel(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    for name in ("full_0000", "full_0001", "partial_0001", "nosuffix"):
        (source / f"{name}.nii.gz").touch()
    rows = prediction_status(source, None, channels=2)
    assert rows == [("full", True, False), ("nosuffix", False, False), ("partial", False, False)]
    assert rows[2].missing == ("0000",) and rows[1].unsuffixed
    assert prediction_status(source, None)[0] == ("full", True, False)  # one channel: _0000 is enough


def test_r9_predict_disabled_with_reason(tmp_path: Path, monkeypatch) -> None:
    _raw(tmp_path, channels=2)
    _model(tmp_path, 1, folds=("0",))
    source = tmp_path / "input"
    source.mkdir()
    for name in ("x_0000", "x_0001", "y_0000", "scan"):
        (source / f"{name}.nii.gz").touch()
    env = _Env(tmp_path, monkeypatch, size=(1600, 1000))
    try:
        predict = env.open("predict")
        predict._input.setText(str(source))
        predict._output.setText(str(tmp_path / "output"))
        predict._folders_changed()
        env.wait()
        marks = {predict._table.item(row, 0).text(): predict._table.item(row, 1).text()
                 for row in range(predict._table.rowCount())}
        assert marks == {"scan": "✗", "x": "✓", "y": "✗"}
        assert not predict._predict.isEnabled()
        reason = predict._predict.toolTip()
        assert "y: _0001 missing" in reason and "without channel suffix" in reason and "scan" in reason
        assert "_0001" in predict._table.item(2, 1).toolTip()
        (source / "y_0001.nii.gz").touch()
        (source / "scan.nii.gz").rename(source / "scan_0000.nii.gz")
        (source / "scan_0001.nii.gz").touch()
        predict._folders_changed()
        env.wait()
        assert predict._predict.isEnabled(), predict._predict.toolTip()
    finally:
        env.close()


# ---------------------------------------------------------------- R15 / R24 / R16 / R18 layout at 1200 x 700

def _three_models(root: Path) -> None:
    _raw(root, classes=2)
    _preprocessed(root)
    for index in range(3):
        _model(root, 2, folds=("0", "1", "2", "3", "4", "all"), trainer=LONG_TRAINER.replace("07", f"0{index}"),
               plans="nnUNetResEncUNetLPlans_custom_patchsize_192x192x192")


def test_r15_r24_models_tables_fit_the_minimum_window(tmp_path: Path, monkeypatch) -> None:
    from nnworkbench.ui.stages.common import MODEL_MIN_WIDTH

    _three_models(tmp_path)
    env = _Env(tmp_path, monkeypatch, size=(1200, 600))  # 1200 x 700 minus top bar, tabs and console
    try:
        env.open()
        for key in ("train", "validate", "predict"):
            env.page.show_stage(key)
            env.wait()
            stage = env.page.pages[key]
            table = stage._models_table if key == "predict" else stage._models
            # no fixed height: the table shrinks (and scrolls) instead of running under the cards below
            assert table.minimumHeight() < table.maximumHeight()
            assert table.height() <= table.maximumHeight()
            folds = table.columnWidth(2)
            needed = max(table.cellWidget(row, 2).minimumWidth() for row in range(table.rowCount()))
            assert folds >= needed, (key, folds, needed)  # all folds incl. 4 and all
            assert table.columnWidth(0) >= MODEL_MIN_WIDTH
            # model, mean Dice and folds fit next to each other: no sideways scrolling at 1200 px
            assert sum(table.columnWidth(column) for column in range(3)) <= table.viewport().width(), key
            first = table.cellWidget(0, 0).findChildren(QWidget)[0]
            # the start stays visible (elided at the end); with the app's 12 px mono font it shows "variant00"
            assert first.text().startswith("nnUNetTrainer_") and first._mode == Qt.TextElideMode.ElideRight
            assert first.toolTip().startswith(LONG_TRAINER.replace("07", "00"))
            center = stage.center.parentWidget()
            assert center.geometry().right() < stage.right.geometry().left()
            # nothing below the models table overlaps it
            card = table.parentWidget()
            below = [child for child in card.findChildren(QWidget) if child.isVisible()
                     and child.parentWidget() is card and child.y() > table.y()]
            assert all(child.y() >= table.geometry().bottom() for child in below), key
        # every stage fits the minimum window width, also after being shown wide
        env.resize(1920, 1000)
        for key in ("preprocess", "train", "validate", "predict"):
            env.page.show_stage(key)
            env.wait()
        env.resize(1200, 600)
        for key in ("preprocess", "train", "validate", "predict"):
            assert env.page.pages[key].minimumSizeHint().width() <= 1200, key
    finally:
        env.close()


def test_r16_plans_table_columns(tmp_path: Path, monkeypatch) -> None:
    from qtpy.QtGui import QFontMetrics
    from qtpy.QtWidgets import QHeaderView

    _raw(tmp_path)
    _preprocessed(tmp_path, "nnUNetResEncUNetLPlans_custom_patchsize_192x192x192_spacing_iso_very_long")
    env = _Env(tmp_path, monkeypatch, size=(1200, 600))
    try:
        stage = env.open("preprocess")
        table = stage._plans_table
        header = table.horizontalHeader()
        assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch  # long plans names end with "…"
        assert table.item(0, 0).toolTip().startswith("nnUNetResEncUNetLPlans_custom")
        for column in (2, 3):
            item = table.item(0, column)
            assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.Fixed
            assert table.columnWidth(column) >= QFontMetrics(item.font()).horizontalAdvance(item.text()) + 16
        assert sum(table.columnWidth(column) for column in range(4)) <= table.viewport().width()
    finally:
        env.close()


def test_r18_header_badges_wrap(tmp_path: Path, monkeypatch) -> None:
    from nnworkbench.ui.widgets import FlowLayout, badge

    _raw(tmp_path)
    env = _Env(tmp_path, monkeypatch, size=(1200, 600))
    try:
        stage = env.open("train")
        assert isinstance(stage._badges, FlowLayout)
        monkeypatch.setattr(type(stage), "header_badges", lambda _self, _state: [
            badge(f"badge number {index} with a fairly long text", "gray") for index in range(4)])
        stage.set_state(stage.state)
        env.wait()
        badges = [stage._badges.itemAt(index).widget() for index in range(stage._badges.count())]
        assert len(badges) == 4 and len({item.y() for item in badges}) > 1  # wrapped into a second line
        assert all(item.width() >= item.sizeHint().width() for item in badges)  # none cut
    finally:
        env.close()


def test_r25_panel_titles_wrap_long_names() -> None:
    from nnworkbench.ui.stages.common import wrappable

    text = wrappable(LONG_TRAINER)
    assert text.replace("\u200b", "") == LONG_TRAINER
    assert max(len(part) for part in text.replace("_", "\u200b").split("\u200b")) <= 16


def test_r2_dice_bars_tooltip_and_elision() -> None:
    from nnworkbench.ui.stages.validate_stage import DiceBars

    application = _application()
    holder = QWidget()
    try:
        bars = DiceBars(holder)
        bars.resize(300, 10)
        bars.set_rows([("a" * 200, 0.5), ("b", 1.5, "1.50", "custom tip", True)])
        assert bars.height() == 2 * bars.ROW_HEIGHT
        assert bars.rows[0][3] == "a" * 200 + "\nDice 0.500" and bars.rows[1][1] == 1.0 and bars.rows[1][3] == "custom tip"
        bars.grab()  # paints without errors
        application.processEvents()
    finally:
        holder.close()
        holder.deleteLater()
        application.processEvents()


def test_r24_model_name_cell_elides_at_the_end() -> None:
    from nnworkbench.services.workflow_status import ModelInfo
    from nnworkbench.ui.stages.common import model_name_cell

    application = _application()
    holder = QWidget()
    try:
        cell = model_name_cell(ModelInfo(LONG_TRAINER, "nnUNetPlans", "3d_fullres", Path("x")))
        cell.setParent(holder)
        cell.resize(180, 48)
        application.processEvents()
        trainer = cell.findChildren(QWidget)[0]
        assert trainer._mode == Qt.TextElideMode.ElideRight
    finally:
        holder.close()
        holder.deleteLater()
        application.processEvents()
