"""Regression tests of the GUI stress test (Data stage, dataset.json editor, description, statistics, guide)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from nnworkbench.services.dataset_description import (
    dataset_channels,
    description_classes,
    description_path,
)
from nnworkbench.services.dataset_json import definition_errors
from nnworkbench.services.label_analysis import analysis_path, load_label_analysis


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


@pytest.fixture
def messages(monkeypatch) -> list:
    """Message boxes answer at once (question: Yes) and are recorded as (title, text)."""
    shown: list = []

    def answer(result):
        return staticmethod(lambda *arguments, **_keywords: (shown.append(arguments[1:3]), result)[1])

    monkeypatch.setattr(QMessageBox, "warning", answer(QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "information", answer(QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Yes))
    return shown


def _dataset(tmp_path: Path, labels: dict | None = None, channels: dict | None = None) -> Path:
    folder = tmp_path / "Dataset042_Cells"
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    manifest = {"channel_names": channels or {"0": "DAPI"}, "labels": labels or {"background": 0, "cell": 1},
                "numTraining": 0, "file_ending": ".nii.gz"}
    (folder / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _panel(folder: Path):
    from nnworkbench.ui.description_panel import DescriptionPanel

    parent = QWidget()
    panel = DescriptionPanel(parent)
    panel.set_dataset(folder)
    return parent, panel


def _section(panel, name: str):
    return next(panel._class_sections[token] for token, _value, row_name in panel._class_table.rows()
                if row_name == name)


# ---------------------------------------------------------------- D1, D2: description form keeps classes and texts


def test_d1_class_added_in_the_form_survives_edit_and_save(tmp_path: Path, messages: list) -> None:
    application = _application()
    folder = _dataset(tmp_path)
    parent, panel = _panel(folder)
    try:
        panel._create()
        panel._class_table._add_row(2, "nucleus")
        _section(panel, "nucleus").editors["description"].text.setPlainText("NUCLEUS TEXT")
        panel._channel_table._rows[0][3].setText("Hoechst")
        assert panel._save_form()
        panel._start_form()  # Edit again: the tables come from the file, not from dataset.json
        assert panel._class_table.classes() == [(0, "background"), (1, "cell"), (2, "nucleus")]
        assert panel._channel_table.classes() == [(0, "Hoechst")]
        assert panel._save_form()
        text = description_path(folder).read_text(encoding="utf-8")
        assert description_classes(text) == [(0, "background"), (1, "cell"), (2, "nucleus")]
        assert "NUCLEUS TEXT" in text
    finally:
        parent.close()
    assert application is not None


@pytest.mark.parametrize(("change", "message"), [
    ({1: "cell-A", 2: "cell A"}, "kept apart"),
    ({1: "细胞", 2: "细胞核"}, "kept apart"),
    ({2: "Zelle | Membran"}, '"|"'),
    ({"id": ""}, "whole number"),
])
def test_d2_form_refuses_tables_that_would_lose_class_texts(tmp_path: Path, messages: list, change: dict,
                                                             message: str) -> None:
    application = _application()
    folder = _dataset(tmp_path, labels={"background": 0, "Zellkern": 1, "Zelle": 2})
    parent, panel = _panel(folder)
    try:
        panel._create()
        saved = description_path(folder).read_text(encoding="utf-8")
        _section(panel, "Zelle").editors["description"].text.setPlainText("TEXT FOR ZELLE")
        for row, name in change.items():
            if row == "id":
                panel._class_table._rows[2][2].setText(name)  # the ID cleared to retype it
            else:
                panel._class_table._rows[row][3].setText(name)
        assert not panel._save_form()
        assert panel.editing and any(message in text for _title, text in messages)
        assert description_path(folder).read_text(encoding="utf-8") == saved
    finally:
        parent.close()
    assert application is not None


def test_d2_table_problems_accept_ordinary_tables() -> None:
    from nnworkbench.services.dataset_description import table_problems

    assert table_problems([(0, "background"), (1, "Zellkern"), (2, "Zelle"), (3, "ignore")]) == []
    assert table_problems([(0, "细胞")]) == []  # one class without a-z keeps its key
    assert table_problems([(0, "a|b")], "channel") == [
        'Channel names cannot contain "|" (it separates the columns of the table).']


# ---------------------------------------------------------------- D3, D4: odd dataset.json and description files


def test_d3_channel_keys_that_are_digits_but_not_integers() -> None:
    manifest = {"channel_names": {"²": "CT"}, "labels": {"background": 0}}
    assert definition_errors(manifest)  # an error, not a ValueError
    assert definition_errors({"channel_names": {"٠": "CT"}, "labels": {"background": 0}})  # nnU-Net expects "0"


def test_d3_description_ignores_odd_channel_keys(tmp_path: Path) -> None:
    folder = _dataset(tmp_path, channels={"0": "CT", "²": "MR"})
    assert dataset_channels(folder) == [(0, "CT")]


def test_d4_description_in_latin1_opens(tmp_path: Path, messages: list) -> None:
    from nnworkbench.services.dataset_description import description_template

    application = _application()
    folder = _dataset(tmp_path)
    path = description_path(folder)
    path.parent.mkdir()
    path.write_bytes(description_template(folder).replace("**Type:**", "**Type:** Zellkern-Färbung").encode("cp1252", errors="replace"))
    parent, panel = _panel(folder)  # set_dataset reads the file
    try:
        panel._start_form()
        assert panel.editing and panel._editors["type"].answer().text.startswith("Zellkern-Färbung")
        assert panel._save_form()
        assert "Zellkern-Färbung" in path.read_text(encoding="utf-8")  # saved as UTF-8
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D5: files that cannot be written or read


def _read_only(path: Path) -> None:
    path.chmod(0o444)
    path.parent.chmod(0o555)
    if os.access(path, os.W_OK):
        path.parent.chmod(0o755)
        pytest.skip("files stay writable (e.g. as root or on Windows)")


def test_d5_form_save_and_import_on_a_read_only_description(tmp_path: Path, messages: list) -> None:
    application = _application()
    folder = _dataset(tmp_path)
    parent, panel = _panel(folder)
    path = description_path(folder)
    try:
        panel._create()
        assert panel._save_form()
        _read_only(path)
        panel._import_classes()  # viewer: writes the file
        panel._start_form()
        panel._editors["type"].text.setPlainText("typed")
        assert not panel._save_form() and panel.editing
        assert any(title == "Not saved" for title, _text in messages)
    finally:
        path.parent.chmod(0o755)
        path.chmod(0o644)
        parent.close()
    assert application is not None


def test_d5_adding_a_missing_image_shows_a_message(tmp_path: Path, messages: list, monkeypatch) -> None:
    application = _application()
    folder = _dataset(tmp_path)
    parent, panel = _panel(folder)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *_arguments, **_keywords: (str(tmp_path / "gone.png"), "")))
    try:
        panel._create()
        editor = panel._editors["representative"]
        editor._add_image()
        assert editor.answer().images == () and messages[-1][0] == "Image not added"
    finally:
        parent.close()
    assert application is not None


def test_d5_text_dialog_with_a_deleted_folder(tmp_path: Path, messages: list) -> None:
    from nnworkbench.ui.description_dialog import DescriptionDialog

    application = _application()
    folder = tmp_path / "nnworkbench"
    folder.mkdir()
    path = folder / "dataset-description.md"
    path.write_text("# Cells\n", encoding="utf-8")
    parent = QWidget()
    dialog = DescriptionDialog(path, parent)
    try:
        dialog._start_edit()
        path.unlink()
        folder.rmdir()
        dialog._editor.setPlainText("# Cells, edited\n")
        dialog._save_edit()  # the folder is gone: a message, the text stays
        assert messages[-1][0] == "Not saved" and dialog._editor.toPlainText() == "# Cells, edited\n"
        assert dialog._has_changes()
        dialog._start_edit()  # unreadable: a message
        assert messages[-1][0] == "Description not read"
    finally:
        dialog.close()
        parent.close()
    assert application is not None


def test_d5_text_dialog_adds_images_next_to_the_description(tmp_path: Path, messages: list, monkeypatch) -> None:
    from nnworkbench.ui.description_dialog import DescriptionDialog

    application = _application()
    folder = tmp_path / "nnworkbench"
    folder.mkdir()
    path = folder / "dataset-description.md"
    path.write_text("# Cells\n", encoding="utf-8")
    picture = tmp_path / "cell.png"
    picture.write_bytes(b"png")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *_arguments, **_keywords: (str(picture), "")))
    parent = QWidget()
    dialog = DescriptionDialog(path, parent)
    try:
        dialog._start_edit()
        dialog._insert_image()
        assert (folder / "description_images" / "cell.png").is_file()  # the link in the text resolves
        assert "](description_images/cell.png)" in dialog._editor.toPlainText()
    finally:
        dialog.close()
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D6, D11: dataset.json editor


def _json_panel(folder: Path, manifest: dict):
    from nnworkbench.ui.json_editor import DatasetJsonPanel

    parent = QWidget()
    panel = DatasetJsonPanel(parent)
    panel.set_manifest(folder, manifest, editable=True)
    panel._start_edit()
    return parent, panel


@pytest.mark.parametrize("edit", ["channel superscript", "dashes then ignore", "superscript then ignore"])
def test_d6_json_editor_odd_ids(tmp_path: Path, edit: str) -> None:
    application = _application()
    folder = _dataset(tmp_path)
    before = (folder / "dataset.json").read_text(encoding="utf-8")
    parent, panel = _json_panel(folder, json.loads(before))
    try:
        labels, channels = panel._editors["labels"], panel._editors["channel_names"]
        if edit == "channel superscript":
            channels._rows[0][1].setText("²")
        else:
            labels._rows[1][2].setText("--1" if edit == "dashes then ignore" else "²")
            labels._add_ignore_row()
        panel._save_edit()
        assert "Not saved" in panel._note.text()
        assert (folder / "dataset.json").read_text(encoding="utf-8") == before
    finally:
        parent.close()
    assert application is not None


def test_d6_json_editor_with_labels_as_a_list(tmp_path: Path) -> None:
    application = _application()
    folder = _dataset(tmp_path)
    parent, panel = _json_panel(folder, {"channel_names": {"0": "CT"}, "labels": ["background", "cell"],
                                         "numTraining": 0, "file_ending": ".nii.gz"})
    try:
        panel._save_edit()
        assert "Not saved" in panel._note.text()
    finally:
        parent.close()
    assert application is not None


@pytest.mark.parametrize(("key", "value"), [("numTraining", "1e999"), ("numTraining", "NaN"), ("file_ending", ""),
                                            ("channel", "٠"), ("duplicate", "cell")])
def test_d11_json_editor_refuses_invalid_values(tmp_path: Path, key: str, value: str) -> None:
    application = _application()
    folder = _dataset(tmp_path, labels={"background": 0, "cell": 1, "nucleus": 2})
    before = (folder / "dataset.json").read_text(encoding="utf-8")
    parent, panel = _json_panel(folder, json.loads(before))
    try:
        if key == "channel":
            panel._editors["channel_names"]._rows[0][1].setText(value)
        elif key == "duplicate":  # nucleus: 2 becomes a second cell: 1, merged silently into one label before
            panel._editors["labels"]._rows[2][1].setText(value)
            panel._editors["labels"]._rows[2][2].setText("1")
        else:
            panel._editors[key].setText(value)
        panel._save_edit()
        assert "Not saved" in panel._note.text()
        assert (folder / "dataset.json").read_text(encoding="utf-8") == before
    finally:
        parent.close()
    assert application is not None


def test_json_panel_shows_a_deeply_nested_manifest(tmp_path: Path) -> None:
    from nnworkbench.ui.json_editor import DatasetJsonPanel

    application = _application()
    nested: list = []
    for _level in range(5000):
        nested = [nested]
    parent = QWidget()
    panel = DatasetJsonPanel(parent)
    try:
        panel.set_manifest(tmp_path, {"labels": {"a": 1}, "x": nested}, editable=True)  # no RecursionError
        assert "cannot be displayed" in panel._view.toPlainText()
        assert panel._edit.isHidden()
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D7: description table IDs


def test_d7_table_editor_with_odd_ids() -> None:
    from nnworkbench.ui.description_panel import _TableEditor

    application = _application()
    parent = QWidget()
    table = _TableEditor([(0, "background"), (1, "cell")])
    table.setParent(parent)
    try:
        table._rows[1][2].setText("²")
        assert table.classes() == [(0, "background"), (-1, "cell")]  # -1: not a valid ID
        table._rows[1][2].setText("--1")
        assert table.classes() == [(0, "background"), (-1, "cell")]
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D8: case panel with many label values


def test_d8_case_panel_caps_the_class_list() -> None:
    from nnworkbench.ui.stages.data_stage import MAX_CASE_CLASSES, CaseStatisticsPanel

    application = _application()
    stage = SimpleNamespace(step_case=lambda _offset: None, close_case=lambda: None, inspect=lambda _row: None,
                            annotate=lambda _row: None)
    parent = QWidget()
    panel = CaseStatisticsPanel(stage)
    panel.setParent(parent)
    try:
        values = 2000
        names = {value: f"class{value}" for value in range(values + 500)}
        state = SimpleNamespace(names=names, analysis={"per_image": [
            {"image": "a.nii.gz", "voxels": {str(value): value + 1 for value in range(values)}}]})
        started = time.perf_counter()
        panel._add_classes(SimpleNamespace(label_file=Path("a.nii.gz")), state)
        assert time.perf_counter() - started < 2
        assert panel._content.count() <= MAX_CASE_CLASSES + 4  # head, rows, "and N more", missing
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D9: large Markdown tables


def test_d9_large_markdown_table_renders_quickly(tmp_path: Path) -> None:
    from nnworkbench.ui.workflow_guide import GuideBrowser

    application = _application()
    path = tmp_path / "table.md"
    path.write_text("# T\n\n| ID | Class |\n| --- | --- |\n" + "\n".join(f"| {i} | `class{i}` |" for i in range(400))
                    + "\n", encoding="utf-8")
    parent = QWidget()
    browser = GuideBrowser(parent)
    parent.resize(900, 700)
    browser.resize(900, 700)
    parent.show()
    try:
        started = time.perf_counter()
        browser.load(path)
        assert time.perf_counter() - started < 1.5  # 18 s before
        assert "class399" in browser.toPlainText()
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- D10: statistics


@pytest.mark.parametrize("change", [
    lambda analysis: analysis["per_image"][0].__setitem__("shape", 5),
    lambda analysis: analysis["classes"][0].__setitem__("voxels", "many"),
    lambda analysis: analysis["per_image"][0].__setitem__("voxels", None),
    lambda analysis: analysis["per_image"][0].__setitem__("voxels", {"x": 1}),
    lambda analysis: analysis["per_image"][0].__setitem__("spacing", "1,1,1"),
])
def test_d10_malformed_statistics_cache_is_ignored(tmp_path: Path, change) -> None:
    analysis = {"classes": [{"id": 0, "name": "background", "images": 1, "voxels": 27}], "per_image": [
        {"image": "a.nii.gz", "voxels": {"0": 27}, "shape": [3, 3, 3], "spacing": [1, 1, 1], "validity": []}]}
    analysis_path(tmp_path).parent.mkdir()
    analysis_path(tmp_path).write_text(json.dumps(analysis), encoding="utf-8")
    assert load_label_analysis(tmp_path) == analysis
    change(analysis)
    analysis_path(tmp_path).write_text(json.dumps(analysis), encoding="utf-8")
    assert load_label_analysis(tmp_path) is None


def test_d10_unreadable_and_float_label_files(tmp_path: Path) -> None:
    from nnworkbench.services.label_analysis import (
        NOT_WHOLE_NUMBERS,
        UNREADABLE,
        analyse_label_distribution,
    )

    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    labels = tmp_path / "labelsTr"
    labels.mkdir()
    (tmp_path / "dataset.json").write_text('{"labels": {"background": 0, "cell": 1}}', encoding="utf-8")
    mitk.Image(np.array([[[0.0, 0.3], [0.7, 1.0]]], dtype=np.float32)).save(labels / "a.nii.gz")
    (labels / "b.nii.gz").write_bytes(b"\x1f\x8b not a label")
    progress: list[str] = []
    analysis = analyse_label_distribution(tmp_path, workers=1, progress=progress.append)
    first, second = analysis["per_image"]
    assert first["voxels"] == {"0": 3, "1": 1}  # 0.0, 0.3 and 0.7 all count as 0
    assert NOT_WHOLE_NUMBERS in first["validity"]
    assert second["voxels"] == {} and second["validity"][0].startswith(UNREADABLE)
    assert analysis["unreadable_files"] == ["b.nii.gz"]
    assert progress[-1].startswith("Could not read label 2/2: b.nii.gz")
    assert load_label_analysis(tmp_path) == analysis


def test_d10_statistics_view_with_an_unreadable_file() -> None:
    from nnworkbench.ui.label_statistics_dialog import LabelStatisticsView

    application = _application()
    parent = QWidget()
    analysis = {"classes": [{"id": 0, "name": "background", "images": 1, "voxels": 8}], "per_image": [
        {"image": "a.nii.gz", "voxels": {"0": 8}, "shape": [2, 2, 2], "spacing": [1.0, 1.0, 1.0], "validity": []},
        {"image": "b.nii.gz", "voxels": {}, "shape": [], "spacing": [], "validity": ["Unreadable label file: x"]}]}
    view = LabelStatisticsView(analysis, annotation_workflow_enabled=True, parent=parent)
    try:
        assert view.image_table.item(1, 4).text() == "—"  # no labeled share for an unreadable file
    finally:
        parent.close()
    assert application is not None
