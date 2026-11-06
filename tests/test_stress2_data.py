"""Regression tests of the second GUI stress test (scale and appearance) for the Data page: statistics with hundreds
of classes and thousands of images, undefined label IDs, the case panel, the description form, the dataset.json
editor, the Napari legend and the statistics console output (R1, R5, R6, R13, R17, R22, R23, R25-R28)."""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtWidgets import QApplication, QMessageBox, QScrollArea, QWidget

from nnworkbench import path_setup
from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.label_analysis import analysis_path


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _wait(application: QApplication, condition, seconds: float = 10.0) -> bool:
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        application.processEvents()
        time.sleep(0.01)
    return condition()


def _analysis(classes: int, images: int, undefined: tuple[int, ...] = ()) -> dict:
    """A cached statistics result like label_analysis writes it (without reading label files)."""
    names = {0: "background", **{index: f"class_{index:03d}" for index in range(1, classes)}}
    per_image = []
    for image in range(images):
        voxels = {"0": 1000, **{str(1 + (image + step) % (classes - 1)): 10 + step for step in range(20)}}
        if image == 0:
            voxels.update({str(value): 5 for value in undefined})
        per_image.append({"image": f"case_{image:05d}.nii.gz", "voxels": voxels, "shape": [16, 16, 16],
                          "spacing": [1.0, 1.0, 1.0],
                          "validity": [f"Undefined label IDs: {', '.join(map(str, undefined))}"]
                          if image == 0 and undefined else []})
    ids = sorted(set(names) | set(undefined))
    summary = [{"id": class_id, "name": names.get(class_id, f"Class {class_id}"),
                "images": sum(str(class_id) in item["voxels"] for item in per_image),
                "voxels": sum(item["voxels"].get(str(class_id), 0) for item in per_image)} for class_id in ids]
    return {"classes": summary, "per_image": per_image,
            "undefined_label_ids": {"case_00000.nii.gz": list(undefined)} if undefined else {},
            "unreadable_files": []}


# ---------------------------------------------------------------- R1, R22, R23, R6: the statistics view


def test_r1_statistics_view_with_500_classes_and_5000_images_opens_at_once() -> None:
    from nnworkbench.ui.label_statistics_dialog import LabelStatisticsView

    application = _application()
    analysis = _analysis(500, 5000)
    parent = QWidget()
    try:
        start = time.perf_counter()
        view = LabelStatisticsView(analysis, sparse_labels=True, parent=parent)
        parent.resize(900, 600)
        parent.show()
        application.processEvents()
        elapsed = time.perf_counter() - start
        assert elapsed < 10, f"{elapsed:.1f} s"  # 42 s with one table item per cell and resize-to-contents
        model = view.image_table.model()
        assert (model.rowCount(), model.columnCount()) == (5000, 5 + 500)
        assert model.index(1, 0).data() == "case_00001.nii.gz"
        assert model.index(0, 5).data() == "1,000"  # background voxels of the first image
        assert model.index(0, 4).data() == f"{100 * sum(analysis['per_image'][0]['voxels'].values()) / 4096:.2f}%"
        assert view.class_table.model().rowCount() == 500
    finally:
        parent.close()
    assert application is not None


def test_r6_statistics_count_only_defined_classes_and_flag_undefined_ids() -> None:
    from nnworkbench.ui.label_statistics_dialog import UNDEFINED, LabelStatisticsView

    application = _application()
    parent = QWidget()
    try:
        view = LabelStatisticsView(_analysis(3, 4, undefined=(7, 9)), parent=parent)
        summary = parent.findChild(QWidget, "analysisSummary")
        assert "3 classes" in summary.text()  # was "5 classes": the undefined IDs were counted
        assert "7, 9" in view.undefined_badge.text() and "7, 9" in view.undefined_badge.toolTip()
        model = view.class_table.model()
        names = {model.index(row, 0).data(): model.index(row, 1).data() for row in range(model.rowCount())}
        assert names == {"0": "background", "1": "class_001", "2": "class_002", "7": UNDEFINED, "9": UNDEFINED}
        assert model.index(3, 1).data(Qt.ItemDataRole.ForegroundRole) is not None  # flagged in orange
    finally:
        parent.close()
    assert application is not None


def test_r23_class_summary_shows_names_without_id_prefix_and_with_tooltips() -> None:
    from nnworkbench.ui.label_statistics_dialog import LabelStatisticsView

    application = _application()
    long_name = "Very_long_class_name_without_any_spaces_for_testing_the_layout_of_tables_and_XYZ"
    analysis = _analysis(3, 2)
    analysis["classes"][1]["name"] = long_name
    parent = QWidget()
    try:
        view = LabelStatisticsView(analysis, parent=parent)
        model = view.class_table.model()
        assert model.index(1, 1).data() == long_name  # was "1: Very_long…" and wrapped to "1: ..."
        assert model.index(1, 1).data(Qt.ItemDataRole.ToolTipRole) == long_name
        assert view.class_table.textElideMode() == Qt.TextElideMode.ElideRight and not view.class_table.wordWrap()
    finally:
        parent.close()
    assert application is not None


def test_r22_class_charts_have_one_readable_row_per_class() -> None:
    from nnworkbench.ui.label_statistics_dialog import LabelStatisticsView, _ClassChart

    application = _application()
    parent = QWidget()
    try:
        view = LabelStatisticsView(_analysis(500, 3), parent=parent)
        chart = view.voxel_chart
        assert chart.minimumHeight() >= 500 * _ClassChart.ROW  # was 210 px for all 500 bars
        assert chart.parentWidget() is not None and isinstance(chart.parentWidget().parentWidget(), QScrollArea)
        assert chart._row_at(_ClassChart.TOP + 2 * _ClassChart.ROW + 1) == 2
        chart.set_ignore_background(True)
        assert chart.minimumHeight() == _ClassChart.TOP + 499 * _ClassChart.ROW + 10
        assert chart._classes[0]["id"] == 1
        parent.resize(700, 500)
        parent.show()
        view.findChild(QScrollArea).widget().grab()  # paints only the rows in view without errors
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- the Data page (R1 lazy, R6, R13, R23, R25)


@pytest.fixture
def window(tmp_path: Path, monkeypatch):
    from nnworkbench.ui.main_window import MainWindow

    monkeypatch.setattr(path_setup, "configuration_path", lambda: tmp_path / "config" / "paths.json")
    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok))
    application = _application()
    main = MainWindow()
    yield main
    main.context.tasks.shutdown()
    main.close()
    application.processEvents()


def _dataset(tmp_path: Path, labels: dict, analysis: dict, cases: int = 1) -> DatasetRecord:
    raw = tmp_path / "Dataset321_Undefined"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for index in range(cases):
        (raw / "imagesTr" / f"case_{index:05d}_0000.nii.gz").write_bytes(b"")
        (raw / "labelsTr" / f"case_{index:05d}.nii.gz").write_bytes(b"")
    (raw / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": labels,
                                                  "numTraining": cases, "file_ending": ".nii.gz"}), encoding="utf-8")
    analysis_path(raw).parent.mkdir(exist_ok=True)
    analysis_path(raw).write_text(json.dumps(analysis), encoding="utf-8")
    return DatasetRecord(321, "Undefined", raw_path=raw)


def test_r1_data_page_builds_the_statistics_only_when_they_are_shown(tmp_path: Path, window, monkeypatch) -> None:
    from nnworkbench.ui.dataset_state import load_state
    from nnworkbench.ui.stages import data_stage

    built = []
    original = data_stage.LabelStatisticsView
    monkeypatch.setattr(data_stage, "LabelStatisticsView", lambda *args, **kwargs: (
        built.append(True), original(*args, **kwargs))[1])
    state = load_state(_dataset(tmp_path, {"background": 0, "a": 1, "b": 2}, _analysis(3, 1)))
    data = window.dataset_page.pages["data"]
    data.set_state(state)
    assert state.analysis is not None and built == []  # opening and refreshing the dataset does not build it
    data._views.setCurrentIndex(1)
    assert built == [True]
    data._views.setCurrentIndex(0)
    data._views.setCurrentIndex(1)
    assert built == [True]  # built once per state
    data.set_state(state)  # a refresh while Statistics is shown rebuilds at once
    assert built == [True, True]


def test_r6_r13_r23_case_panel_counts_defined_classes_scrolls_and_elides(tmp_path: Path, window) -> None:
    from qtpy.QtWidgets import QLabel

    from nnworkbench.ui.dataset_state import load_state
    from nnworkbench.ui.widgets import ElidedLabel

    long_name = "A rather long class name with spaces that should wrap or elide nicely in the user interface"
    analysis = _analysis(3, 1, undefined=(7, 9))
    analysis["per_image"][0]["voxels"] = {"0": 60, "1": 1, "2": 1, "7": 1, "9": 1}
    state = load_state(_dataset(tmp_path, {"background": 0, long_name: 1, "b": 2}, analysis))
    window.resize(1366, 768)
    window.show()
    window.open_dataset(state.record)
    data = window.dataset_page.pages["data"]
    assert _wait(QApplication.instance(), lambda: window.dataset_page.state is not None)
    window.dataset_page.show_stage("data")
    data.show_case(0)
    QApplication.processEvents()
    panel = data._case_panel
    texts = [widget.text() for widget in panel.findChildren(QLabel)]
    assert "3 of 3" in texts  # was "5 of 3": the undefined IDs 7 and 9 were counted as classes
    assert any("Undefined label IDs: 7" in text and "9" in text for text in texts)
    name = next(widget for widget in panel.findChildren(ElidedLabel) if long_name in widget.toolTip())
    assert name.text() != long_name and name.text().endswith("…")  # was cut at 128 px without "…"
    # files, geometry and classes scroll; the head and the actions stay outside the scroll area
    area = panel.findChild(QScrollArea)
    assert area is not None and area.widget().isAncestorOf(name)
    assert not area.widget().isAncestorOf(panel._inspect) and not area.widget().isAncestorOf(panel._case)


def test_r25_files_card_path_elides_so_the_toggle_keeps_its_text(tmp_path: Path, window) -> None:
    from nnworkbench.ui.dataset_state import load_state
    from nnworkbench.ui.widgets import ElidedLabel

    record = _dataset(tmp_path, {"background": 0, "a": 1}, _analysis(2, 1))
    long = tmp_path / ("Dataset322_" + "Very_long_dataset_name_" * 6)
    record.raw_path.rename(long)
    state = load_state(DatasetRecord(322, long.name[11:], raw_path=long))
    data = window.dataset_page.pages["data"]
    data.set_state(state)
    assert isinstance(data._path, ElidedLabel) and data._path.minimumSizeHint().width() <= 60
    assert long.name in data._path.toolTip()


# ---------------------------------------------------------------- R5, R17: the description form


def _description_panel(tmp_path: Path, classes: int, name: str = "class"):
    from nnworkbench.ui.description_panel import DescriptionPanel

    folder = tmp_path / "Dataset323_Form"
    (folder / "imagesTr").mkdir(parents=True)
    labels = {"background": 0, **{f"{name}_{index:03d}": index for index in range(1, classes)}}
    (folder / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": labels,
                                                     "file_ending": ".nii.gz"}), encoding="utf-8")
    parent = QWidget()
    panel = DescriptionPanel(parent)
    panel.set_dataset(folder)
    return parent, panel


def test_r5_description_form_with_500_classes_opens_folded_and_typing_stays_fast(tmp_path: Path) -> None:
    application = _application()
    parent, panel = _description_panel(tmp_path, 500)
    try:
        panel._create()
        assert len(panel._class_sections) == 500
        # folded sections create their text boxes only when opened (6 s to open the form before)
        assert all(section._editors is None for section in panel._class_sections.values())
        token, _row, _identifier, title = panel._class_table._rows[5]
        section = panel._class_sections[token]
        order = list(panel._class_order)
        start = time.perf_counter()
        for character in "abcdefghij":
            title.setText(title.text() + character)
            application.processEvents()
        per_key = (time.perf_counter() - start) / 10
        assert per_key < 0.1, f"{per_key:.3f} s per keystroke"  # 0.4 s before
        assert _wait(application, lambda: section._toggle.toolTip().startswith("Class 5: class_005abcdefghij"))
        assert panel._class_order == order  # a renamed class does not re-order the sections
        section.set_collapsed(False)
        assert section._editors is not None and set(section.answers()) == set(section.editors)
        panel._class_table._rows[5][2].setText("999")  # a changed ID moves its section to the end
        assert _wait(application, lambda: panel._class_order[-1] == token)
    finally:
        parent.close()
    assert application is not None


def test_r17_long_class_titles_elide_and_the_form_never_scrolls_sideways(tmp_path: Path) -> None:
    application = _application()
    name = "Very_long_class_name_without_any_spaces_for_testing_the_layout_of_tables_and_XYZ"
    parent, panel = _description_panel(tmp_path, 3, name=name)
    try:
        panel._create()
        panel._set_all_collapsed(False)
        parent.resize(420, 600)
        parent.show()
        application.processEvents()
        scroll = panel._form_scroll
        assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width()  # was ~800 px
        button = next(section._toggle for section in panel._class_sections.values() if name in section._toggle.toolTip())
        assert button.text().endswith("…") and name in button.toolTip()
    finally:
        parent.close()
    assert application is not None


# ---------------------------------------------------------------- R26, R27, R28


def test_r26_napari_legend_is_a_scrollable_table_of_limited_height() -> None:
    from nnworkbench.services.napari_inspector import LEGEND_ROWS, annotation_legend

    application = _application()
    parent = QWidget()
    try:
        labels = tuple((index, f"class_{index:03d}") for index in range(500))
        table = annotation_legend(labels, parent)
        assert table.rowCount() == 500 and table.item(42, 1).text() == "class_042"
        assert table.item(42, 1).toolTip() == "42: class_042"
        row = table.verticalHeader().defaultSectionSize()
        assert table.height() <= (LEGEND_ROWS + 2) * row + 40  # was one 4720 px label
        small = annotation_legend(((0, "background"), (1, "cell")), parent)
        assert small.height() < table.height()
    finally:
        parent.close()
    assert application is not None


def test_r27_statistics_silence_the_native_reader_output(monkeypatch, capfd) -> None:
    np = pytest.importorskip("numpy")
    from nnworkbench.services import label_analysis

    class _Image:  # MITK logs about ten lines per file directly to the native stdout
        def __init__(self, filename: str) -> None:
            os.write(1, f"loading {filename} via itk::ImageIOFactory...\n".encode())
            os.write(2, b"[0.588] ioRegion: ImageIORegion\n")
            self.shape, self.spacing = (2, 2, 2), (1.0, 1.0, 1.0)

        def __array__(self, dtype=None, copy=None):
            return np.zeros((2, 2, 2), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "mitk", types.SimpleNamespace(Image=_Image))
    result = label_analysis._analyse_file("label.nii.gz")
    os.write(1, b"progress line\n")
    output = capfd.readouterr()
    assert result["voxels"] == {0: 8}
    assert "ImageIOFactory" not in output.out and "ioRegion" not in output.err
    assert "progress line" in output.out  # our own output is kept


def test_r28_json_editor_fields_show_the_start_of_long_names(tmp_path: Path) -> None:
    from qtpy.QtWidgets import QLineEdit

    from nnworkbench.ui.json_editor import DatasetJsonPanel

    application = _application()
    name = "Very_long_class_name_without_any_spaces_for_testing_the_layout_of_tables_and_XYZ"
    manifest = {"channel_names": {"0": "CT"}, "labels": {"background": 0, name: 1}, "numTraining": 1,
                "file_ending": ".nii.gz", "reference": "a long custom value " * 5}
    (tmp_path / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    parent = QWidget()
    try:
        panel = DatasetJsonPanel(parent)
        panel.set_manifest(tmp_path, manifest, editable=True)
        panel._start_edit()
        fields = [field for field in panel.findChildren(QLineEdit) if len(field.text()) > 30]
        assert {field.text() for field in fields} >= {name, manifest["reference"]}
        assert all(field.cursorPosition() == 0 for field in fields)  # the end of the name was shown
        panel._cancel_edit()
    finally:
        parent.close()
    assert application is not None


def test_only_the_exact_ignore_key_is_the_ignore_label(tmp_path: Path) -> None:
    """As nnU-Net: "Ignore" is an ordinary class (no ID shift, dense description); only "ignore" is the ignore label."""
    from nnworkbench.services.dataset_description import description_template
    from nnworkbench.services.napari_inspector import annotation_label_summary

    for key, sparse in (("Ignore", False), ("ignore", True)):
        # "Ignore" and "ignore" are the same folder on Windows and macOS (case-insensitive file systems)
        folder = tmp_path / ("sparse" if sparse else "dense") / "Dataset001_Cells"
        (folder / "imagesTr").mkdir(parents=True)
        (folder / "imagesTr" / "cells_000_0000.nii.gz").touch()
        (folder / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1, key: 2},
            "numTraining": 1, "file_ending": ".nii.gz"}), encoding="utf-8")
        summary = {name: value for value, name in annotation_label_summary(folder / "imagesTr" / "cells_000_0000.nii.gz")}
        assert summary == ({"ignore": 0, "background": 1, "cell": 2} if sparse else {"background": 0, "cell": 1, "Ignore": 2})
        assert ("**Dense or sparse:** sparse" in description_template(folder)) is sparse
