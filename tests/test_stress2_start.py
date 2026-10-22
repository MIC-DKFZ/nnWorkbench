"""Start screen at scale and on small screens (GUI stress test round 2: R3, R4, R10-R12, R19-R21, R25, R28)."""

from __future__ import annotations

import os
import time
from itertools import pairwise
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QApplication, QDialog, QLabel, QWidget

from nnworkbench import path_setup
from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.ui import overview_page
from nnworkbench.ui.main_window import MainWindow
from nnworkbench.ui.new_dataset_dialog import NewDatasetDialog
from nnworkbench.ui.widgets import FlowLayout

# CI machines are slower than a desktop: the bounds are several times what the fixed code needs (< 0.05 s here),
# but below what the old code took (1.2-1.6 s per keystroke or sort, 1 s to open the switcher).
_FAST = 0.5
_LONG = "VeryLongDatasetName_" + "abcdefghij" * 18


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> Path:
    """An isolated paths.json (not written: no nnU-Net folders)."""
    path = tmp_path / "config" / "paths.json"
    monkeypatch.setattr(path_setup, "configuration_path", lambda: path)
    return path


def _window(application: QApplication, size: tuple[int, int] = (1200, 700)) -> MainWindow:
    window = MainWindow()
    window.resize(*size)
    window.show()
    application.processEvents()
    return window


def _settle(application: QApplication) -> None:
    """Let the posted layout requests run (scroll ranges and geometries are updated after them)."""
    for _ in range(5):
        application.processEvents()
        time.sleep(0.01)


def _close(window: MainWindow, application: QApplication) -> None:
    window.switcher.hide()
    window.close()
    application.processEvents()


def _details(labels: int, **manifest) -> dict:
    manifest = {"channel_names": {str(index): f"channel_{index}" for index in range(50)},
                "labels": {"background": 0, **{f"class{index}": index for index in range(1, labels)}},
                "numTraining": 100, "file_ending": ".nii.gz", **manifest}
    return {"manifest": manifest, "raw_folders": ["imagesTr", "labelsTr"], "plans": ["nnUNetPlans_" + "x" * 120],
            "models": 1, "folds": 5}


def _many(count: int = 1000) -> list[DatasetRecord]:
    return [DatasetRecord(index, f"Dataset_{index}" if index % 7 else f"Very_long_name_{index}_" + "x" * 120)
            for index in range(1, count + 1)]


def test_window_minimum_fits_a_laptop_screen(config: Path) -> None:
    application = _application()
    window = _window(application)
    try:
        minimum = window.minimumSize()
        assert minimum.width() <= 1200 and minimum.height() <= 700  # 1280x760 did not fit 1366x768 (~728 usable)
        room = 700 - window.app_bar.height() - window.console.minimumSizeHint().height()
        hint = window.overview.minimumSizeHint()
        assert hint.width() <= 1200 and hint.height() <= room
    finally:
        _close(window, application)


def test_details_panel_scrolls_and_shows_every_chip(config: Path, tmp_path: Path) -> None:
    application = _application()
    window = _window(application)
    details = window.overview.details
    try:
        record = DatasetRecord(1, _LONG, raw_path=tmp_path)
        details.set_record(record, _details(500, description="Long " * 200))
        _settle(application)
        content = details._scroll.widget()
        assert details._scroll.verticalScrollBar().maximum() > 0  # scrolls instead of squashing the rows
        chips_holder = details._info.itemAt(2).widget()
        chips = chips_holder.findChildren(QLabel)
        assert chips[-1].text() == "+400 more" and chips[-1].isVisibleTo(details)
        # every chip that is promised is laid out inside the holder (not cut off below it) and within its width
        for chip in chips:
            assert chip.geometry().bottom() <= chips_holder.height()
            assert chip.geometry().right() <= chips_holder.width()
        assert chips_holder.height() >= chips_holder.layout().heightForWidth(chips_holder.width())
        assert content.height() >= content.sizeHint().height() - 1
        # the grid rows (Modality, Cases, Labels, File ending) do not overlap
        grid = details._info.itemAt(1).layout()
        tops = [grid.itemAtPosition(row, 0).widget().geometry() for row in range(4)]
        assert all(upper.bottom() < lower.top() for upper, lower in pairwise(tops))
        modality = grid.itemAtPosition(0, 1).widget()
        assert modality.width() <= grid.geometry().width()
        assert "50 channels" in modality.text() and modality.toolTip().startswith("channel_0")
    finally:
        _close(window, application)


def test_details_header_elides_long_names(config: Path, tmp_path: Path) -> None:
    application = _application()
    window = _window(application)
    details = window.overview.details
    try:
        details.set_record(DatasetRecord(4, _LONG, raw_path=tmp_path), _details(3))
        _settle(application)
        for widget, full in ((details._name, _LONG), (details._folder, f"Dataset004_{_LONG}")):
            assert "…" in widget.text() and widget.toolTip() == full
            assert widget.geometry().right() < details._open.geometry().left()
        assert details._name.textFormat() == Qt.TextFormat.PlainText
    finally:
        _close(window, application)


def test_flow_layout_cuts_wide_items_to_the_line() -> None:
    application = _application()
    holder = QWidget()
    try:
        layout = FlowLayout(holder)
        for text in ("short", "x" * 300):
            layout.addWidget(QLabel(text))
        holder.resize(200, layout.heightForWidth(200))
        layout.setGeometry(holder.rect())
        assert all(layout.itemAt(index).geometry().right() < 200 for index in range(layout.count()))
    finally:
        holder.close()
        application.processEvents()


def test_id_badge_row_stays_in_its_column(config: Path) -> None:
    application = _application()
    window = _window(application, (1366, 768))
    table = window.overview._table
    try:
        records = [DatasetRecord(7, "Dup_" + _LONG, id_problem=("duplicate ID", "ID 7 is used by 2 folders")),
                   DatasetRecord(8, "Other")]
        window.overview.set_records(records)
        _settle(application)
        cell = table.cellWidget(0, 1)
        column = table.columnWidth(1)
        assert cell.width() <= column  # was as wide as the full name: drawn over AVAILABLE and MODIFIED
        badge = next(child for child in cell.findChildren(QLabel) if child.text() == "duplicate ID")
        assert badge.isVisibleTo(cell) and badge.geometry().right() <= column
        name = next(child for child in cell.findChildren(QLabel) if child.toolTip() == "Dup_" + _LONG)
        assert name.text().endswith("…")
    finally:
        _close(window, application)


def test_search_and_sort_with_1000_datasets_are_fast(config: Path) -> None:
    application = _application()
    window = _window(application, (1366, 768))
    overview = window.overview
    table = overview._table
    try:
        overview.set_records(_many())
        application.processEvents()
        assert table.rowCount() == 1000 and table.cellWidget(0, 2) is None  # pills painted, no widget per row
        for text, shown in (("D", 1000), ("Dataset_", 858), ("long", 142), ("999", 1), ("", 1000)):
            start = time.perf_counter()
            overview._search.setText(text)
            application.processEvents()
            assert time.perf_counter() - start < _FAST, text
            assert sum(not table.isRowHidden(row) for row in range(table.rowCount())) == shown
        overview._search.setText("999")
        assert overview._selected_record.dataset_id == 999  # the only match is selected
        overview._search.setText("")
        for index in (1, 2, 0):
            start = time.perf_counter()
            overview._sort.setCurrentIndex(index)
            application.processEvents()
            assert time.perf_counter() - start < _FAST, index
        assert overview._selected_record.dataset_id == 999  # kept while sorting
        overview._sort.setCurrentIndex(1)
        names = [table.item(row, 0).data(Qt.ItemDataRole.UserRole).name.casefold() for row in range(5)]
        assert names == sorted(names)
        overview._search.setText("no such dataset")
        assert overview._table_stack.currentWidget() is overview._empty
    finally:
        _close(window, application)


def test_switcher_opens_fast_with_1000_datasets(config: Path) -> None:
    application = _application()
    window = _window(application, (1366, 768))
    switcher = window.switcher
    try:
        records = _many()
        window.records = records
        start = time.perf_counter()
        switcher.open(window.app_bar.switcher_anchor(), records, records[500])
        application.processEvents()
        assert time.perf_counter() - start < _FAST
        item = switcher._list.item(6)
        assert switcher._list.itemWidget(item) is None  # rows are painted
        assert item.toolTip().startswith(records[6].name)  # long names are shortened, full name on hover
        assert switcher._list.currentRow() == 500
        switcher._search.setText("long")
        assert sum(not switcher._list.item(row).isHidden() for row in range(switcher._list.count())) == 142
    finally:
        _close(window, application)


def test_path_profiles_scroll_and_elide(config: Path) -> None:
    application = _application()
    window = _window(application)
    paths = window.overview.paths
    try:
        long_name = "Cluster_" + "x" * 80
        paths.set_profiles({long_name: {}, **{f"Node{index}": {} for index in range(8)}})
        _settle(application)
        assert paths._profile_scroll.height() <= overview_page._PROFILES_HEIGHT
        assert paths._profile_scroll.verticalScrollBar().maximum() > 0
        assert paths._add.isVisibleTo(paths) and paths._add.geometry().bottom() <= paths.height()
        title = next(child for child in paths._profile_scroll.widget().findChildren(type(paths._add))
                     if child.toolTip().endswith(long_name))
        assert title.text().startswith("☁ C") and title.text().endswith("…")  # the start stays readable
        export = next(child for child in title.parentWidget().findChildren(type(paths._add)) if child.text() == "Export")
        assert title.geometry().right() < export.geometry().left()
        assert paths.height() < 260  # the card stays low (7 profiles made it collapse at 1280x760)
    finally:
        _close(window, application)


@pytest.mark.parametrize("sources", [False, True])
def test_new_dataset_dialog_stays_narrow_with_long_names(config: Path, tmp_path: Path, sources: bool) -> None:
    application = _application()
    window = _window(application)
    records = [DatasetRecord(index, f"{index}_{_LONG[:150]}", raw_path=tmp_path) for index in range(1, 4)]
    dialog = (NewDatasetDialog(window, title="Create next iteration", sources=records, source=records[1],
                               suggest=lambda record: (record.dataset_id + 100, record.name)) if sources
              else NewDatasetDialog(window, dataset_id=1, name=_LONG))
    try:
        dialog.show()
        dialog.adjustSize()
        _settle(application)
        assert dialog.width() <= 600  # was 1467 px: OK and Cancel were off a 1366 screen
        assert dialog.folder.text() == f"Dataset{102 if sources else 1:03d}_{records[1].name if sources else _LONG}"
        assert dialog.folder.toolTip() == dialog.folder.text()
    finally:
        dialog.close()
        _close(window, application)


def test_new_dataset_dialog_explains_names_without_latin_letters(config: Path) -> None:
    application = _application()
    window = _window(application)
    dialog = NewDatasetDialog(window, dataset_id=3, name="肝脏肿瘤")
    try:
        assert "A–Z" in dialog.folder.text() and "letters or numbers" in dialog.folder.text()
        assert not dialog._ok.isEnabled()  # the folder could not be created
        dialog.name.setText("Liver 肝脏")
        assert dialog.folder.text() == "Dataset003_Liver" and dialog._ok.isEnabled()
    finally:
        dialog.close()
        _close(window, application)


def test_guide_window_fits_the_screen(config: Path) -> None:
    application = _application()
    window = _window(application)
    try:
        window._open_guide("README.md")
        application.processEvents()
        guide: QDialog = window._guide
        room = (window.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        assert guide.height() <= room.height() and guide.width() <= room.width()
        guide.close()
    finally:
        _close(window, application)
