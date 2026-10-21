"""Start screen, paths, discovery, new dataset and switcher with broken input (GUI stress test S1-S13)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QStandardPaths, Qt
from qtpy.QtWidgets import QApplication, QLabel, QMessageBox

from nnworkbench import path_setup
from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_creator import normalise_name
from nnworkbench.services.dataset_scanner import discover
from nnworkbench.ui import main_window as main_window_module
from nnworkbench.ui import overview_page
from nnworkbench.ui.main_window import MainWindow
from nnworkbench.ui.new_dataset_dialog import NewDatasetDialog
from nnworkbench.ui.widgets import format_time

_POSIX_PERMISSIONS = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


@pytest.fixture
def messages(monkeypatch) -> list[tuple[str, str]]:
    """Message boxes are recorded instead of shown."""
    shown: list[tuple[str, str]] = []
    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(
            lambda _parent, title, _text, *_args, kind=kind: shown.append((kind, title)) or
            QMessageBox.StandardButton.Ok))
    return shown


@pytest.fixture
def config(tmp_path: Path, monkeypatch, messages) -> Path:
    """An isolated paths.json."""
    path = tmp_path / "config" / "paths.json"
    monkeypatch.setattr(path_setup, "configuration_path", lambda: path)
    return path


def _write_config(path: Path, raw: Path | str = "", preprocessed: Path | str = "", results: Path | str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"raw": str(raw), "preprocessed": str(preprocessed), "results": str(results)}),
                    encoding="utf-8")


def _close(window: MainWindow, application: QApplication) -> None:
    window.close()
    application.processEvents()


def _wait(application: QApplication, condition, seconds: float = 5.0) -> None:
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        application.processEvents()
        time.sleep(0.01)


def _texts(layout) -> list[str]:
    """Texts of the labels in a layout, including nested layouts."""
    texts = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is not None and hasattr(item.widget(), "text"):
            texts.append(item.widget().text())
        elif item.layout() is not None:
            texts.extend(_texts(item.layout()))
    return texts


@pytest.mark.skipif(not _POSIX_PERMISSIONS, reason="needs POSIX permissions and a non-root user")
def test_s1_unreadable_root_has_no_datasets_and_a_red_mark(tmp_path: Path, config: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "Dataset001_A").mkdir(parents=True)
    _write_config(config, raw)
    os.chmod(raw, 0)
    application = _application()
    try:
        assert discover(raw) == {}
        window = MainWindow()  # raised PermissionError: every start crashed
        try:
            mark = window.overview.paths._marks["raw"]
            assert window.records == [] and mark.text() == "✗" and "not readable" in mark.toolTip()
            assert "not readable" in window.overview.paths._status.text()
        finally:
            _close(window, application)
    finally:
        os.chmod(raw, 0o755)


def test_s2_broken_paths_json_does_not_stop_the_start(config: Path) -> None:
    config.parent.mkdir(parents=True)
    config.write_bytes(b'{"raw": "/data/n\xfcnet"}')  # Latin-1
    with pytest.raises(RuntimeError):
        path_setup.configured_paths()
    application = _application()
    window = MainWindow()
    try:
        assert window.overview.paths._local == ("", "", "")
    finally:
        _close(window, application)
    config.unlink()
    config.mkdir()  # a folder instead of the file
    with pytest.raises(RuntimeError):
        path_setup.configured_paths()


def test_s3_far_future_time_is_shown_as_unknown() -> None:
    assert format_time(1e12) == "—"  # year 33658
    assert format_time(1e300) == "—"


def test_s4_odd_path_text_is_not_found_instead_of_raising(tmp_path: Path, config: Path) -> None:
    application = _application()
    window = MainWindow()
    card = window.overview.paths
    try:
        for text in ("~nosuchuser_xyz/nnUNet_raw", str(tmp_path / ("x" * 300))):
            card._fields["raw"].setText(text)
            card._local = card._texts()
            card._update_marks()
            assert card._marks["raw"].text() == "✗"
            window.refresh()  # raised RuntimeError / OSError (name too long)
            assert window.records == []
    finally:
        _close(window, application)


def test_s5_details_show_an_error_instead_of_loading_forever(tmp_path: Path, config: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    (raw / "Dataset001_A").mkdir(parents=True)
    _write_config(config, raw)

    def broken(_record):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(overview_page, "dataset_details", broken)
    application = _application()
    window = MainWindow()
    details = window.overview.details
    try:
        window.overview._table.selectRow(0)
        _wait(application, lambda: not any("Loading" in text for text in _texts(details._info)))
        assert any(text.startswith("Could not read") for text in _texts(details._info))
        deep: list = []
        for _ in range(5000):
            deep = [deep]
        record = DatasetRecord(2, "Deep", raw_path=raw)
        details.set_record(record, {"manifest": {"labels": {"a": 1}, "x": deep}, "raw_folders": [], "plans": [],
                                    "models": 0, "folds": 0})  # json.dumps raised RecursionError
        assert "cannot be shown" in details._json.toPlainText()
    finally:
        _close(window, application)


def test_s6_new_dataset_errors_are_warnings(tmp_path: Path, config: Path, messages, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_config(config, raw)

    def read_only(*_args):
        raise PermissionError(13, "Permission denied", str(raw / "Dataset001_A"))

    monkeypatch.setattr(main_window_module, "create_dataset", read_only)
    monkeypatch.setattr(NewDatasetDialog, "exec", lambda dialog: 1)
    application = _application()
    window = MainWindow()
    try:
        window._create_dataset()  # PermissionError escaped
        assert ("warning", "Dataset not created") in messages
    finally:
        _close(window, application)
    with pytest.raises(ValueError, match="too long"):
        normalise_name("x" * 300)


def test_s7_paths_apply_when_saving_fails(tmp_path: Path, config: Path, messages, monkeypatch) -> None:
    raw = tmp_path / "raw"
    (raw / "Dataset001_A").mkdir(parents=True)

    def read_only(*_args):
        raise PermissionError(13, "Permission denied", str(config))

    monkeypatch.setattr(main_window_module, "save_paths", read_only)
    application = _application()
    window = MainWindow()
    try:
        window.overview.paths._local = (str(raw), "", "")
        window._paths_changed()
        window._paths_changed()
        assert [record.name for record in window.records] == ["A"]
        assert messages == [("warning", "Paths not saved")]  # only once
    finally:
        _close(window, application)


def test_s8_details_cap_description_and_label_chips(tmp_path: Path, config: Path) -> None:
    application = _application()
    window = MainWindow()
    details = window.overview.details
    try:
        manifest = {"description": "word " * 20000, "labels": {f"class{index}": index for index in range(20000)}}
        details.set_record(DatasetRecord(1, "Big", raw_path=tmp_path),
                           {"manifest": manifest, "raw_folders": [], "plans": [], "models": 0, "folds": 0})
        texts = _texts(details._info)
        assert len(texts[0]) <= 1001 and texts[0].endswith("…")
        chips = details._info.itemAt(2).widget().findChildren(type(details._name))
        assert len(chips) == 101 and chips[-1].text() == "+19900 more"
    finally:
        _close(window, application)


def test_s10_switcher_enter_does_not_open_a_hidden_dataset(tmp_path: Path, config: Path) -> None:
    application = _application()
    window = MainWindow()
    chosen = []
    try:
        records = [DatasetRecord(1, "Alpha", raw_path=tmp_path), DatasetRecord(2, "Beta", raw_path=tmp_path)]
        window.records = records
        window.switcher.dataset_chosen.connect(chosen.append)
        window.switcher.open(window.app_bar.switcher_anchor(), records, records[1])
        window.switcher._search.setText("no such dataset")
        window.switcher._open_current()
        assert chosen == []
        window.switcher._search.setText("alp")
        window.switcher._open_current()
        assert [record.name for record in chosen] == ["Alpha"]
    finally:
        window.switcher.hide()
        _close(window, application)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows folder names cannot end in a space")
def test_s11_trailing_space_is_kept_for_an_existing_folder(tmp_path: Path, config: Path) -> None:
    (tmp_path / "raw ").mkdir()
    application = _application()
    window = MainWindow()
    card = window.overview.paths
    try:
        card._fields["raw"].setText(str(tmp_path / "raw "))
        card._edited()
        assert card._local[0] == str(tmp_path / "raw ") and card._marks["raw"].text() == "✓"
        card._fields["raw"].setText(f"  {tmp_path / 'raw'}  ")  # no such folder: the spaces are typos
        card._edited()
        assert card._local[0] == str(tmp_path / "raw")
    finally:
        _close(window, application)


def test_s12_new_dataset_dialog_shows_the_folder_name(config: Path) -> None:
    application = _application()
    window = MainWindow()
    dialog = NewDatasetDialog(window, dataset_id=12)
    try:
        dialog.name.setText("Leber CT 数据")
        assert dialog.folder.text() == "Dataset012_Leber_CT"
        dialog.name.setText("数据")
        assert "letters or numbers" in dialog.folder.text()
    finally:
        dialog.close()
        _close(window, application)


def test_s13_dataset_json_text_is_not_rendered_as_html(tmp_path: Path, config: Path) -> None:
    application = _application()
    window = MainWindow()
    details = window.overview.details
    try:
        manifest = {"description": "<h1>BIG</h1>", "labels": {"<b>x</b>": 1}, "file_ending": "<script>"}
        details.set_record(DatasetRecord(1, "<i>Name</i>", raw_path=tmp_path),
                           {"manifest": manifest, "raw_folders": [], "plans": [], "models": 0, "folds": 0})
        description = details._info.itemAt(0).widget()
        assert description.textFormat() == Qt.TextFormat.PlainText
        assert details._name.textFormat() == Qt.TextFormat.PlainText
        assert any("&lt;script&gt;" in text for text in _texts(details._info))
        chip = details._info.itemAt(2).widget().findChildren(type(details._name))[0]
        assert chip.textFormat() == Qt.TextFormat.PlainText
    finally:
        _close(window, application)


def test_dataset_ids_nnunet_refuses_are_reported(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_scanner import id_badge, scan_datasets

    raw, preprocessed, results = (tmp_path / name for name in ("raw", "preprocessed", "results"))
    for folder in (raw / "Dataset042_Cells", preprocessed / "Dataset042_Cells", results / "Dataset042_Cells",
                   raw / "Dataset420_A", results / "Dataset420_B",  # one ID, two names across roots
                   raw / "Dataset0007_Old", raw / "Dataset007_New",  # nnU-Net uses Dataset007_New, ignores 0007
                   raw / "Dataset500_X", raw / "Dataset5000_Y",  # nnU-Net's prefix Dataset500 matches both
                   raw / "Dataset0099_Lonely"):  # nnU-Net looks for Dataset099
        folder.mkdir(parents=True)
    problems = {record.dataset_id: record.id_problem for record in scan_datasets(raw, preprocessed, results)}
    assert problems[42] == ("", "")
    assert problems[420][0] == "duplicate ID" and "Dataset420_A, Dataset420_B" in problems[420][1]
    records = {record.dataset_id: record for record in scan_datasets(raw, preprocessed, results)}
    assert problems[7] == ("", "") and records[7].raw_path.name == "Dataset007_New"  # as nnU-Net
    assert records[7].ignored_folders == ("Dataset0007_Old",)
    assert id_badge(records[7])[:2] == ("folder ignored by nnU-Net", "orange")
    assert id_badge(records[420])[1] == "red" and id_badge(records[42]) is None
    assert problems[500][0] == "duplicate ID" and "Dataset5000_Y" in problems[500][1]
    assert problems[5000] == ("", "")
    assert problems[99][0] == "name not found by nnU-Net" and "Dataset099_Lonely" in problems[99][1]


def test_duplicate_id_badge_on_the_overview_and_the_data_page(tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    from nnworkbench import command_runner
    from nnworkbench.services.dataset_scanner import scan_datasets
    from nnworkbench.ui.dataset_state import load_state
    from nnworkbench.ui.main_window import MainWindow

    raw, results = tmp_path / "raw", tmp_path / "results"
    (raw / "Dataset420_A" / "imagesTr").mkdir(parents=True)
    (results / "Dataset420_B").mkdir(parents=True)
    records = scan_datasets(raw, None, results)
    application = _application()
    window = MainWindow()
    try:
        window.overview.set_records(records)
        cell = window.overview._table.cellWidget(0, 1)
        assert cell is not None and any(child.text() == "duplicate ID" for child in cell.findChildren(QLabel))
        window.overview.details.set_record(records[0], None)
        assert any(child.text() == "duplicate ID" for child in window.overview.details.findChildren(QLabel))
        state = load_state(records[0])
        assert any("used by 2 folders" in problem for problem in state.data_problems())
        assert state.data_state()[0] == "error"
        badges = window.dataset_page.pages["data"].header_badges(state)
        assert any(item.text() == "duplicate ID" for item in badges)
        for item in badges:
            item.deleteLater()
    finally:
        window.close()
        application.processEvents()

    monkeypatch.setattr(command_runner, "configured_paths", lambda: {
        "nnUNet_raw": str(raw), "nnUNet_preprocessed": "", "nnUNet_results": str(results)})
    assert command_runner._run_builtin_command(["status", "-d", "420", "--json", "--no-device"]) == 0
    assert any("used by 2 folders" in problem for problem in json.loads(capsys.readouterr().out)["problems"])
