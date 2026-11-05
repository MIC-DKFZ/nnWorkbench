"""Regressions found on Windows (line ends, encodings, exit codes, paths); they run on every system."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from nnworkbench import path_setup
from nnworkbench.json_output import use_utf8_output
from nnworkbench.services import dataset_creator
from nnworkbench.services.dataset_description import (
    Answer,
    create_description,
    description_classes,
    read_answers,
    read_description,
    save_answers,
)
from nnworkbench.services.dataset_json import write_manifest
from nnworkbench.services.process_runner import crash_note, exit_status


def _dataset(tmp_path: Path, labels: dict | None = None) -> Path:
    folder = tmp_path / "Dataset001_Cells"
    folder.mkdir()
    (folder / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": labels or {"background": 0, "cell": 1},
        "numTraining": 0, "file_ending": ".nii.gz"}), encoding="utf-8")
    return folder


def test_description_is_written_with_lf_and_crlf_files_are_read(tmp_path: Path) -> None:
    folder = _dataset(tmp_path)
    path = create_description(folder)
    assert b"\r\n" not in path.read_bytes()  # Path.write_text would write "\r\n" on Windows
    text = read_description(path)
    fields = read_answers(text, folder)
    assert fields and description_classes(text) == [(0, "background"), (1, "cell")]
    # the same file with Windows line ends (Notepad, git autocrlf) and a BOM reads the same
    path.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    assert read_description(path) == text
    answers = read_answers(read_description(path), folder)
    answers["type"] = Answer("CT scans", ())
    save_answers(folder, answers)
    saved = path.read_bytes()
    assert b"\r" not in saved and read_answers(read_description(path), folder)["type"].text == "CT scans"


def test_dataset_json_is_ascii_so_nnunet_reads_it_with_any_encoding(tmp_path: Path) -> None:
    # nnU-Net reads dataset.json with open()'s default encoding: cp1252 on Windows
    folder = _dataset(tmp_path)
    write_manifest(folder, {"channel_names": {"0": "Färbung"}, "labels": {"background": 0, "Ősejt": 1},
                            "numTraining": 0, "file_ending": ".nii.gz"})
    raw = (folder / "dataset.json").read_bytes()
    assert raw.isascii() and json.loads(raw.decode("cp1252"))["labels"] == {"background": 0, "Ősejt": 1}
    dataset_creator._write_manifest(folder, {"labels": {"Zellkern_ü": 1}})
    assert (folder / "dataset.json").read_bytes().isascii()


def test_windows_crash_status_is_passed_on() -> None:
    # 0xC0000005 as the same 32 bits: SystemExit(3221225477) would overflow and exit with -1
    assert exit_status(3221225477, windows=True) == -1073741819
    assert exit_status(1, windows=True) == 1 and exit_status(0, windows=True) == 0
    note = crash_note(r"C:\env\Scripts\nnUNetv2_train.exe", 3221225477, windows=True)
    assert note == "nnUNetv2_train.exe ended with Windows status 0xC0000005 (access violation)."
    assert crash_note("nnUNetv2_train.exe", 1, windows=True) == ""
    assert exit_status(-9, windows=False) == 137 and crash_note("x", -9, windows=False) == ""


def test_text_output_becomes_utf8_when_the_stream_is_not(monkeypatch) -> None:
    # a pipe on Windows: the ANSI code page, which cannot encode "＋" or "→"
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    use_utf8_output()
    print("＋ → ä")
    stream.flush()
    assert stream.encoding == "utf-8" and stream.buffer.getvalue().decode("utf-8").strip() == "＋ → ä"


def test_non_ascii_paths_are_flagged_only_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(path_setup, "_utf8_code_page", lambda: False)
    assert path_setup.unusable_for_nnunet(r"C:\Users\Jürgen\nnUNet_raw", windows=True)
    assert not path_setup.unusable_for_nnunet(r"C:\nnUNet\nnUNet_raw", windows=True)
    assert not path_setup.unusable_for_nnunet("/home/jürgen/nnUNet_raw", windows=False)
    monkeypatch.setattr(path_setup, "_utf8_code_page", lambda: True)  # Windows set to UTF-8: nnU-Net works there
    assert not path_setup.unusable_for_nnunet(r"C:\Users\Jürgen\nnUNet_raw", windows=True)


def test_set_paths_lists_non_ascii_folders(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(path_setup, "configuration_path", lambda: tmp_path / "config" / "paths.json")
    monkeypatch.setattr(path_setup, "unusable_for_nnunet", lambda path, windows=None: not str(path).isascii())
    monkeypatch.setattr(sys, "argv", ["nnworkbench-set-paths", "--raw", str(tmp_path / "räw"), "--preprocessed",
                                      str(tmp_path / "pre"), "--results", str(tmp_path / "res"), "--json"])
    path_setup.set_paths_main()
    result = json.loads(capsys.readouterr().out)
    assert result["non_ascii"] == [str(tmp_path / "räw")]


@pytest.mark.parametrize("cpus, windows, expected", [(128, True, 61), (128, False, 128), (8, True, 8)])
def test_worker_count_is_capped_on_windows(monkeypatch, cpus: int, windows: bool, expected: int) -> None:
    from nnworkbench.services import label_analysis

    monkeypatch.setattr(label_analysis.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(label_analysis, "MAX_WORKERS", 61 if windows else None)
    assert label_analysis.available_workers() == expected


def test_cli_does_not_import_torch() -> None:
    import subprocess

    code = "import sys, nnworkbench.command_runner; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"  # ~2 s per call, and a broken torch would break `status`
