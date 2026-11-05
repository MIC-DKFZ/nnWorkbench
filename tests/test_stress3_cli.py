"""Services and command line after the user stress test (round 3: personas expert E, chaos C, agent A, beginner B,
input P). Each test fails without its fix."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nnworkbench import command_runner, path_setup
from nnworkbench.cli_tools import matching_models, suggestion, validation_report
from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_contents import trained_models
from nnworkbench.services.dataset_description import add_image
from nnworkbench.services.dataset_json import file_ending_error, manifest_source
from nnworkbench.services.defaults import preprocessing_configuration
from nnworkbench.services.workflow_status import fold_info, newest_label_time, plans_overview
from nnworkbench.status_report import build_status
from nnworkbench.ui.dataset_state import load_state

NAME = "Dataset007_Cells"
MODEL = "nnUNetTrainer__nnUNetPlans__3d_fullres"


def _roots(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in ("raw", "preprocessed", "results")}
    for folder in roots.values():
        folder.mkdir(exist_ok=True)
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {
        f"nnUNet_{name}": str(folder) for name, folder in roots.items()})
    return roots


def _raw(raw_root: Path, images: tuple[str, ...] = ("a_0000.nii.gz", "b_0000.nii.gz"),
         labels: tuple[str, ...] = ("a.nii.gz", "b.nii.gz"), **manifest) -> Path:
    folder = raw_root / NAME
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    for name in images:
        (folder / "imagesTr" / name).touch()
    for name in labels:
        (folder / "labelsTr" / name).touch()
    (folder / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1}, "numTraining": len(images),
        "file_ending": ".nii.gz", **manifest}), encoding="utf-8")
    return folder


def _state(raw: Path | None = None, preprocessed: Path | None = None, results: Path | None = None):
    return load_state(DatasetRecord(7, "Cells", raw, preprocessed, results))


def _fold(results_root: Path, fold: str = "0", log: str = "", age: float = 0.0) -> Path:
    folder = results_root / NAME / MODEL / f"fold_{fold}"
    folder.mkdir(parents=True)
    path = folder / "training_log_2026_1_1_00_00_00.txt"
    path.write_text(log, encoding="utf-8")
    os.utime(path, (time.time() - age,) * 2)
    return folder


# ---------------------------------------------------------------- paths (P2, P3, P6, A1, E4)

@pytest.mark.parametrize(("typed", "expected"), [
    ('"/data/nnUNet raw"', "/data/nnUNet raw"), ("'/data/x'", "/data/x"), ("file:///data/a%20b", "/data/a b"),
    ("/data/a b", "/data/a b"), ("$NNWB_TEST_ROOT/x", "/root-from-env/x")])
def test_p6_typed_path_takes_pasted_forms(typed: str, expected: str, monkeypatch) -> None:
    monkeypatch.setenv("NNWB_TEST_ROOT", "/root-from-env")
    assert Path(path_setup.typed_path(typed)) == Path(expected)


def test_p2_a1_set_paths_saves_absolute_expanded_paths_and_lists_missing(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "config" / "paths.json"
    monkeypatch.setattr(path_setup, "configuration_path", lambda: config)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # "~" on Windows
    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw").mkdir()
    # as pasted from the GUI's Export (quoted: the remote shell does not expand ~ or $HOME)
    monkeypatch.setattr(sys, "argv", ["nnworkbench-set-paths", "--raw", "~/raw", "--preprocessed", "$HOME/pre",
                                      "--results", "relative/results", "--json"])
    path_setup.set_paths_main()
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved == {"raw": str(tmp_path / "raw"), "preprocessed": str(tmp_path / "pre"),
                     "results": str(tmp_path / "relative" / "results")}
    assert json.loads(capsys.readouterr().out)["missing"] == [str(tmp_path / "pre"),
                                                             str(tmp_path / "relative" / "results")]


def test_e4_own_trainer_folders_are_kept(monkeypatch) -> None:
    monkeypatch.setenv("nnUNet_extTrainer", "/my/trainers")
    assert path_setup.trainer_folders() == os.pathsep.join([str(path_setup.trainer_directory()), "/my/trainers"])
    monkeypatch.delenv("nnUNet_extTrainer")
    assert path_setup.trainer_folders() == str(path_setup.trainer_directory())


# ---------------------------------------------------------------- training guard (E10, A5, A6, C7, C11)

def _no_run(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(command_runner, "find_executable", lambda name: name)
    monkeypatch.setattr(command_runner.shutil, "which", lambda name: name)
    monkeypatch.setattr(command_runner.subprocess, "run",
                        lambda arguments, **_kwargs: calls.append(arguments) or SimpleNamespace(returncode=0))
    monkeypatch.delenv(command_runner.YES_ENVIRONMENT, raising=False)
    return calls


@pytest.mark.parametrize("command", [
    ["nnUNetv2_train", "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "7", "3d_fullres", "0"],
    ["nnUNetv2_train", "7", "3d_fullres", "0", "-tr=nnUNetTrainer", "-p=nnUNetPlans"],
    ["nnUNetv2_train", "007", "3d_fullres", "00"]])
def test_e10_a5_retrain_guard_reads_arguments_as_nnunet(command, tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path, monkeypatch)
    _fold(roots["results"], age=3600)
    calls = _no_run(monkeypatch)
    with pytest.raises(ValueError, match="already has a training"):
        command_runner.run_with_configured_paths(command)
    assert calls == []


def test_a6_c7_a_fold_still_training_needs_yes_also_with_continue(tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path, monkeypatch)
    _fold(roots["results"])  # log written just now
    calls = _no_run(monkeypatch)
    command = ["nnUNetv2_train", "7", "3d_fullres", "0", "--c"]
    with pytest.raises(ValueError, match="still training"):
        command_runner.run_with_configured_paths(command)
    assert calls == []
    assert command_runner.run_with_configured_paths([*command, "--yes"]) == 0 and calls == [command]


def test_a6_slow_epochs_keep_a_fold_training_and_c7_a_future_log_does_not(tmp_path) -> None:
    slow = _fold(tmp_path, "0", "2026-01-01 10:00:00.000000: Epoch time: 3600.0 s\n", age=40 * 60)
    assert fold_info(slow).status == "training"  # two epochs of an hour, not 15 minutes
    future = _fold(tmp_path, "1", age=-3 * 24 * 3600)
    assert fold_info(future).status == "stopped"


def test_c11_killed_job_exits_with_128_plus_signal(tmp_path, monkeypatch) -> None:
    _roots(tmp_path, monkeypatch)
    _no_run(monkeypatch)
    monkeypatch.setattr(command_runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=-9))
    assert command_runner.run_with_configured_paths(["nnUNetv2_predict", "-h"]) == 137


# ---------------------------------------------------------------- agent-facing CLI (A8-A16)

def _main(monkeypatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["nnworkbench-run", *arguments])
    with pytest.raises(SystemExit) as stop:
        command_runner.main()
    return stop.value.code


def test_a8_os_errors_and_others_are_json_errors(tmp_path, monkeypatch, capfd) -> None:
    _roots(tmp_path, monkeypatch)
    for error in (PermissionError(13, "Permission denied"), KeyError("unexpected")):
        def fail(_command, error=error):
            raise error
        monkeypatch.setattr(command_runner, "_run_builtin_command", fail)
        assert _main(monkeypatch, "create-dataset", "--json") == 1
        assert "error" in json.loads(capfd.readouterr().out)  # the JSON goes to the real stdout (descriptor)


def test_a9_reduce_checkpoints_without_yes_prints_one_json(tmp_path, capsys) -> None:
    model = tmp_path / MODEL
    (model / "fold_0").mkdir(parents=True)
    (model / "fold_0" / "checkpoint_best.pth").touch()
    assert command_runner._run_builtin_command(["reduce-checkpoints", str(model), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)  # one document ("Extra data" before)
    assert "--yes" in result["error"] and result["actions"]
    assert (model / "fold_0" / "checkpoint_best.pth").is_file()


def test_a10_validation_open_refuses_json(tmp_path, monkeypatch) -> None:
    _roots(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stop:
        command_runner._run_builtin_command(["validation", "-d", "7", "--open", "a", "--json"])
    assert stop.value.code == 2


@pytest.mark.parametrize("ending", ["nii.gz", ".jpg", ".nii.gz "])
def test_a12_p5_create_dataset_and_status_refuse_unusable_file_endings(ending, tmp_path, monkeypatch) -> None:
    _roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="file-ending"):
        command_runner._run_builtin_command(["create-dataset", "-d", "7", "--name", "Cells", "--labels", "cell",
                                             "--channels", "light", "--file-ending", ending])
    assert not any((tmp_path / "raw").iterdir())
    assert file_ending_error({"file_ending": ending}) is not None or ending == ".jpg"


def test_a16_help_lists_the_tools(monkeypatch, capsys) -> None:
    assert _main(monkeypatch, "--help") == 0
    assert "create-dataset" in capsys.readouterr().out


# ---------------------------------------------------------------- results and plans (E2, E8, E11-E13, A14)

def test_e2_old_npz_preprocessing_and_e11_inherited_configurations(tmp_path) -> None:
    preprocessed = tmp_path / NAME
    (preprocessed / "nnUNetPlans_3d_fullres").mkdir(parents=True)
    for case in ("a", "b"):
        (preprocessed / "nnUNetPlans_3d_fullres" / f"{case}.npz").touch()
        (preprocessed / "nnUNetPlans_3d_fullres" / f"{case}.pkl").touch()
    plans = {"configurations": {
        "3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres", "patch_size": [64, 64, 64]},
        "3d_fullres_bs4": {"inherits_from": "3d_fullres", "batch_size": 4},
        "loop": {"inherits_from": "loop"}, "odd": {"inherits_from": ["3d_fullres"]}}}
    (preprocessed / "nnUNetPlans.json").write_text(json.dumps(plans), encoding="utf-8")
    configurations = {config.name: config for config in plans_overview(preprocessed)[0].configurations}
    assert configurations["3d_fullres"].preprocessed_cases == 2
    inherited = configurations["3d_fullres_bs4"]
    assert inherited.preprocessed_cases == 2 and inherited.patch_size == (64, 64, 64) and inherited.batch_size == 4


def test_e8_model_copies_in_subfolders_are_not_models(tmp_path) -> None:
    for model in (tmp_path / MODEL, tmp_path / "old_runs" / MODEL):
        (model / "fold_0").mkdir(parents=True)
    assert trained_models(tmp_path) == [Path(MODEL)]


def test_e12_results_only_dataset_reads_the_model_dataset_json(tmp_path) -> None:
    (tmp_path / MODEL / "fold_0").mkdir(parents=True)
    (tmp_path / MODEL / "dataset.json").write_text("{}", encoding="utf-8")
    assert manifest_source(None, None, tmp_path) == ("results", tmp_path / MODEL)


def _validated(results: Path, model: str, folds=("0",)) -> None:
    for fold in folds:
        folder = results / NAME / model / f"fold_{fold}"
        (folder / "validation").mkdir(parents=True)
        (folder / "checkpoint_final.pth").touch()
        (folder / "validation" / "summary.json").write_text(json.dumps({
            "foreground_mean": {"Dice": 0.8}, "mean": {"1": {"Dice": 0.8}}, "metric_per_case": []}), encoding="utf-8")


def test_e13_a14_model_and_fold_filters(tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path, monkeypatch)
    _validated(roots["results"], MODEL)
    _validated(roots["results"], MODEL + "_bs4")
    assert [model["model"] for model in validation_report(roots, 7, model=MODEL)["models"]] == [MODEL]
    assert matching_models([SimpleNamespace(label=MODEL), SimpleNamespace(label=MODEL + "_bs4")], "bs4")[0].label \
        == MODEL + "_bs4"
    assert validation_report(roots, 7, fold="fold_0")["models"]  # the folder name works for fold 0
    with pytest.raises(ValueError, match="Models:"):  # not "no validation results yet"
        validation_report(roots, 7, model="BetterIgnore")


# ---------------------------------------------------------------- data checks (B1, B4, B6, B7, B13, B14, C8, A4)

def test_e1_b1_2d_images_preprocess_2d() -> None:
    assert preprocessing_configuration({"file_ending": ".png"}) == "2d"
    assert preprocessing_configuration({"file_ending": ".tif"}, ["2d"]) == "2d"  # nnU-Net planned only 2d
    assert preprocessing_configuration({"file_ending": ".tif"}, ["2d", "3d_fullres"]) == "3d_fullres"
    assert preprocessing_configuration({"file_ending": ".nii.gz"}) == "3d_fullres"


def test_b4_images_and_labels_with_another_ending_and_b14_jpeg(tmp_path) -> None:
    raw = _raw(tmp_path, images=("a_0000.tif", "b_0000.TIF", "c_0000.jpg"), labels=("a.tif", "b.png"),
               file_ending=".tif", numTraining=2)
    problems = " ".join(_state(raw).data_problems())
    assert "1 image (e.g. b_0000.TIF) and 1 label file (e.g. b.png) do not end with file_ending" in problems
    assert "JPEG" in problems and "c_0000.jpg" in problems


def test_b6_extra_channel_and_b13_label_without_image(tmp_path) -> None:
    raw = _raw(tmp_path, images=("a_0000.nii.gz", "a_0001.nii.gz"), labels=("a.nii.gz", "a_0000.nii.gz"),
               numTraining=1)
    problems = " ".join(_state(raw).data_problems())
    assert "channel _0001, which dataset.json does not list" in problems
    assert "1 label file match no image (e.g. a_0000.nii.gz)" in problems


def test_b7_undefined_label_ids_from_the_statistics(tmp_path) -> None:
    analysis = {"files_analyzed": 2, "undefined_label_ids": {"a.nii.gz": [1, 255]}}  # 1 is defined by now
    state = dataclasses.replace(_state(_raw(tmp_path)), analysis=analysis, analysis_outdated=0)
    assert state.undefined_label_ids() == [255]
    assert dataclasses.replace(state, analysis_outdated=1).undefined_label_ids() == []  # labels changed since
    assert any("255" in problem for problem in state.data_problems())


def test_b14_images_in_the_dataset_folder_and_c9_missing_folder(tmp_path) -> None:
    raw = _raw(tmp_path, images=(), labels=(), numTraining=0)
    (raw / "img 1.tif").touch()
    assert "in the dataset folder itself" in _state(raw).data_state()[1]
    gone = tmp_path / "Dataset007_Gone"
    assert "no longer exists" in _state(gone).data_state()[1]


def test_c8_label_from_the_future_does_not_outdate_preprocessing(tmp_path) -> None:
    raw = _raw(tmp_path)
    os.utime(raw / "labelsTr" / "a.nii.gz", (time.time() + 30 * 24 * 3600,) * 2)
    os.utime(raw / "labelsTr" / "b.nii.gz", (1_000_000,) * 2)
    assert newest_label_time(raw) == 1_000_000


def test_a4_status_lists_labeled_and_unlabeled_cases(tmp_path) -> None:
    raw = _raw(tmp_path, images=("a_0000.nii.gz", "b_0000.nii.gz"), labels=("a.nii.gz",))
    status = build_status(_state(raw))
    assert status["labeled_cases"] == 1 and status["unlabeled_cases"] == ["b"]


def test_a13_suggest_warns_about_the_state(tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path, monkeypatch)
    _raw(roots["raw"], images=("a_0000.nii.gz",), labels=())
    result = suggestion(roots, 7, "preprocess")
    assert result["warnings"] and "without label" in result["warnings"][0]


def test_c5_adding_an_image_never_recreates_a_removed_dataset_folder(tmp_path) -> None:
    image = tmp_path / "example.png"
    image.write_bytes(b"png")
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        add_image(tmp_path / "Dataset042_Renamed", image)
    assert not (tmp_path / "Dataset042_Renamed").exists()


def test_p14_search_matches_the_folder_name() -> None:
    record = DatasetRecord(8, "Liver_CT")
    assert all(record.matches(text) for text in ("Dataset008_Liver_CT", "dataset008", "Liver CT", "8", ""))
    assert not record.matches("Lung")


# ---------------------------------------------------------------- decisions of 2026-10-22

def test_e5_nnunet_folder_variables_without_saved_paths_and_when_they_differ(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "paths.json"
    monkeypatch.setattr(path_setup, "configuration_path", lambda: config)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "env_raw"))
    paths = path_setup.configured_paths()  # no paths.json: the variables
    assert paths["nnUNet_raw"] == str(tmp_path / "env_raw") and paths["nnUNet_results"] == ""
    path_setup.save_paths(str(tmp_path / "raw"), "", "")
    assert path_setup.configured_paths()["nnUNet_raw"] == str(tmp_path / "raw")  # saved paths win
    differences = path_setup.environment_differences(path_setup.configured_paths())
    assert len(differences) == 1 and "env_raw" in differences[0]
    monkeypatch.setattr(sys, "argv", ["nnworkbench-get-paths", "--json"])
    path_setup.get_paths_main()
    assert json.loads(capsys.readouterr().out)["environment_differs"] == differences


def test_a19_list_datasets(tmp_path, monkeypatch, capsys) -> None:
    roots = _roots(tmp_path, monkeypatch)
    for name in ("Dataset007_Cells", "Dataset008_Cells_iter1", "Dataset009_Cells_iter1"):
        (roots["raw"] / name).mkdir()
    (roots["results"] / "Dataset007_Cells").mkdir()
    assert command_runner._run_builtin_command(["list", "--json"]) == 0
    datasets = json.loads(capsys.readouterr().out)["datasets"]
    assert [(item["id"], item["base_name"], item["iteration"]) for item in datasets] == [
        (7, "Cells", 0), (8, "Cells", 1), (9, "Cells", 1)]  # a branch keeps the name, the ID tells them apart
    assert datasets[0]["raw"] and datasets[0]["results"] and not datasets[0]["preprocessed"]


@pytest.mark.parametrize("windows", [False, True])
def test_p7_one_quoting_for_bash_and_powershell(windows, monkeypatch) -> None:
    from nnworkbench.services import commands

    monkeypatch.setattr(commands, "WINDOWS", windows)
    assert commands.shell_join(["nnworkbench-run", "-i", r"C:\R&D\in (1)", "-o", "/data/my out", "-f", "0"]) == \
        r"nnworkbench-run -i 'C:\R&D\in (1)' -o '/data/my out' -f 0"
    assert commands.shell_quote("$HOME,x@y") == "'$HOME,x@y'"  # PowerShell: $ expands, , makes a list, @ splats
    assert commands.shell_quote("Ana's") == ("'Ana''s'" if windows else "'Ana'\"'\"'s'")
    assert commands.join_commands(["a", "b"]) == "a\nb"  # no && (Windows PowerShell 5.1)
    assert path_setup.set_paths_command("~/raw", "/p", "/r") == "nnworkbench-set-paths --raw '~/raw' --preprocessed /p --results /r"


def test_p8_windows_arguments_read_like_the_shells(monkeypatch) -> None:
    from nnworkbench.services import commands

    monkeypatch.setattr(commands, "WINDOWS", True)
    assert commands.split_arguments(r"""-x="C:\My Folder\w.pth" -chk 'checkpoint best.pth' "C:\a b""x" """) == [
        r"-x=C:\My Folder\w.pth", "-chk", "checkpoint best.pth", r"C:\a bx"]
    with pytest.raises(ValueError):
        commands.split_arguments("-x 'open")


def test_b4_prediction_input_with_another_file_ending(tmp_path) -> None:
    from nnworkbench.services.workflow_status import prediction_status

    for name in ("a_0000.tif", "b_0000.TIF"):
        (tmp_path / name).touch()
    cases = {case.case_id: case for case in prediction_status(tmp_path, None, 1, file_ending=".tif")}
    assert cases["a"].ready and not cases["b"].ready and cases["b"].wrong_ending == ("b_0000.TIF",)
    assert not cases["b"].for_channels(1).ready  # kept when checked against the channels again


def _png(path: Path, mode: str) -> None:
    from PIL import Image

    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    Image.new(mode, (8, 6), (10,) * channels if channels > 1 else 10).save(path)


@pytest.mark.parametrize(("mode", "names", "problem"), [
    ("RGB", ["R", "G", "B"], ""), ("RGB", ["grey"], "3 colour channels"), ("L", ["grey"], ""),
    ("LA", ["grey"], "grey + alpha"), ("RGBA", ["R", "G", "B", "A"], "")])
def test_b5_rgb_files_count_as_three_channels(mode, names, problem, tmp_path) -> None:
    raw = _raw(tmp_path, images=(), labels=("a.png",), file_ending=".png", numTraining=1,
               channel_names={str(index): name for index, name in enumerate(names)})
    _png(raw / "imagesTr" / "a_0000.png", mode)
    state = _state(raw)
    assert state.raw.required_suffixes == ("0000",) and state.raw.cases[0].channels_present
    problems = " ".join(state.data_problems())
    assert (problem in problems) if problem else not problems


def test_b5_predict_takes_rgb_input_as_all_channels(tmp_path) -> None:
    from nnworkbench.services.workflow_status import prediction_status

    _png(tmp_path / "x_0000.png", "RGB")
    assert prediction_status(tmp_path, None, file_ending=".png")[0].for_channels(("0000",)).ready
    assert not prediction_status(tmp_path, None, file_ending=".png")[0].for_channels(3).ready  # one file per name


def _listed(tmp_path: Path) -> Path:
    other = tmp_path / "elsewhere"
    other.mkdir()
    listing = {}
    for index in range(3):
        _png(other / f"scan{index}.png", "L")
        if index < 2:  # case2 has no label yet
            _png(other / f"scan{index}_mask.png", "L")
        listing[f"case{index}"] = {"images": [f"../elsewhere/scan{index}.png" if index else
                                              str(other / f"scan{index}.png")],
                                   "label": f"$NNWB_LISTED/scan{index}_mask.png"}
    return _raw(tmp_path / "raw", images=(), labels=(), file_ending=".png", numTraining=3, dataset=listing)


def test_e3_cases_from_the_dataset_file_list(tmp_path, monkeypatch) -> None:
    from nnworkbench.services.label_analysis import label_files

    monkeypatch.setenv("NNWB_LISTED", str(tmp_path / "elsewhere"))
    raw = _listed(tmp_path)
    shutil.copytree(tmp_path / "elsewhere", raw.parent / "elsewhere")  # "../elsewhere" from the dataset folder
    state = _state(raw)  # imagesTr is empty: nnU-Net reads the list
    assert [case.case_id for case in state.raw.cases] == ["case0", "case1", "case2"]
    assert state.raw.file_list and state.raw.image_files == 3 and state.raw.label_files == 2
    problems = " ".join(state.data_problems())
    assert "1 image without label" in problems and "not found" not in problems  # a label still to annotate
    assert [path.name for path in label_files(raw)] == ["scan0_mask.png", "scan1_mask.png"]


def test_e3_new_iteration_copies_the_listed_files(tmp_path, monkeypatch) -> None:
    from nnworkbench.services.dataset_creator import create_next_iteration

    monkeypatch.setenv("NNWB_LISTED", str(tmp_path / "elsewhere"))
    raw = _listed(tmp_path)
    shutil.copytree(tmp_path / "elsewhere", raw.parent / "elsewhere")
    created = create_next_iteration(raw.parent, [raw.parent], raw, 8, "Cells_iter1")
    assert sorted(path.name for path in (created / "imagesTr").iterdir()) == [
        "case0_0000.png", "case1_0000.png", "case2_0000.png"]
    assert sorted(path.name for path in (created / "labelsTr").iterdir()) == ["case0.png", "case1.png"]
    assert "dataset" not in json.loads((created / "dataset.json").read_text(encoding="utf-8"))
    assert json.loads((raw / "dataset.json").read_text(encoding="utf-8"))["dataset"]  # the source is unchanged


def test_c7_b9_fold_status_says_how_sure_and_ended_gui_jobs(tmp_path) -> None:
    fold = _fold(tmp_path, age=120)
    info = fold_info(fold)
    assert info.status == "training" and info.log_note().startswith("log updated 2 min ago")
    assert fold_info(fold, ended=time.time()).status == "stopped"  # this GUI's job for it has ended
    old = fold_info(_fold(tmp_path, "1", age=3 * 3600))
    assert old.status == "stopped" and "no log update for 3 h" in old.log_note()


# ---------------------------------------------------------------- new iteration: linked images, inherits_from

def _round(raw_root: Path, name: str = "Dataset042_Cells") -> Path:
    folder = raw_root / name
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    for index in range(3):
        _png(folder / "imagesTr" / f"cells_{index}_0000.png", "L")
        if index < 2:
            _png(folder / "labelsTr" / f"cells_{index}.png", "L")
    (folder / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "grey"}, "labels": {"background": 0, "cell": 1}, "numTraining": 3,
        "file_ending": ".png"}), encoding="utf-8")
    return folder


def test_new_iteration_links_the_images_and_names_its_source(tmp_path) -> None:
    from nnworkbench.services.dataset_creator import create_next_iteration

    raw = tmp_path / "raw"
    source = _round(raw)
    linked = create_next_iteration(raw, [raw], source, 43, "Cells_iter1", link_images=True)
    manifest = json.loads((linked / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["inherits_from"] == "Dataset042_Cells"
    assert manifest["dataset"]["cells_0"] == {"images": ["../Dataset042_Cells/imagesTr/cells_0_0000.png"],
                                              "label": "labelsTr/cells_0.png"}
    assert not any((linked / "imagesTr").iterdir())  # no copies
    assert sorted(path.name for path in (linked / "labelsTr").iterdir()) == ["cells_0.png", "cells_1.png"]
    state = _state(linked)
    assert state.inherits_from == "Dataset042_Cells" and state.raw.file_list and len(state.raw.cases) == 3
    assert build_status(state)["inherits_from"] == "Dataset042_Cells"
    # a round from a linked round points at the original images, not through the middle one
    chained = create_next_iteration(raw, [raw], linked, 44, "Cells_iter2", link_images=True)
    images = json.loads((chained / "dataset.json").read_text(encoding="utf-8"))["dataset"]["cells_2"]["images"]
    assert images == ["../Dataset042_Cells/imagesTr/cells_2_0000.png"]
    # copied from a linked round: its own imagesTr again
    copied = create_next_iteration(raw, [raw], linked, 45, "Copied")
    assert (copied / "imagesTr" / "cells_2_0000.png").is_file()
    assert "dataset" not in json.loads((copied / "dataset.json").read_text(encoding="utf-8"))


def test_images_in_imagestr_outside_the_file_list_are_flagged(tmp_path) -> None:
    from nnworkbench.services.dataset_creator import create_next_iteration

    raw = tmp_path / "raw"
    linked = create_next_iteration(raw, [raw], _round(raw), 43, "Cells_iter1", link_images=True)
    _png(linked / "imagesTr" / "new_0000.png", "L")  # nnU-Net uses only the listed files
    assert any("new_0000.png" in problem and "file list" in problem for problem in _state(linked).data_problems())


def test_cli_new_iteration_link_images_and_list(tmp_path, monkeypatch, capsys) -> None:
    roots = _roots(tmp_path, monkeypatch)
    _round(roots["raw"])
    assert command_runner._run_builtin_command(["new-iteration", "-d", "42", "--link-images", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == "linked"
    assert command_runner._run_builtin_command(["list", "--json"]) == 0
    datasets = json.loads(capsys.readouterr().out)["datasets"]
    assert [item["inherits_from"] for item in datasets] == [None, "Dataset042_Cells"]
