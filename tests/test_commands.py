import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nnworkbench import command_runner, path_setup
from nnworkbench.path_setup import set_paths_command, trainer_directory
from nnworkbench.services.commands import (
    prediction_command,
    preprocessing_command,
    training_command,
    wrapped_command,
)


def test_nnunet_command_generation() -> None:
    assert preprocessing_command(1, "3d_fullres", "ExperimentPlanner") == (
        "nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -pl ExperimentPlanner --verify_dataset_integrity")
    assert preprocessing_command(1, "2d", "ExperimentPlanner", "--np 8") == (
        "nnUNetv2_plan_and_preprocess -d 1 -c 2d -pl ExperimentPlanner --verify_dataset_integrity --np 8")
    assert training_command(1, "custom_plans.json", "2d", "nnUNetTrainer", "all") == (
        "nnUNetv2_train 1 2d all -tr nnUNetTrainer -p custom_plans")
    assert training_command(1, "nnUNetPlans.json", "3d_fullres", "nnUNetTrainer", "0", "--npz") == (
        "nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --npz")
    assert prediction_command(1, "/input path", "/output path", "nnUNetPlans", "3d_fullres") == (
        "nnUNetv2_predict -i '/input path' -o '/output path' -d 1 -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainer -f 0 1 2 3 4 -npp 3 -nps 3")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", "--save_probabilities") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr nnUNetTrainer -f 0 1 2 3 4 -npp 3 -nps 3 --save_probabilities")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", trainer="CustomTrainer") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr CustomTrainer -f 0 1 2 3 4 -npp 3 -nps 3")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", folds="0 2") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr nnUNetTrainer -f 0 2 -npp 3 -nps 3")
    assert wrapped_command("nnUNetv2_train 1 2d 0") == "nnworkbench-run nnUNetv2_train 1 2d 0"


def test_bundled_trainer_directory_exists() -> None:
    assert (trainer_directory() / "nnUNetTrainer_betterIgnoreSampling.py").is_file()


def test_path_setup_command() -> None:
    assert set_paths_command("/raw path", "/preprocessed", "/results") == (
        "nnworkbench-set-paths --raw '/raw path' --preprocessed /preprocessed --results /results"
    )


def test_saved_path_configuration_includes_trainers(tmp_path, monkeypatch) -> None:
    configuration = tmp_path / "nnworkbench" / "paths.json"
    monkeypatch.setattr(path_setup, "configuration_path", lambda: configuration)

    path_setup.save_paths("/raw", "/preprocessed", "/results")

    assert path_setup.configured_paths() == {
        "nnUNet_raw": "/raw",
        "nnUNet_preprocessed": "/preprocessed",
        "nnUNet_results": "/results",
        "nnUNet_extTrainer": str(trainer_directory()),
    }


def test_command_runner_applies_saved_paths(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Result:
        returncode = 7

    monkeypatch.setattr(command_runner, "configured_paths", lambda: {"nnUNet_raw": "/raw"})
    monkeypatch.setattr(command_runner.subprocess, "run", lambda command, **kwargs: (
        captured.update(command=command, **kwargs) or Result()))

    monkeypatch.setattr(command_runner, "find_executable", lambda name: f"/env/bin/{name}")
    assert command_runner.run_with_configured_paths(["nnUNetv2_train", "1", "2d", "0"]) == 7
    assert captured["command"] == ["/env/bin/nnUNetv2_train", "1", "2d", "0"]
    assert captured["env"]["nnUNet_raw"] == "/raw"


def test_command_runner_builtin_tools(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []
    raw = tmp_path / "nnUNet_raw"; raw.mkdir()
    dataset = raw / "Dataset003_MarrowCells"; dataset.mkdir()
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {"nnUNet_raw": str(raw)})
    monkeypatch.setattr(command_runner, "analyse_label_distribution", lambda directory, workers, **kwargs: (
        calls.append(("analysis", (directory, workers, kwargs))) or {"files_analyzed": 2}
    ))

    assert command_runner._run_builtin_command(["compute-statistics", "-d", "3", "--workers", "3"]) == 0
    assert calls == [
        ("analysis", (dataset, 3, {"progress": print})),
    ]


def test_stack_images_command(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    input_directory = tmp_path / "slices"; input_directory.mkdir()
    output = tmp_path / "volume.nii.gz"
    monkeypatch.setattr(
        command_runner,
        "stack_image_folder",
        lambda directory, **kwargs: captured.update(directory=directory, **kwargs) or output,
    )

    assert command_runner._run_builtin_command([
        "stack-images", str(input_directory), "--file-type", ".png", "-o", str(output),
        "--workers", "2", "--scale-factor", "2", "--spacing", "0.5", "0.5", "1.0", "--overwrite",
    ]) == 0
    assert captured == {
        "directory": input_directory,
        "file_type": ".png",
        "output_path": output,
        "workers": 2,
        "scale_factor": 2,
        "spacing": (0.5, 0.5, 1.0),
        "overwrite": True,
        "quiet": True,
        "progress": print,
    }


def test_reduce_checkpoints_command(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    model_directory = tmp_path / "trained_model"; model_directory.mkdir()
    fold = model_directory / "fold_0"; fold.mkdir()
    monkeypatch.setattr(
        command_runner,
        "reduce_model_for_sharing",
        lambda directory, **kwargs: captured.update(directory=directory, **kwargs) or [fold],
    )

    assert command_runner._run_builtin_command([
        "reduce-checkpoints", str(model_directory), "--reset-trainer-name", "--yes",
    ]) == 0
    assert captured == {
        "directory": model_directory,
        "reset_trainer_name": True,
        "progress": print,
    }


def test_inspect_command(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        command_runner,
        "run_dataset_inspection",
        lambda dataset, index, case: captured.update(dataset=dataset, index=index, case=case) or 0,
    )

    assert command_runner._run_builtin_command(["inspect", "-d", "42", "-i", "12"]) == 0
    assert captured == {"dataset": 42, "index": 12, "case": ""}
    assert command_runner._run_builtin_command(["inspect", "-d", "42"]) == 0
    assert captured == {"dataset": 42, "index": 1, "case": ""}


def test_dataset_inspection_opens_selected_image(monkeypatch, tmp_path) -> None:
    from nnworkbench.services import napari_inspector

    images = tmp_path / "imagesTr"; images.mkdir()
    for case in ("case_001", "case_002", "case_003"):
        (images / f"{case}_0000.nii.gz").touch()
    opened: list[Path] = []
    monkeypatch.setattr(command_runner, "raw_dataset_directory", lambda _dataset: tmp_path)
    monkeypatch.setattr(napari_inspector, "open_case", lambda image: opened.append(image))

    assert command_runner.run_dataset_inspection(42, 2) == 0
    assert opened == [images / "case_002_0000.nii.gz"]
    with pytest.raises(ValueError, match="between 1 and 3"):
        command_runner.run_dataset_inspection(42, 4)


def test_find_executable_looks_next_to_python_when_not_on_path(tmp_path, monkeypatch) -> None:
    # e.g. started from the macOS Dock or a Windows shortcut: the environment's scripts are not on PATH
    scripts = tmp_path / "bin"; scripts.mkdir()
    tool = _fake_program(scripts, "nnUNetv2_train")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(command_runner.sys, "executable", str(scripts / "python"))
    assert _same_file(command_runner.find_executable("nnUNetv2_train"), tool)
    assert command_runner.find_executable("missing_tool") == "missing_tool"


def _same_file(found: str, expected: Path) -> bool:
    # on Windows, shutil.which appends the extension as PATHEXT spells it (.EXE): compare case-insensitively there
    return os.path.normcase(found) == os.path.normcase(str(expected))


def _fake_program(folder: Path, name: str) -> Path:
    """An executable stub that shutil.which finds: a script on POSIX, an .exe name on Windows (PATHEXT)."""
    tool = folder / (f"{name}.exe" if os.name == "nt" else name)
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    return tool


def test_find_executable_prefers_this_environment_over_path(tmp_path, monkeypatch) -> None:
    # another environment's nnU-Net earlier on PATH (e.g. conda base): the one next to this Python wins
    own, other = tmp_path / "env" / "bin", tmp_path / "base" / "bin"
    own.mkdir(parents=True); other.mkdir(parents=True)
    mine, foreign = _fake_program(own, "nnUNetv2_train"), _fake_program(other, "nnUNetv2_train")
    only_on_path = _fake_program(other, "some_tool")
    monkeypatch.setenv("PATH", str(other))
    monkeypatch.setattr(command_runner.sys, "executable", str(own / "python"))
    found = command_runner.find_executable("nnUNetv2_train")
    assert _same_file(found, mine) and not _same_file(found, foreign)
    assert _same_file(command_runner.find_executable("some_tool"), only_on_path)  # still found on PATH


def test_training_guard_recognises_other_spellings_on_windows() -> None:
    # Windows starts nnU-Net's training for all of these: the --yes guard must see them too
    for spelling in ("nnUNetv2_train", "nnunetv2_train", "nnUNetv2_train.exe", r"C:\env\Scripts\NNUNETV2_TRAIN.EXE"):
        assert command_runner.is_program(spelling, "nnUNetv2_train", windows=True), spelling
    assert not command_runner.is_program("nnUNetv2_train_other", "nnUNetv2_train", windows=True)
    assert command_runner.is_program("/env/bin/nnUNetv2_train", "nnUNetv2_train", windows=False)
    assert not command_runner.is_program("nnunetv2_train", "nnUNetv2_train", windows=False)  # case matters there


def test_windows_quoting_and_argument_splitting(monkeypatch) -> None:
    from nnworkbench.services import commands

    monkeypatch.setattr(commands, "WINDOWS", True)
    assert commands.split_arguments(r'-chk "C:\models\best ckpt.pth" -o D:\out') == [
        "-chk", r"C:\models\best ckpt.pth", "-o", r"D:\out"]
    # PowerShell habits and cmd-style quotes inside an option: one argument each, backslashes kept
    assert commands.split_arguments(r"""-chk 'checkpoint best.pth' -pretrained_weights="C:\My Folder\w.pth" """) == [
        "-chk", "checkpoint best.pth", r"-pretrained_weights=C:\My Folder\w.pth"]
    internal = commands.prediction_command(1, r"C:\My data", r"D:\out", "nnUNetPlans", "3d_fullres")
    # single quotes: read alike by PowerShell and bash (cmd is not supported)
    assert commands.wrapped_command(internal).startswith(r"nnworkbench-run nnUNetv2_predict -i 'C:\My data' -o 'D:\out'")
    assert commands.shell_quote(r"C:\Users\Ana\R&D") == r"'C:\Users\Ana\R&D'"
    assert commands.shell_quote("Ana's data") == "'Ana''s data'"  # PowerShell doubles it


def test_status_command_reports_stages_and_problems(tmp_path, monkeypatch, capsys) -> None:
    import json

    raw = tmp_path / "raw" / "Dataset012_Cells"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    for case in ("a", "b"):
        (raw / "imagesTr" / f"{case}_0000.nii.gz").touch()
    (raw / "labelsTr" / "a.nii.gz").touch()  # b has no label
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "light"}, "labels": {"background": 0, "cell": 1, "ignore": 2},
        "numTraining": 2, "file_ending": ".nii.gz"}), encoding="utf-8")
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {
        "nnUNet_raw": str(tmp_path / "raw"), "nnUNet_preprocessed": "", "nnUNet_results": ""})
    from nnworkbench import status_report
    from nnworkbench.services.device_probe import DeviceInfo

    monkeypatch.setattr(status_report, "probe_device", lambda: DeviceInfo("cuda", "2.8.0", "12.8", (("RTX 4090", 24.0),)))

    assert command_runner._run_builtin_command(["status", "-d", "12"]) == 0
    text = capsys.readouterr().out
    assert "Dataset012_Cells" in text and "sparse labels (ignore = 2)" in text
    assert "[fix]" in text and "1 image without label" in text  # the problem is listed below the stages
    assert text.isascii() or set(text) - set(map(chr, range(128))) <= {"·"}  # prints in every console

    assert command_runner._run_builtin_command(["status", "-d", "12", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["stages"]["data"]["state"] == "error" and status["stages"]["preprocess"]["state"] == "none"
    assert status["cases"] == 2 and status["models"] == []
    assert status["device"]["kind"] == "cuda" and status["device"]["gpus"] == [{"name": "RTX 4090", "memory_gb": 24.0}]
    assert command_runner._run_builtin_command(["status", "-d", "12", "--json", "--no-device"]) == 0
    assert json.loads(capsys.readouterr().out)["device"] is None


def _roots(tmp_path, monkeypatch) -> None:
    for name in ("raw", "preprocessed", "results"):
        (tmp_path / name).mkdir(exist_ok=True)
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {
        f"nnUNet_{name}": str(tmp_path / name) for name in ("raw", "preprocessed", "results")})


def test_create_dataset_and_new_iteration_commands(tmp_path, monkeypatch, capsys) -> None:
    import json

    _roots(tmp_path, monkeypatch)
    assert command_runner._run_builtin_command(["create-dataset", "-d", "42", "--name", "Cell Segmentation",
                                                "--labels", "cell,nucleus", "--ignore",
                                                "--channels", "fluorescence", "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    folder = tmp_path / "raw" / "Dataset042_Cell_Segmentation"
    assert created["path"] == str(folder) and (folder / "imagesTr").is_dir() and (folder / "labelsTr").is_dir()
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["labels"] == {"background": 0, "cell": 1, "nucleus": 2, "ignore": 3}
    assert manifest["channel_names"] == {"0": "fluorescence"} and manifest["file_ending"] == ".nii.gz"
    with pytest.raises(ValueError, match="already exists"):
        command_runner._run_builtin_command(["create-dataset", "-d", "42", "--name", "Other", "--labels", "a"])
    with pytest.raises(ValueError, match="--ignore"):
        command_runner._run_builtin_command(["create-dataset", "-d", "7", "--name", "X", "--labels", "a,ignore"])

    (folder / "labelsTr" / "c.nii.gz").touch()
    assert command_runner._run_builtin_command(["new-iteration", "-d", "42", "--json"]) == 0
    iteration = json.loads(capsys.readouterr().out)
    assert iteration["dataset"] == "Dataset043_Cell_Segmentation_iter1"
    assert (tmp_path / "raw" / iteration["dataset"] / "labelsTr" / "c.nii.gz").is_file()


def test_validation_command_sorts_cases_weakest_first(tmp_path, monkeypatch, capsys) -> None:
    import json

    _roots(tmp_path, monkeypatch)
    raw = tmp_path / "raw" / "Dataset042_Cells"; (raw / "imagesTr").mkdir(parents=True)
    (raw / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "cell": 1, "nucleus": 2}}),
                                      encoding="utf-8")
    model = tmp_path / "results" / "Dataset042_Cells" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    for fold, cases in (("0", {"a": (0.9, 0.8), "b": (0.5, 0.7)}), ("1", {"c": (0.95, 0.4)})):
        validation = model / f"fold_{fold}" / "validation"; validation.mkdir(parents=True)
        (model / f"fold_{fold}" / "checkpoint_final.pth").touch()
        (model / f"fold_{fold}" / "checkpoint_best.pth").touch()
        per_case = [{"reference_file": f"/gt/{case}.nii.gz", "prediction_file": f"/val/{case}.nii.gz",
                     "metrics": {"1": {"Dice": cell}, "2": {"Dice": nucleus}}} for case, (cell, nucleus) in cases.items()]
        dice = sum(sum(values) / 2 for values in cases.values()) / len(cases)
        (validation / "summary.json").write_text(json.dumps({
            "foreground_mean": {"Dice": dice}, "mean": {"1": {"Dice": 0.8}, "2": {"Dice": 0.6}},
            "metric_per_case": per_case}), encoding="utf-8")

    assert command_runner._run_builtin_command(["validation", "-d", "42", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    [entry] = report["models"]
    assert entry["model"] == "nnUNetTrainer__nnUNetPlans__3d_fullres" and set(entry["folds"]) == {"0", "1"}
    assert entry["classes"] == {"cell": pytest.approx(0.8), "nucleus": pytest.approx(0.6)}
    assert [case["case"] for case in entry["cases"]] == ["b", "c", "a"]  # weakest first
    assert entry["cases"][0]["weakest_class"] == "cell" and entry["cases"][1]["weakest_class"] == "nucleus"

    assert command_runner._run_builtin_command(["validation", "-d", "42", "--cases", "1"]) == 0
    text = capsys.readouterr().out
    assert "weakest cases (1 of 3)" in text and "b " in text

    from nnworkbench.services import napari_inspector

    opened = []
    monkeypatch.setattr(napari_inspector, "open_validation_prediction",
                        lambda prediction, raw: opened.append((prediction.parent.parent.name, prediction.name, raw.name)))
    (model / "fold_0" / "validation" / "b.nii.gz").touch()  # summary.json has paths of another machine
    assert command_runner._run_builtin_command(["validation", "-d", "42", "--open", "b"]) == 0
    assert opened == [("fold_0", "b.nii.gz", "Dataset042_Cells")]
    with pytest.raises(ValueError, match="No validation result"):
        command_runner._run_builtin_command(["validation", "-d", "42", "--open", "x"])


def test_annotate_and_inspect_open_the_case_by_name(tmp_path, monkeypatch) -> None:
    from nnworkbench.services import napari_inspector

    _roots(tmp_path, monkeypatch)
    images = tmp_path / "raw" / "Dataset042_Cells" / "imagesTr"; images.mkdir(parents=True)
    for case in ("cells_001", "cells_002"):
        (images / f"{case}_0000.nii.gz").touch()
    opened = []
    monkeypatch.setattr(napari_inspector, "open_annotation", lambda path: opened.append(("annotate", path.name)))
    monkeypatch.setattr(napari_inspector, "open_case", lambda path: opened.append(("inspect", path.name)))

    assert command_runner._run_builtin_command(["annotate", "-d", "42", "--case", "cells_002"]) == 0
    assert command_runner._run_builtin_command(["inspect", "-d", "42", "--case", "cells_001"]) == 0
    assert command_runner._run_builtin_command(["annotate", "-d", "42", "-i", "1"]) == 0
    assert opened == [("annotate", "cells_002_0000.nii.gz"), ("inspect", "cells_001_0000.nii.gz"),
                      ("annotate", "cells_001_0000.nii.gz")]
    with pytest.raises(ValueError, match="No case 'cells_009'"):
        command_runner._run_builtin_command(["annotate", "-d", "42", "--case", "cells_009"])


def test_json_output_for_statistics_and_reduction_keeps_progress_on_stderr(tmp_path, monkeypatch, capsys) -> None:
    import json

    _roots(tmp_path, monkeypatch)
    (tmp_path / "raw" / "Dataset042_Cells" / "labelsTr").mkdir(parents=True)

    def analyse(directory, workers, progress):
        progress("Analyzed label 1/1: a.nii.gz")
        return {"files_analyzed": 1, "classes": [{"id": 1, "name": "cell"}]}

    monkeypatch.setattr(command_runner, "analyse_label_distribution", analyse)
    assert command_runner._run_builtin_command(["compute-statistics", "-d", "42", "--json"]) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["dataset"] == "Dataset042_Cells" and result["files_analyzed"] == 1
    assert "Analyzed label" in output.err  # progress does not break the JSON

    def reduce(model, reset_trainer_name, progress):
        progress("Reduced fold_0")
        return [model / "fold_0", model / "fold_1"]

    monkeypatch.setattr(command_runner, "reduce_model_for_sharing", reduce)
    monkeypatch.setattr(command_runner, "reduction_plan", lambda model, reset_trainer_name: [])
    assert command_runner._run_builtin_command(["reduce-checkpoints", str(tmp_path / "model"), "--json", "--yes"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["folds"] == ["fold_0", "fold_1"] and result["schema_version"] == 1


def test_errors_are_json_with_the_json_flag(tmp_path, monkeypatch, capfd) -> None:
    import json
    import sys

    _roots(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["nnworkbench-run", "validation", "-d", "99", "--json"])
    with pytest.raises(SystemExit) as stop:
        command_runner.main()
    assert stop.value.code == 1
    assert "No dataset with ID 99" in json.loads(capfd.readouterr().out)["error"]  # stdout holds only JSON


def test_get_paths_json(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    monkeypatch.setattr(path_setup, "configuration_path", lambda: tmp_path / "paths.json")
    path_setup.save_paths("/r", "/p", "/s")
    monkeypatch.setattr(sys, "argv", ["nnworkbench-get-paths", "--json"])
    path_setup.get_paths_main()
    paths = json.loads(capsys.readouterr().out)["paths"]
    assert paths["nnUNet_raw"] == "/r" and paths["nnUNet_results"] == "/s"


def test_reduce_checkpoints_needs_yes_and_dry_run_changes_nothing(tmp_path, capsys) -> None:
    import json

    model = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (model / "fold_0" / "validation").mkdir(parents=True)
    (model / "fold_0" / "checkpoint_best.pth").touch()

    assert command_runner._run_builtin_command(["reduce-checkpoints", str(model), "--dry-run", "--json"]) == 0
    actions = json.loads(capsys.readouterr().out)["actions"]
    assert {(item["action"], Path(item["path"]).name) for item in actions} == {
        ("delete", "checkpoint_best.pth"), ("delete folder", "validation")}
    with pytest.raises(ValueError, match="--yes"):
        command_runner._run_builtin_command(["reduce-checkpoints", str(model)])
    assert (model / "fold_0" / "checkpoint_best.pth").is_file() and (model / "fold_0" / "validation").is_dir()


def test_training_an_existing_fold_again_needs_yes(tmp_path, monkeypatch, capsys) -> None:
    _roots(tmp_path, monkeypatch)
    fold = tmp_path / "results" / "Dataset042_Cells" / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "training_log_1.txt").touch()
    os.utime(fold / "training_log_1.txt", (time.time() - 3600,) * 2)  # stopped an hour ago, not training
    command = ["nnUNetv2_train", "42", "3d_fullres", "0", "-p", "nnUNetResEncUNetLPlans"]
    calls = []
    monkeypatch.setattr(command_runner, "find_executable", lambda name: name)
    monkeypatch.setattr(command_runner.shutil, "which", lambda name: name)
    monkeypatch.setattr(command_runner.subprocess, "run",
                        lambda arguments, **_kwargs: calls.append(arguments) or SimpleNamespace(returncode=0))
    monkeypatch.delenv(command_runner.YES_ENVIRONMENT, raising=False)

    with pytest.raises(ValueError, match="already has a training"):
        command_runner.run_with_configured_paths(command)
    assert calls == []
    assert command_runner.run_with_configured_paths([*command, "--dry-run"]) == 0
    assert "replaces its results (needs --yes)" in capsys.readouterr().out and calls == []
    assert command_runner.run_with_configured_paths([*command, "--c"]) == 0  # continuing is fine
    assert command_runner.run_with_configured_paths([*command, "--yes"]) == 0
    assert calls[-1] == command  # --yes is not passed on to nnU-Net
    monkeypatch.setenv(command_runner.YES_ENVIRONMENT, "1")  # the GUI asks the user itself
    assert command_runner.run_with_configured_paths(command) == 0
    assert command_runner.run_with_configured_paths(["nnUNetv2_train", "42", "3d_fullres", "1"]) == 0  # new fold


def test_describe_and_create_dataset_add_the_description(tmp_path, monkeypatch, capsys) -> None:
    import json

    _roots(tmp_path, monkeypatch)
    assert command_runner._run_builtin_command(["create-dataset", "-d", "42", "--name", "Cells", "--labels", "cell",
                                                "--add-description", "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert Path(created["description"]).name == "dataset-description.md"
    assert command_runner._run_builtin_command(["create-dataset", "-d", "43", "--name", "Liver", "--labels", "liver"]) == 0
    capsys.readouterr()
    assert command_runner._run_builtin_command(["describe", "-d", "43", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True
    assert command_runner._run_builtin_command(["describe", "-d", "43", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is False  # never replaced


@pytest.mark.parametrize("cpus, expected", [(16, 3), (3, 3), (2, 2), (1, 1), (None, 1)])
def test_prediction_processes_never_exceed_the_cpu_cores(monkeypatch, cpus: int | None, expected: int) -> None:
    """The GUI's -npp/-nps fields cannot go beyond the cores; `suggest` gives the same number (a CI runner has 2)."""
    from nnworkbench.services import defaults

    monkeypatch.setattr(defaults.os, "cpu_count", lambda: cpus)
    assert defaults.prediction_processes() == expected
