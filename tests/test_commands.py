from nnworkbench import command_runner, path_setup
from nnworkbench.path_setup import set_paths_command, trainer_directory
from nnworkbench.services.commands import (
    prediction_command,
    preprocessing_command,
    training_command,
    wrapped_command,
)
from nnworkbench.services.nnunet_discovery import available_trainers


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
        "nnUNetv2_predict -i '/input path' -o '/output path' -d 1 -c 3d_fullres -p nnUNetPlans")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", "--save_probabilities") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans --save_probabilities")
    assert wrapped_command("nnUNetv2_train 1 2d 0") == "nnworkbench-run nnUNetv2_train 1 2d 0"


def test_bundled_annotation_workflow_trainer_is_discoverable() -> None:
    trainers = available_trainers()

    assert "nnUNetTrainer" in trainers
    assert "nnUNetTrainerBetterIgnoreSampling" in trainers
    assert "nnUNetTrainerBetterIgnoreSamplingOnlyMirror01" in trainers
    assert "nnUNetTrainerBetterIgnoreSamplingNoMirroring" in trainers


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

    assert command_runner.run_with_configured_paths(["nnUNetv2_train", "1", "2d", "0"]) == 7
    assert captured["command"] == ["nnUNetv2_train", "1", "2d", "0"]
    assert captured["env"]["nnUNet_raw"] == "/raw"
