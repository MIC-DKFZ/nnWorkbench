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
        "nnUNetv2_predict -i '/input path' -o '/output path' -d 1 -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainer -f 0 1 2 3 4")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", "--save_probabilities") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr nnUNetTrainer -f 0 1 2 3 4 --save_probabilities")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", trainer="CustomTrainer") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr CustomTrainer -f 0 1 2 3 4")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", folds="0 2") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans -tr nnUNetTrainer -f 0 2")
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


def test_command_runner_builtin_tools(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []
    raw = tmp_path / "nnUNet_raw"; raw.mkdir()
    dataset = raw / "Dataset003_MarrowCells"; dataset.mkdir()
    monkeypatch.setattr(command_runner, "configured_paths", lambda: {"nnUNet_raw": str(raw)})
    monkeypatch.setattr(command_runner, "analyse_label_distribution", lambda directory, workers, **kwargs: (
        calls.append(("analysis", (directory, workers, kwargs))) or {"files_analyzed": 2}
    ))
    monkeypatch.setattr(command_runner, "convert_annotations", lambda directory, workers, **kwargs: (
        calls.append(("conversion", (directory, workers, kwargs))) or type(
            "Result", (), {"converted": (tmp_path / "labelsTr" / "case.nii.gz",), "warnings": ()})()
    ))

    assert command_runner._run_builtin_command(["label-analysis", "-d", "3", "--workers", "3"]) == 0
    assert command_runner._run_builtin_command(["convert-annotations", "-d", "3", "--workers", "2"]) == 0
    assert calls == [
        ("analysis", (dataset, 3, {"progress": print})),
        ("conversion", (dataset, 2, {"progress": print})),
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
    fold = model_directory / "fold_0"
    monkeypatch.setattr(
        command_runner,
        "reduce_model_for_sharing",
        lambda directory, **kwargs: captured.update(directory=directory, **kwargs) or [fold],
    )

    assert command_runner._run_builtin_command([
        "reduce-checkpoints", str(model_directory), "--reset-trainer-name",
    ]) == 0
    assert captured == {
        "directory": model_directory,
        "reset_trainer_name": True,
        "progress": print,
    }
