from pathlib import Path

from nnworkbench.services.dataset_contents import (
    json_object,
    json_value,
    model_folds,
    model_trainer_plan_configuration,
    trained_models,
)
from nnworkbench.services.dataset_json import (
    definition_errors,
    file_ending_error,
    manifest_source,
    num_training_error,
    write_manifest,
)


def test_model_discovery(tmp_path: Path) -> None:
    dataset = tmp_path / "Dataset001_Test"
    (dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0").mkdir(parents=True)
    (dataset / "dataset.json").touch()

    assert [str(path) for path in trained_models(dataset)] == ["nnUNetTrainer__nnUNetPlans__3d_fullres"]
    assert model_trainer_plan_configuration(dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres") == (
        "nnUNetTrainer", "nnUNetPlans", "3d_fullres")


def test_model_folds_and_summary_json(tmp_path: Path) -> None:
    model = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (model / "fold_0" / "validation").mkdir(parents=True)
    (model / "fold_1").mkdir()
    (model / "fold_0" / "validation" / "summary.json").write_text(
        '{"foreground_mean": {"Dice": 0.91}}', encoding="utf-8")

    assert [path.name for path in model_folds(model)] == ["fold_0", "fold_1"]
    assert json_object(model / "fold_1" / "validation" / "summary.json") is None
    assert json_object(model / "fold_0" / "validation" / "summary.json") == {"foreground_mean": {"Dice": 0.91}}


def test_json_value_reads_splits_array(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text('[{"train": ["case_001"], "val": ["case_002"]}]', encoding="utf-8")

    assert json_value(splits) == [{"train": ["case_001"], "val": ["case_002"]}]
    assert json_object(splits) is None


def test_manifest_source_prefers_raw_then_preprocessed_then_results(tmp_path: Path) -> None:
    raw, preprocessed, results = (tmp_path / name for name in ("raw", "preprocessed", "results"))
    for directory in (raw, preprocessed, results):
        directory.mkdir()
    (preprocessed / "dataset.json").write_text("{}", encoding="utf-8")
    (results / "dataset.json").write_text("{}", encoding="utf-8")

    assert manifest_source(raw, preprocessed, results) == ("preprocessed", preprocessed)
    (raw / "dataset.json").write_text("{}", encoding="utf-8")
    assert manifest_source(raw, preprocessed, results) == ("raw", raw)
    (raw / "dataset.json").unlink(); (preprocessed / "dataset.json").unlink()
    assert manifest_source(raw, preprocessed, results) == ("results", results)


def test_write_manifest_creates_missing_dataset_json(tmp_path: Path) -> None:
    manifest = {"channel_names": {"0": "CT"}, "labels": {"background": 0}, "file_ending": ".nii.gz"}

    write_manifest(tmp_path, manifest)

    assert json_object(tmp_path / "dataset.json") == manifest


def test_num_training_error_matches_nnunet_integrity_check() -> None:
    assert num_training_error({"numTraining": 116}, 116) is None
    assert "21" in num_training_error({"numTraining": 21}, 116) and "116" in num_training_error({"numTraining": 21}, 116)
    assert num_training_error({}, 3) is not None
    assert num_training_error({"numTraining": True}, 1) is not None


def test_definition_errors_reports_invalid_labels() -> None:
    errors = definition_errors({"channel_names": {"0": "CT"}, "labels": {"": 0}})

    assert errors == ("Labels must have continuous numeric IDs and a name for every label.",)


def test_file_ending_error() -> None:
    assert file_ending_error({"file_ending": ".nii.gz"}) is None
    assert file_ending_error({}) is not None
    assert file_ending_error({"file_ending": "nii.gz"}) is not None


def test_missing_or_broken_raw_dataset_json_is_an_error(tmp_path: Path) -> None:
    from nnworkbench.models.dataset import DatasetRecord
    from nnworkbench.ui.dataset_state import load_state

    raw = tmp_path / "Dataset009_Empty"
    raw.mkdir()
    state = load_state(DatasetRecord(9, "Empty", raw_path=raw))
    assert state.data_state()[0] == "error" and "missing" in state.manifest_errors[0]
    (raw / "dataset.json").write_text("{ broken", encoding="utf-8")
    state = load_state(DatasetRecord(9, "Empty", raw_path=raw))
    assert "cannot be read" in state.manifest_errors[0]


def test_region_based_labels_are_valid() -> None:
    manifest = {"channel_names": {"0": "T1"}, "file_ending": ".nii.gz", "regions_class_order": [1, 2, 3],
                "labels": {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": [3]}}
    assert definition_errors(manifest) == ()
