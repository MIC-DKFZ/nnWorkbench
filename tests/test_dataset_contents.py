from pathlib import Path

from nnworkbench.services.dataset_contents import (
    FoldTrainingStatus,
    fold_training_status,
    json_object,
    json_value,
    model_folds,
    model_plan_configuration,
    plan_files,
    preprocessed_files,
    raw_files,
    trained_models,
    validation_files,
    validation_summary,
)
from nnworkbench.services.dataset_json import (
    definition_errors,
    image_count_check,
    labels_tr_count,
    manifest_source,
    write_manifest,
)


def test_raw_files_includes_standard_and_optional_annotation_folders(tmp_path: Path) -> None:
    dataset = tmp_path / "Dataset001_Test"
    for folder, filename in (("imagesTr", "case_001_0000.nii.gz"), ("labelsTr", "case_001.nii.gz"),
                             ("annotationsTr", "case_001.json")):
        directory = dataset / folder; directory.mkdir(parents=True)
        (directory / filename).touch()

    contents = raw_files(dataset)

    assert list(contents) == ["imagesTr", "labelsTr", "annotationsTr"]
    assert [path.name for path in contents["imagesTr"]] == ["case_001_0000.nii.gz"]


def test_plan_and_model_discovery(tmp_path: Path) -> None:
    dataset = tmp_path / "Dataset001_Test"
    (dataset / "nnUNetPlans.json").parent.mkdir(parents=True)
    (dataset / "nnUNetPlans.json").touch()
    (dataset / "configuration" / "custom_plans.json").parent.mkdir(exist_ok=True)
    (dataset / "configuration" / "custom_plans.json").touch()
    (dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0").mkdir(parents=True)
    (dataset / "dataset.json").touch()

    assert [str(path) for path in plan_files(dataset)] == ["configuration/custom_plans.json", "nnUNetPlans.json"]
    assert [str(path) for path in preprocessed_files(dataset)] == ["configuration/custom_plans.json", "dataset.json", "nnUNetPlans.json"]
    assert [str(path) for path in trained_models(dataset)] == ["nnUNetTrainer__nnUNetPlans__3d_fullres"]
    assert model_plan_configuration(dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres") == ("nnUNetPlans", "3d_fullres")


def test_model_folds_and_validation_summary(tmp_path: Path) -> None:
    model = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (model / "fold_0" / "validation").mkdir(parents=True)
    (model / "fold_1").mkdir()
    (model / "fold_0" / "validation" / "summary.json").write_text(
        '{"foreground_mean": {"Dice": 0.91}}', encoding="utf-8")

    assert [path.name for path in model_folds(model)] == ["fold_0", "fold_1"]
    (model / "fold_0" / "validation" / "case_001.nii.gz").touch()
    (model / "fold_0" / "validation" / "nested").mkdir()
    (model / "fold_0" / "validation" / "nested" / "case_002.nii.gz").touch()
    assert [str(path) for path in validation_files(model / "fold_0")] == [
        "case_001.nii.gz", "nested/case_002.nii.gz", "summary.json"
    ]
    assert validation_summary(model / "fold_0") == {"foreground_mean": {"Dice": 0.91}}
    assert validation_summary(model / "fold_1") is None
    assert json_object(model / "fold_0" / "validation" / "summary.json") == {"foreground_mean": {"Dice": 0.91}}


def test_fold_training_status_reads_latest_logged_epoch_and_completion(tmp_path: Path) -> None:
    fold = tmp_path / "fold_0"; fold.mkdir()
    log = fold / "training_log_2026_09_02_12_13_06.txt"
    log.write_text(
        "2026-09-02 12:13:26.152147: Epoch 0\n"
        "2026-09-02 14:44:12.000000: Epoch 123\n",
        encoding="utf-8",
    )

    assert fold_training_status(fold) == FoldTrainingStatus(epoch=123, complete=False)
    (fold / "checkpoint_final.pth").touch()
    assert fold_training_status(fold) == FoldTrainingStatus(epoch=123, complete=True)


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


def test_labels_tr_count_counts_only_files(tmp_path: Path) -> None:
    labels = tmp_path / "labelsTr"; labels.mkdir()
    (labels / "case_001.nii.gz").touch(); (labels / "case_002.nii.gz").touch()
    (labels / "nested").mkdir()

    assert labels_tr_count(tmp_path) == 2


def test_image_count_check_compares_channels_and_labels(tmp_path: Path) -> None:
    for directory in (tmp_path / "imagesTr", tmp_path / "labelsTr"):
        directory.mkdir()
    for name in ("case_001_0000.nii.gz", "case_001_0001.nii.gz", "case_002_0000.nii.gz"):
        (tmp_path / "imagesTr" / name).touch()
    for name in ("case_001.nii.gz", "case_002.nii.gz"):
        (tmp_path / "labelsTr" / name).touch()
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "CT", "1": "MR"}, "labels": {"background": 0}}',
        encoding="utf-8",
    )

    check = image_count_check(tmp_path, None, None)

    assert check is not None
    assert (check.images, check.channels, check.labels, check.labels_directory_exists, check.expected_images) == (3, 2, 2, True, 4)
    assert not check.matches


def test_image_count_check_reports_missing_or_empty_label_folder(tmp_path: Path) -> None:
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "CT"}, "labels": {"background": 0}}', encoding="utf-8"
    )

    missing = image_count_check(tmp_path, None, None)
    assert missing is not None and not missing.labels_directory_exists and missing.labels == 0
    (tmp_path / "labelsTr").mkdir()
    empty = image_count_check(tmp_path, None, None)
    assert empty is not None and empty.labels_directory_exists and empty.labels == 0


def test_invalid_definitions_suppress_image_count_check(tmp_path: Path) -> None:
    for directory in (tmp_path / "imagesTr", tmp_path / "labelsTr"):
        directory.mkdir()
    (tmp_path / "imagesTr" / "case_001_0000.nii.gz").touch()
    (tmp_path / "labelsTr" / "case_001.nii.gz").touch()
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": ""}, "labels": {"background": 0}}', encoding="utf-8"
    )

    errors = definition_errors({"channel_names": {"0": ""}, "labels": {"background": 0}})

    assert errors == ("Channels must have continuous numeric IDs and a name for every channel.",)
    assert image_count_check(tmp_path, None, None) is None


def test_definition_errors_reports_invalid_labels() -> None:
    errors = definition_errors({"channel_names": {"0": "CT"}, "labels": {"": 0}})

    assert errors == ("Labels must have continuous numeric IDs and a name for every label.",)
