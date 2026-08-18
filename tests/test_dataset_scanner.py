from pathlib import Path

import pytest

from nnworkbench.services.dataset_creator import create_dataset
from nnworkbench.services.dataset_scanner import scan_datasets


def test_scanner_merges_optional_roots(tmp_path: Path):
    raw, preprocessed, results = (tmp_path / name for name in ("raw", "preprocessed", "results"))
    for root in (raw, preprocessed, results): root.mkdir()
    (raw / "Dataset001_Liver").mkdir()
    (preprocessed / "Dataset001_Liver").mkdir()
    (results / "Dataset002_Kidney").mkdir()
    records = scan_datasets(raw, preprocessed, results)
    assert [(record.dataset_id, bool(record.raw_path), bool(record.preprocessed_path), bool(record.results_path))
            for record in records] == [(1, True, True, False), (2, False, False, True)]


def test_creator_rejects_id_present_in_another_root(tmp_path: Path):
    raw, results = tmp_path / "raw", tmp_path / "results"
    raw.mkdir(); results.mkdir(); (results / "Dataset007_Existing").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        create_dataset(raw, [raw, None, results], 7, "New")


def test_creator_creates_raw_structure(tmp_path: Path):
    raw = tmp_path / "raw"; raw.mkdir()
    created = create_dataset(raw, [raw, None, None], 3, "My task")
    assert created.name == "Dataset003_My_task"
    assert all((created / directory).is_dir() for directory in ("imagesTr", "labelsTr", "imagesTs"))
    assert (created / "dataset.json").is_file()
