import json
from pathlib import Path

from nnworkbench.workflow import accept_label, initialise, status


CONFIG = {"data_root": "data", "dataset_id": "Dataset001_nnWorkbench", "modalities": {"0": "CT"},
          "labels": {"background": 0, "target": 1}}


def test_initialise_and_accept_label(tmp_path: Path):
    root = initialise(CONFIG, tmp_path)
    assert json.loads((root / "dataset.json").read_text())["numTraining"] == 0
    (root / "imagesTr" / "case_001_0000.nii.gz").touch()
    candidate = tmp_path / "approved.nii.gz"
    candidate.touch()
    accept_label(CONFIG, tmp_path, "case_001", candidate)
    assert status(CONFIG, tmp_path) == {"images": 1, "accepted_labels": 1, "pending": 0}
