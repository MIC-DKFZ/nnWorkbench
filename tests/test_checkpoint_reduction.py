from pathlib import Path

import pytest
import torch

from nnworkbench.services.checkpoint_reduction import reduce_checkpoint, reduce_model_for_sharing


def _write_checkpoint(path: Path, **fields: object) -> None:
    torch.save({"optimizer_state": {"foo": 1}, "network_weights": {"bar": 2}, **fields}, path)


def test_reduce_checkpoint_drops_optimizer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_final.pth"
    _write_checkpoint(checkpoint, trainer_name="nnUNetTrainerCustom")

    reduce_checkpoint(checkpoint)

    saved = torch.load(checkpoint, weights_only=False)
    assert "optimizer_state" not in saved
    assert saved["network_weights"] == {"bar": 2}
    assert saved["trainer_name"] == "nnUNetTrainerCustom"


def test_reduce_checkpoint_can_reset_trainer_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_final.pth"
    _write_checkpoint(checkpoint, trainer_name="nnUNetTrainerCustom")

    reduce_checkpoint(checkpoint, reset_trainer_name=True)

    assert torch.load(checkpoint, weights_only=False)["trainer_name"] == "nnUNetTrainer"


def test_reduce_model_for_sharing_processes_every_fold(tmp_path: Path) -> None:
    model = tmp_path / "Dataset001__nnUNetTrainer__nnUNetPlans__3d_fullres"
    for fold_name in ("fold_0", "fold_1"):
        fold = model / fold_name; fold.mkdir(parents=True)
        _write_checkpoint(fold / "checkpoint_final.pth")
        _write_checkpoint(fold / "checkpoint_best.pth")
        validation = fold / "validation"; validation.mkdir()
        (validation / "summary.json").write_text("{}")

    folds = reduce_model_for_sharing(model)

    assert [fold.name for fold in folds] == ["fold_0", "fold_1"]
    for fold_name in ("fold_0", "fold_1"):
        fold = model / fold_name
        assert "optimizer_state" not in torch.load(fold / "checkpoint_final.pth", weights_only=False)
        assert not (fold / "checkpoint_best.pth").exists()
        assert not (fold / "validation").exists()


def test_reduce_model_for_sharing_requires_folds(tmp_path: Path) -> None:
    model = tmp_path / "empty_model"; model.mkdir()

    with pytest.raises(ValueError, match="No fold_"):
        reduce_model_for_sharing(model)
