"""Everything the stage pages show about one dataset, read in the background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_contents import json_object
from nnworkbench.services.dataset_json import definition_errors, manifest_source
from nnworkbench.services.label_analysis import load_label_analysis
from nnworkbench.services.workflow_status import (
    ModelInfo,
    PlansInfo,
    RawOverview,
    any_fold_training,
    ignore_label_id,
    label_names,
    models_overview,
    plans_overview,
    raw_overview,
    splits_fold_count,
)


@dataclass(frozen=True)
class DatasetState:
    record: DatasetRecord
    manifest: dict | None
    manifest_kind: str | None  # "raw", "preprocessed" or "results"
    manifest_errors: tuple[str, ...]
    raw: RawOverview
    plans: tuple[PlansInfo, ...]
    models: tuple[ModelInfo, ...]
    analysis: dict | None
    analysis_outdated: int  # labels changed after the last analysis
    fold_count: int

    @property
    def ignore_id(self) -> int | None:
        return ignore_label_id(self.manifest)

    @property
    def sparse(self) -> bool:
        return self.ignore_id is not None

    @property
    def names(self) -> dict[int, str]:
        return label_names(self.manifest)

    @property
    def raw_path(self) -> Path | None:
        return self.record.raw_path

    # ------------------------------------------------------------ stage states (tab icons)

    def data_state(self) -> str:
        if not self.raw.cases:
            return ""
        return "warn" if self.raw.missing_labels or self.manifest_errors or not self.raw.channels_complete else "ok"

    def preprocess_state(self) -> str:
        preprocessed = [plans for plans in self.plans if plans.preprocessed_time]
        if not preprocessed:
            return ""
        return "warn" if any(plans.outdated for plans in preprocessed) else "ok"

    def train_state(self) -> str:
        if any_fold_training(self.models):
            return "run"
        return "ok" if any(model.finished_folds for model in self.models) else ""

    def validate_state(self) -> str:
        return "ok" if any(fold.has_validation for model in self.models for fold in model.folds) else ""

    def predict_state(self) -> str:
        return ""


def _analysis_outdated(raw_path: Path | None, analysis: dict | None) -> int:
    if raw_path is None or analysis is None:
        return 0
    try:
        analysed = (raw_path / "class_distribution.json").stat().st_mtime
    except OSError:
        return 0
    labels = raw_path / "labelsTr"
    if not labels.is_dir():
        return 0
    return sum(1 for path in labels.iterdir() if path.is_file() and path.stat().st_mtime > analysed)


def load_state(record: DatasetRecord) -> DatasetState:
    source = manifest_source(record.raw_path, record.preprocessed_path, record.results_path)
    manifest = json_object(source[1] / "dataset.json") if source else None
    errors = definition_errors(manifest) if isinstance(manifest, dict) else ()
    analysis = load_label_analysis(record.raw_path) if record.raw_path else None
    return DatasetState(
        record=record,
        manifest=manifest,
        manifest_kind=source[0] if source else None,
        manifest_errors=tuple(errors),
        raw=raw_overview(record.raw_path, manifest),
        plans=plans_overview(record.preprocessed_path, record.raw_path),
        models=models_overview(record.results_path),
        analysis=analysis,
        analysis_outdated=_analysis_outdated(record.raw_path, analysis),
        fold_count=splits_fold_count(record.preprocessed_path),
    )
