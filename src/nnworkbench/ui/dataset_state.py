"""Everything the stage pages show about one dataset, read in the background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_contents import json_object
from nnworkbench.services.dataset_json import (
    definition_errors,
    file_ending_error,
    has_regions,
    manifest_source,
    num_training_error,
)
from nnworkbench.services.label_analysis import analysis_path, load_label_analysis
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
    def regions(self) -> bool:
        return has_regions(self.manifest)

    @property
    def names(self) -> dict[int, str]:
        return label_names(self.manifest)

    @property
    def raw_path(self) -> Path | None:
        return self.record.raw_path

    # ------------------------------------------------------------ stage states (tab icons)

    # Each returns (state, hover text). States: "" grey (nothing yet, or not possible yet), "todo" orange ring
    # (ready to run, or running), "warn" orange ! (outdated), "error" red ! (must be fixed), "ok" green ✓.

    def data_problems(self) -> list[str]:
        raw = self.raw
        problems = list(self.manifest_errors)
        if raw.missing_labels:
            problems.append(f"{raw.missing_labels} image{'s' if raw.missing_labels != 1 else ''} without label")
        if not raw.names_valid:
            problems.append("image names need a _0000 channel suffix")
        if not raw.channels_complete:
            problems.append("channels missing")
        return problems

    def data_state(self) -> tuple[str, str]:
        if not self.raw.cases:
            if self.raw_path is None:
                return "", "Data · no raw data on this computer (nnUNet_raw)"
            return "error", "Data · no images in imagesTr yet: add images and labels before preprocessing"
        problems = self.data_problems()
        if problems:
            return "error", "Data · must be fixed before preprocessing: " + "; ".join(problem.rstrip(".")
                                                                                     for problem in problems)
        return "ok", "Data · ready for preprocessing"

    def preprocess_state(self) -> tuple[str, str]:
        preprocessed = [plans for plans in self.plans if plans.preprocessed_time]
        if preprocessed:
            if any(plans.outdated for plans in preprocessed):
                return "warn", "Preprocess · outdated: labelsTr changed after preprocessing"
            return "ok", "Preprocess · preprocessed and up to date"
        if self.raw.cases and not self.data_problems():
            return "todo", "Preprocess · ready to preprocess"
        if self.raw.cases or self.raw_path is not None:
            return "", "Preprocess · fix the problems on Data first"
        return "", "Preprocess · no preprocessed data on this computer"

    def train_state(self) -> tuple[str, str]:
        if any_fold_training(self.models):
            return "todo", "Train · a fold is training"
        finished = sum(len(model.finished_folds) for model in self.models)
        if finished:
            return "ok", f"Train · {finished} fold{'s' if finished != 1 else ''} finished"
        stopped = sum(fold.status == "stopped" for model in self.models for fold in model.folds)
        if stopped:
            return "todo", f"Train · {stopped} fold{'s' if stopped != 1 else ''} stopped, can be continued"
        if any(plans.preprocessed_time for plans in self.plans):
            return "todo", "Train · ready to train (preprocessed data available)"
        return "", "Train · no preprocessed data"

    def validate_state(self) -> tuple[str, str]:
        validated = sum(any(fold.has_validation for fold in model.folds) for model in self.models)
        if validated:
            return "ok", f"Validate · {validated} model{'s' if validated != 1 else ''} with validation results"
        only_all = any(fold.name == "all" for model in self.models for fold in model.folds)
        return "", "Validate · no validation results yet" + (" (fold all has no validation)" if only_all else "")

    def predict_state(self) -> tuple[str, str]:
        usable = sum(1 for model in self.models if model.finished_folds)
        if usable:
            return "ok", f"Predict · {usable} model{'s' if usable != 1 else ''} ready for prediction"
        return "", "Predict · no model with a finished fold yet"


def _analysis_outdated(raw_path: Path | None, analysis: dict | None) -> int:
    if raw_path is None or analysis is None:
        return 0
    try:
        analysed = analysis_path(raw_path).stat().st_mtime
    except OSError:
        return 0
    labels = raw_path / "labelsTr"
    if not labels.is_dir():
        return 0
    return sum(1 for path in labels.iterdir() if path.is_file() and path.stat().st_mtime > analysed)


def load_state(record: DatasetRecord) -> DatasetState:
    source = manifest_source(record.raw_path, record.preprocessed_path, record.results_path)
    manifest = json_object(source[1] / "dataset.json") if source else None
    errors = [*definition_errors(manifest), *filter(None, [file_ending_error(manifest)])] \
        if isinstance(manifest, dict) else []
    raw_json = record.raw_path / "dataset.json" if record.raw_path else None
    if raw_json is not None and not raw_json.is_file():
        errors.insert(0, "dataset.json is missing in nnUNet_raw.")
    elif raw_json is not None and source and source[0] == "raw" and not isinstance(manifest, dict):
        errors.insert(0, "dataset.json cannot be read (no valid JSON object).")
    raw = raw_overview(record.raw_path, manifest)
    if isinstance(manifest, dict) and source and source[0] == "raw" and raw.cases:
        problem = num_training_error(manifest, len(raw.cases))
        if problem:
            errors.append(problem)
    analysis = load_label_analysis(record.raw_path) if record.raw_path else None
    return DatasetState(
        record=record,
        manifest=manifest,
        manifest_kind=source[0] if source else None,
        manifest_errors=tuple(errors),
        raw=raw,
        plans=plans_overview(record.preprocessed_path, record.raw_path),
        models=models_overview(record.results_path),
        analysis=analysis,
        analysis_outdated=_analysis_outdated(record.raw_path, analysis),
        fold_count=splits_fold_count(record.preprocessed_path),
    )
