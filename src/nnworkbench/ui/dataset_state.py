"""Everything the stage pages show about one dataset, read in the background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.path_setup import NON_ASCII_HINT, unusable_for_nnunet
from nnworkbench.services.dataset_contents import json_object
from nnworkbench.services.dataset_json import (
    definition_errors,
    file_ending_error,
    file_ending_mismatch,
    has_regions,
    listed_cases,
    manifest_source,
    num_training_error,
)
from nnworkbench.services.image_files import has_image_suffix
from nnworkbench.services.label_analysis import analysis_path, load_label_analysis
from nnworkbench.services.workflow_status import (
    UNSUPPORTED_IMAGE_SUFFIXES,
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
    def inherits_from(self) -> str:
        """The dataset this one is a later iteration of (New iteration writes it into dataset.json), or ""."""
        value = self.manifest.get("inherits_from") if isinstance(self.manifest, dict) else None
        return value if isinstance(value, str) else ""

    @property
    def raw_path(self) -> Path | None:
        return self.record.raw_path

    # ------------------------------------------------------------ stage states (tab icons)

    # Each returns (state, hover text). States: "" grey (nothing yet, or not possible yet), "todo" orange ring
    # (ready to run, or running), "warn" orange ! (outdated), "error" red ! (must be fixed), "ok" green ✓.

    def data_problems(self) -> list[str]:
        raw = self.raw
        problems = list(self.manifest_errors)
        if self.record.id_problem[1]:
            problems.append(self.record.id_problem[1])
        unusable = [str(path) for path in (self.record.raw_path, self.record.preprocessed_path,
                                           self.record.results_path) if path is not None and unusable_for_nnunet(path)]
        if unusable:
            problems.append(f"{NON_ASCII_HINT}: {', '.join(unusable)}")
        if raw.missing_labels:
            problems.append(f"{raw.missing_labels} image{'s' if raw.missing_labels != 1 else ''} without label")
        if not raw.names_valid:
            problems.append("image names need a _0000 channel suffix")
        if raw.unlisted_images:
            problems.append(f"{len(raw.unlisted_images)} image{'s' if len(raw.unlisted_images) != 1 else ''} in "
                            f"imagesTr (e.g. {raw.unlisted_images[0]}) not in the \"dataset\" file list of "
                            "dataset.json: nnU-Net uses only the listed files. Add them to the list, or remove them")
        if raw.missing_files:
            problems.append(f"{len(raw.missing_files)} file{'s' if len(raw.missing_files) != 1 else ''} listed in "
                            f"dataset.json (\"dataset\") not found (e.g. {raw.missing_files[0]})")
        if raw.channel_problem:
            problems.append(raw.channel_problem)
        elif not raw.channels_complete:
            problems.append("channels missing")
        extra = [] if raw.file_list else sorted(
            {channel for case in raw.cases for channel in case.extra_channels(raw.required_suffixes)})
        if extra and not raw.channel_problem:
            problems.append(f"image files with channel {', '.join('_' + channel for channel in extra)}, which "
                            f"dataset.json does not list ({raw.channels} channel{'s' if raw.channels != 1 else ''} in "
                            "channel_names): add the channel there or remove the files")
        if raw.unsupported_images:
            problems.append(f"{len(raw.unsupported_images)} JPEG file{'s' if len(raw.unsupported_images) != 1 else ''} "
                            f"in imagesTr (e.g. {raw.unsupported_images[0]}), which nnU-Net cannot read: convert "
                            "them to .png or .tif")
        if raw.unmatched_labels:
            problems.append(f"{len(raw.unmatched_labels)} label file{'s' if len(raw.unmatched_labels) != 1 else ''} "
                            f"match no image (e.g. {raw.unmatched_labels[0]}): a label is named like its image "
                            "without the _0000 (case_001.nii.gz for case_001_0000.nii.gz)")
        undefined = self.undefined_label_ids()
        if undefined:
            problems.append(f"label files contain IDs that dataset.json does not define: "
                            f"{', '.join(map(str, undefined[:10]))} (see Statistics; e.g. 255 instead of 1); "
                            "nnU-Net fails on them")
        return problems

    def undefined_label_ids(self) -> list[int]:
        """Label IDs that Compute statistics found in labelsTr but dataset.json does not define (while the
        statistics are up to date)."""
        found = self.analysis.get("undefined_label_ids") if isinstance(self.analysis, dict) else None
        if not isinstance(found, dict) or self.analysis_outdated:
            return []
        ids = {value for values in found.values() if isinstance(values, list) for value in values
               if isinstance(value, int) and not isinstance(value, bool)}
        return sorted(ids - set(self.names) - {self.ignore_id})

    def data_state(self) -> tuple[str, str]:
        if self.raw_path is not None and not _is_dir(self.raw_path):
            return "error", (f"Data · {self.raw_path} no longer exists (moved, renamed or its drive disconnected): "
                             "reconnect it, or go back to the overview and click Refresh")
        if not self.raw.cases:
            if self.raw_path is None:
                return "", "Data · no raw data on this computer (nnUNet_raw)"
            if self.raw.file_list:
                return "error", "Data · the \"dataset\" file list in dataset.json lists no cases"
            misplaced = _images_in(self.raw_path)
            if misplaced:
                return "error", (f"Data · {misplaced} image{'s are' if misplaced != 1 else ' is'} in the dataset "
                                 "folder itself: move images into imagesTr (and labels into labelsTr)")
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
        if self.plans and self.raw.cases and not self.data_problems():
            planned = sorted({config.name for plans in self.plans for config in plans.configurations})
            return "todo", (f"Preprocess · planned ({', '.join(planned)}), but nothing preprocessed yet: preprocess "
                            "one of these configurations (2D images have no 3d_fullres)")
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
    listed = listed_cases(raw_path)
    labels = raw_path / "labelsTr"
    try:
        if listed is not None:  # the "dataset" file list of dataset.json
            return sum(1 for _images, path in listed.values() if path.is_file() and path.stat().st_mtime > analysed)
        if not labels.is_dir():
            return 0
        return sum(1 for path in labels.iterdir() if path.is_file() and path.stat().st_mtime > analysed)
    except OSError:  # unreadable labelsTr, or a label removed while counting
        return 0


def _label_analysis(raw_path: Path | None) -> dict | None:
    if raw_path is None:
        return None
    try:
        return load_label_analysis(raw_path)
    except (ValueError, RecursionError):  # a statistics cache that is not UTF-8 or nested too deeply: none
        return None


def _bom_hint(path: Path) -> str:
    """Why a dataset.json that looks fine cannot be read: a UTF-8 byte order mark, which Windows PowerShell 5.1 writes
    (Set-Content / Out-File -Encoding utf8) and which neither nnWorkbench nor nnU-Net (json.load) accept."""
    try:
        with path.open("rb") as file:
            bom = file.read(3) == b"\xef\xbb\xbf"
    except OSError:
        return ""
    return (" It starts with a byte order mark (BOM), e.g. from Windows PowerShell 5.1's Set-Content -Encoding utf8, "
            "which nnU-Net cannot read either: save it as UTF-8 without BOM." if bom else "")


def load_state(record: DatasetRecord, ended: dict[Path, float] | None = None) -> DatasetState:
    """Everything the pages show. ``ended``: fold folders whose GUI training job ended, and when (see fold_info)."""
    source = manifest_source(record.raw_path, record.preprocessed_path, record.results_path)
    manifest = json_object(source[1] / "dataset.json") if source else None
    errors = [*definition_errors(manifest), *filter(None, [file_ending_error(manifest)])] \
        if isinstance(manifest, dict) else []
    raw_json = record.raw_path / "dataset.json" if record.raw_path else None
    if raw_json is not None and not raw_json.is_file():
        errors.insert(0, "dataset.json is missing in nnUNet_raw.")
    elif raw_json is not None and source and source[0] == "raw" and not isinstance(manifest, dict):
        errors.insert(0, "dataset.json cannot be read (no valid JSON object)." + _bom_hint(raw_json))
    raw = raw_overview(record.raw_path, manifest)
    if isinstance(manifest, dict) and source and source[0] == "raw" and raw.cases:
        problem = num_training_error(manifest, len(raw.cases))
        if problem:
            errors.append(problem)
        problem = file_ending_mismatch(manifest, [path.name for case in raw.cases for path in case.image_files],
                                       [case.label_file.name for case in raw.cases if case.label_file])
        if problem:
            errors.append(problem)
    analysis = _label_analysis(record.raw_path)
    return DatasetState(
        record=record,
        manifest=manifest,
        manifest_kind=source[0] if source else None,
        manifest_errors=tuple(errors),
        raw=raw,
        plans=plans_overview(record.preprocessed_path, record.raw_path),
        models=models_overview(record.results_path, ended),
        analysis=analysis,
        analysis_outdated=_analysis_outdated(record.raw_path, analysis),
        fold_count=splits_fold_count(record.preprocessed_path),
    )


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _images_in(folder: Path) -> int:
    """Image files directly in a dataset folder (copied there instead of into imagesTr)."""
    try:
        return sum(1 for path in folder.iterdir() if (has_image_suffix(path) or path.suffix.lower()
                                                       in UNSUPPORTED_IMAGE_SUFFIXES) and path.is_file())
    except OSError:
        return 0
