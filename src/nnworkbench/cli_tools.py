"""`nnworkbench-run create-dataset / new-iteration / validation`: the GUI's dataset steps for terminals and agents.

Qt-free, so they also work in the core installation (e.g. on a cluster).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

from .json_output import print_json
from .path_setup import NON_ASCII_HINT, unusable_for_nnunet
from .services.commands import wrapped_command
from .services.dataset_contents import json_object
from .services.dataset_creator import (
    create_dataset,
    create_next_iteration,
    next_free_dataset_id,
    next_iteration_name,
)
from .services.dataset_description import create_description, description_path
from .services.dataset_json import file_ending_error
from .services.dataset_scanner import scan_datasets
from .services.image_files import IMAGE_SUFFIXES, image_suffix
from .services.workflow_status import (
    label_names,
    mean_class_dice,
    models_overview,
    validation_cases,
)


def _print(value: dict, as_json: bool, text: str) -> None:
    if as_json:
        print_json(value)
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never fail on a console without UTF-8
    print(text)


def _roots_list(roots: dict[str, Path | None]) -> list[Path | None]:
    return [roots.get("raw"), roots.get("preprocessed"), roots.get("results")]


def run_create_dataset(roots: dict[str, Path | None], dataset_id: int, name: str, labels: list[str],
                       channels: list[str], *, ignore: bool = False, file_ending: str = ".nii.gz",
                       description: str = "", add_description: bool = False, as_json: bool = False) -> int:
    """Create ``DatasetXXX_Name`` with imagesTr, labelsTr and a complete dataset.json (as New dataset in the GUI)."""
    if file_ending_error({"file_ending": file_ending}) or image_suffix("x" + file_ending) != file_ending:
        raise ValueError(f"--file-ending {file_ending!r} is not an image ending nnU-Net reads: use one of "
                         f"{', '.join(IMAGE_SUFFIXES)} (with the dot; JPEG is not supported).")
    names = [label.strip() for label in labels if label.strip()]
    if not names or names[0].casefold() != "background":
        names.insert(0, "background")  # nnU-Net needs background as label 0
    if any(label.casefold() == "ignore" for label in names):
        raise ValueError("Use --ignore instead of listing 'ignore' as a label; it must have the highest ID.")
    if len({label.casefold() for label in names}) != len(names):
        raise ValueError("Label names must be unique.")
    channel_names = [channel.strip() for channel in channels if channel.strip()]
    if not channel_names:
        raise ValueError("Give at least one channel name, e.g. --channels fluorescence.")
    destination = create_dataset(roots.get("raw"), _roots_list(roots), dataset_id, name)
    manifest = {
        "name": destination.name.split("_", 1)[1],
        "description": description,
        "channel_names": {str(index): channel for index, channel in enumerate(channel_names)},
        "labels": {label: index for index, label in enumerate(names)} | ({"ignore": len(names)} if ignore else {}),
        "numTraining": 0,
        "file_ending": file_ending,
    }
    (destination / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {"dataset": destination.name, "path": str(destination), "dataset_json": manifest}
    if add_description:
        result["description"] = str(create_description(destination))
    _print(result, as_json, f"Created {destination}\n  labels: "
           + ", ".join(f"{label} = {value}" for label, value in manifest["labels"].items())
           + f"\n  channels: {', '.join(channel_names)} · file ending {file_ending}\n"
           "Copy the images into imagesTr (and labels into labelsTr), then set numTraining "
           "(nnworkbench-run status shows what is missing).")
    return 0


def run_new_iteration(roots: dict[str, Path | None], source_id: int, dataset_id: int | None = None,
                      name: str | None = None, *, link_images: bool = False, as_json: bool = False) -> int:
    """Copy imagesTr, labelsTr and dataset.json of a dataset into the next iteration (as + New iteration in the GUI);
    ``link_images`` lists the source's images in dataset.json instead of copying them."""
    records = scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results"))
    source = next((record for record in records if record.dataset_id == source_id and record.raw_path), None)
    if source is None:
        raise ValueError(f"No dataset with ID {source_id} in nnUNet_raw.")
    new_id = dataset_id or next_free_dataset_id(source_id + 1, _roots_list(roots))
    new_name = name or next_iteration_name(source.name)
    destination = create_next_iteration(roots.get("raw"), _roots_list(roots), source.raw_path, new_id, new_name,
                                        link_images=link_images)
    result = {"source": source.directory_name, "dataset": destination.name, "dataset_id": new_id,
              "path": str(destination), "images": "linked" if link_images else "copied"}
    _print(result, as_json, f"Created {destination}\n  from {source.directory_name} ("
           + ("images linked in dataset.json; labels and dataset.json copied" if link_images
              else "images, labels and dataset.json") + "). Annotate there, then preprocess and train the new dataset.")
    return 0


def dataset_list(roots: dict[str, Path | None]) -> dict:
    """Every dataset of the saved folders, for picking a free ID or the latest iteration."""
    from .services.dataset_scanner import id_badge

    datasets = []
    for record in scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results")):
        manifest = json_object(record.raw_path / "dataset.json") if record.raw_path else None
        source = manifest.get("inherits_from") if isinstance(manifest, dict) else None
        match = re.match(r"^(?P<base>.+)_iter(?P<number>\d+)$", record.name, flags=re.IGNORECASE)
        badge = id_badge(record)
        datasets.append({
            "id": record.dataset_id, "name": record.name, "folder": record.directory_name,
            "raw": record.raw_path is not None, "preprocessed": record.preprocessed_path is not None,
            "results": record.results_path is not None,
            "id_problem": badge[0] if badge else None,
            # iterations: Cells, Cells_iter1, Cells_iter2 (a branch from the same source has the same name;
            # the higher ID is the newer one)
            "base_name": match["base"] if match else record.name,
            "iteration": int(match["number"]) if match else 0,
            "inherits_from": source if isinstance(source, str) else None,  # written by new-iteration
        })
    return {"datasets": datasets}


def run_list(roots: dict[str, Path | None], *, as_json: bool = False) -> int:
    result = dataset_list(roots)
    lines = [f"{item['id']:>4}  {item['folder']}  ("
             + ", ".join(name for name in ("raw", "preprocessed", "results") if item[name]) + ")"
             + (f"  [{item['id_problem']}]" if item["id_problem"] else "")
             + (f"  from {item['inherits_from']}" if item["inherits_from"] else "") for item in result["datasets"]]
    _print(result, as_json, "\n".join(lines) or "No datasets in the saved nnU-Net folders.")
    return 0


def matching_models(models, text: str) -> list:
    """The models whose label is ``text`` exactly, else those that contain it (all for an empty text)."""
    exact = [info for info in models if info.label == text]
    return exact or [info for info in models if text in info.label]


def validation_report(roots: dict[str, Path | None], dataset_id: int, model: str = "", fold: str = "") -> dict:
    """Validation results per model: Dice per fold and class, and every validation case, weakest first.

    Classes and cases come from the cross-validation folds only: fold "all" is validated on its own training
    cases (and would list each case a second time). It is used only when asked for with ``fold="all"``."""
    records = scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results"))
    record = next((item for item in records if item.dataset_id == dataset_id), None)
    if record is None:
        raise ValueError(f"No dataset with ID {dataset_id} in the saved nnU-Net folders.")
    manifest = json_object(record.raw_path / "dataset.json") if record.raw_path else None
    if manifest is None and record.preprocessed_path:
        manifest = json_object(record.preprocessed_path / "dataset.json")
    names = label_names(manifest)
    models = []
    fold = fold.removeprefix("fold_")  # the folder name, fold_0, for fold 0
    overview = models_overview(record.results_path)
    for info in matching_models(overview, model):
        folds, class_values, cases = {}, {}, []
        for fold_info in info.folds:
            if fold and fold_info.name != fold:
                continue
            summary = json_object(fold_info.path / "validation" / "summary.json")
            if summary is None:
                continue
            folds[fold_info.name] = fold_info.dice
            if fold_info.name == "all" and fold != "all":  # validated on its own training cases: not a validation
                continue
            for class_name, dice in mean_class_dice(summary, names):
                class_values.setdefault(class_name, []).append(dice)
            for case in validation_cases(summary, names):
                cases.append({"case": case.case_id, "fold": fold_info.name, "dice": case.mean_dice,
                              "weakest_class": case.weakest[0] if case.weakest else None,
                              "weakest_dice": case.weakest[1] if case.weakest else None,
                              "prediction": case.prediction_file})
        if not folds:
            continue
        cases.sort(key=lambda case: (case["dice"] is None, case["dice"] if case["dice"] is not None else 0))
        models.append({"model": info.label, "trainer": info.trainer, "plans": info.plans,
                       "configuration": info.configuration, "mean_dice": info.mean_dice, "folds": folds,
                       "classes": {name: sum(values) / len(values) for name, values in class_values.items()},
                       "cases": cases})
    if not models and (model or fold) and any(fold_info.dice is not None for info in overview for fold_info in info.folds):
        # not "no results yet": an agent would train again
        available = "; ".join(f"{info.label} (folds {', '.join(item.name for item in info.folds)})" for info in overview)
        raise ValueError(f"No validation results match --model '{model}' --fold '{fold}'. Models: {available}.")
    return {"dataset": record.directory_name, "models": models}


CLASS_LINES = 20  # classes in the text output of `validation` (at least; all with --cases 0)


def _format_validation(report: dict, limit: int, fold: str = "") -> str:
    lines = [report["dataset"]]
    if not report["models"]:
        lines.append("  No validation results yet (train a fold first; nnU-Net validates it at the end).")
    for model in report["models"]:
        mean = f"{model['mean_dice']:.3f}" if model["mean_dice"] is not None else "-"
        lines += ["", f"  {model['model']}   mean Dice {mean} (fold all excluded)"]
        source = ("fold all, trained on these cases" if fold == "all" else f"fold {fold}" if fold
                  else "the cross-validation folds, fold all excluded")
        lines.append("    folds    " + " | ".join(
            f"{name}: {dice:.3f}" if dice is not None else f"{name}: -" for name, dice in model["folds"].items()))
        # one class per line, weakest first; with many classes only the weakest (--json lists all)
        classes = sorted(model["classes"].items(), key=lambda item: item[1])
        shown_classes = classes[:max(limit, CLASS_LINES)] if limit else classes
        if classes:
            lines.append(f"    weakest classes ({len(shown_classes)} of {len(classes)}):" if len(shown_classes) < len(classes)
                         else f"    classes ({len(classes)}), weakest first:")
            lines += [f"      {dice:.3f}  {name}" for name, dice in shown_classes]
        shown = model["cases"][:limit] if limit else model["cases"]
        lines.append(f"    weakest cases ({len(shown)} of {len(model['cases'])}) from {source}:")
        for case in shown:
            dice = f"{case['dice']:.3f}" if case["dice"] is not None else "-"
            weakest = (f"  weakest class {case['weakest_class']} ({case['weakest_dice']:.3f})"
                       if case["weakest_class"] else "")
            lines.append(f"      {case['case']:<24} fold {case['fold']:<4} Dice {dice}{weakest}")
    return "\n".join(lines)


def run_validation(roots: dict[str, Path | None], dataset_id: int, *, model: str = "", fold: str = "",
                   limit: int = 10, as_json: bool = False, open_case: str = "") -> int:
    fold = fold.removeprefix("fold_")
    report = validation_report(roots, dataset_id, model, fold)
    if open_case:
        return _open_validation_case(roots, report, dataset_id, open_case)
    _print(report, as_json, _format_validation(report, limit, fold))
    return 0


def _open_validation_case(roots: dict[str, Path | None], report: dict, dataset_id: int, case: str) -> int:
    """Open a validation case in Napari: image, label and the model's prediction (as on the Validate page)."""
    matches = [(entry, item) for entry in report["models"] for item in entry["cases"] if item["case"] == case]
    if not matches:
        raise ValueError(f"No validation result for case '{case}'. Use --model and --fold to choose, or check the "
                         "case name in the validation output.")
    if len({entry["model"] for entry, _item in matches}) > 1:
        raise ValueError(f"Case '{case}' was validated by several models; choose one with --model.")
    entry, item = matches[0]
    records = scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results"))
    record = next(record for record in records if record.dataset_id == dataset_id)
    prediction = Path(item["prediction"])
    if not prediction.is_file() and record.results_path:  # summary.json may hold paths from another machine
        prediction = record.results_path / entry["model"] / f"fold_{item['fold']}" / "validation" / prediction.name
    if not prediction.is_file():
        raise ValueError(f"Prediction file not found: {prediction}")
    try:
        from .services.napari_inspector import open_validation_prediction
    except ImportError as error:
        raise RuntimeError("Napari is not installed. Use the full installation to inspect results.") from error
    print(f"Opening {case} (fold {item['fold']}, {entry['model']}): image, label and prediction")
    open_validation_prediction(prediction, record.raw_path)
    return 0


# ---------------------------------------------------------------- suggest: the command the GUI would prefill


def suggestion(roots: dict[str, Path | None], dataset_id: int, step: str, *, device=None, model: str = "",
               input_directory: str = "", output_directory: str = "") -> dict:
    """The command the GUI prefills for a step, with the reason for each choice (shared rules: services.defaults)."""
    from .services.commands import prediction_command, preprocessing_command, training_command
    from .services.defaults import (
        DEFAULT_CONFIGURATION,
        DEFAULT_PLANNER,
        PREDICTION_PROCESSES,
        default_configuration,
        default_plans,
        next_fold,
        prediction_folds,
        preprocessing_configuration,
        preprocessing_processes,
        trainer_for,
    )
    from .services.device_probe import default_choice, device_argument
    from .status_report import find_state

    state = find_state(dataset_id, roots)
    dataset = state.record.dataset_id
    choice = default_choice(device)
    device_flag = device_argument(choice)
    device_reason = (f"device: {choice} ({device.summary})" if device is not None
                     else "device: GPU (nnU-Net's default; device not checked)")
    if step == "preprocess":
        processes = preprocessing_processes()
        configuration = preprocessing_configuration(state.manifest, (config.name for plans in state.plans
                                                                     for config in plans.configurations))
        command = preprocessing_command(dataset, [configuration], DEFAULT_PLANNER, processes=processes)
        reasons = [(f"planner {DEFAULT_PLANNER}: the default (needs ~24 GB of GPU memory for training; with less, "
                    "use nnUNetPlannerResEncM or ExperimentPlanner)"),
                   (f"configuration {configuration}: " + ("2D images (.png/.bmp, or plans without 3d_fullres)" if configuration == "2d"
                                                          else "for 3D images (use 2d for 2D images)")),
                   f"{processes} processes (-np): up to 8, never more than CPU cores",
                   "--verify_dataset_integrity: checks the dataset before preprocessing"]
    elif step == "train":
        preprocessed = [plans for plans in state.plans if any(config.preprocessed for config in plans.configurations)]
        if not preprocessed:
            raise ValueError("No preprocessed data yet: preprocess first (suggest --step preprocess).")
        plans_name = default_plans(plans.name for plans in preprocessed)
        plans = next(item for item in preprocessed if item.name == plans_name)
        configuration = default_configuration(config.name for config in plans.configurations if config.preprocessed)
        trainer = trainer_for(state.sparse)
        existing = next((item for item in state.models
                         if (item.trainer, item.plans, item.configuration) == (trainer, plans_name, configuration)), None)
        fold = next_fold(state.fold_count, (item.name for item in existing.folds) if existing else ())
        if fold is None:
            raise ValueError(f"All folds of {trainer}__{plans_name}__{configuration} exist already. To resume a "
                             "stopped fold, add --c to its command; to train one again, use the GUI or the "
                             "command with that fold.")
        command = training_command(dataset, plans_name, configuration, trainer, fold, device=device_flag)
        reasons = [f"trainer {trainer}: " + ("sparse labels (dataset.json has an ignore label)" if state.sparse
                                             else "dense labels (no ignore label)"),
                   f"plans {plans_name}: " + ("the default plans" if plans_name == "nnUNetResEncUNetLPlans"
                                              else "the first preprocessed plans (the default ones are missing)"),
                   f"configuration {configuration}: " + ("the default" if configuration == DEFAULT_CONFIGURATION
                                                         else "the first preprocessed configuration"),
                   (f"fold {fold}: the next fold this model does not have yet ({state.fold_count} folds in "
                    "splits_final.json, then all)"),
                   device_reason]
    elif step == "predict":
        models = [item for item in matching_models(state.models, model) if item.finished_folds]
        if not models:
            raise ValueError("No model with a finished fold" + (f" matching '{model}'" if model else "") + ".")
        chosen = models[0]
        folds = prediction_folds(fold.name for fold in chosen.finished_folds)
        # ~ expanded as in the GUI (an unknown ~user stays as typed)
        input_directory, output_directory = (os.path.expanduser(folder) if folder else ""
                                             for folder in (input_directory, output_directory))
        command = prediction_command(dataset, input_directory or "INPUT_FOLDER", output_directory or "OUTPUT_FOLDER",
                                     chosen.plans, chosen.configuration, trainer=chosen.trainer,
                                     folds=" ".join(folds), continue_prediction=True,
                                     preprocessing_processes=PREDICTION_PROCESSES,
                                     export_processes=PREDICTION_PROCESSES, device=device_flag)
        reasons = [f"model {chosen.label}: " + ("the first model with finished folds" if not model
                                                else f"the first model matching '{model}'"),
                   f"folds {' '.join(folds)}: the finished cross-validation folds (fold all only if it is the only one)",
                   "mirroring (test-time augmentation) on: best quality (add --disable_tta to be faster)",
                   "--continue_prediction: skips cases that already have a prediction in the output folder",
                   f"{PREDICTION_PROCESSES} processes for preprocessing (-npp) and export (-nps), nnU-Net's default",
                   device_reason]
        if not input_directory or not output_directory:
            reasons.append("replace INPUT_FOLDER and OUTPUT_FOLDER (or pass --input and --output)")
    else:
        raise ValueError("Step must be preprocess, train or predict.")
    arguments = ["nnworkbench-run", *shlex.split(command)]
    return {"dataset": state.record.directory_name, "step": step, "command": wrapped_command(command),
            "arguments": arguments, "reasons": reasons,
            "warnings": _suggest_warnings(state, step, input_directory, output_directory)}


def _suggest_warnings(state, step: str, input_directory: str = "", output_directory: str = "") -> list[str]:
    """What speaks against running the suggested command now (the GUI disables or asks in these cases)."""
    warnings = []
    if step == "preprocess" and state.data_problems():
        warnings.append("Data must be fixed first: " + "; ".join(state.data_problems()))
    if step == "train":
        if any(plans.outdated for plans in state.plans):
            warnings.append("Preprocessing is outdated (labelsTr changed after it): preprocess again first, or the "
                            "training uses the old labels.")
        for model in state.models:
            for fold in model.folds:
                if fold.status in ("training", "stopped"):
                    warnings.append(f"{model.label} fold {fold.name} is {fold.status}" + (
                        ": wait for it" if fold.status == "training" else ": resume it with --c before a new fold"))
    if step == "predict" and input_directory:
        source = Path(input_directory)
        if not source.is_dir():
            warnings.append(f"The input folder {source} does not exist.")
        elif output_directory and source.resolve() == Path(output_directory).resolve():
            warnings.append("The output folder is the input folder: choose another output folder.")
        unusable = [folder for folder in (input_directory, output_directory) if folder and unusable_for_nnunet(folder)]
        if unusable:
            warnings.append(f"{NON_ASCII_HINT}: {', '.join(unusable)}.")
    return warnings


def run_suggest(roots: dict[str, Path | None], dataset_id: int, step: str, *, check_device: bool = True,
                model: str = "", input_directory: str = "", output_directory: str = "", as_json: bool = False) -> int:
    from .services.device_probe import probe_device

    device = probe_device() if check_device and step in ("train", "predict") else None
    result = suggestion(roots, dataset_id, step, device=device, model=model, input_directory=input_directory,
                        output_directory=output_directory)
    warnings = "".join(f"\nWarning: {warning}" for warning in result["warnings"])
    _print(result, as_json, result["command"] + "\n\nWhy:\n" + "\n".join(f"  - {reason}" for reason in result["reasons"])
           + ("\n" + warnings if warnings else ""))
    return 0


def run_describe(roots: dict[str, Path | None], dataset_id: int, *, as_json: bool = False) -> int:
    """Create dataset-description.md for a dataset (never replaces an existing one) and report its path."""
    records = scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results"))
    record = next((item for item in records if item.dataset_id == dataset_id and item.raw_path), None)
    if record is None:
        raise ValueError(f"No dataset with ID {dataset_id} in nnUNet_raw.")
    path = description_path(record.raw_path)
    created = not path.is_file()
    if created:
        create_description(record.raw_path)
    _print({"dataset": record.directory_name, "path": str(path), "created": created}, as_json,
           f"{'Created' if created else 'Already there:'} {path}\n"
           "Fill it in with a text editor or in the GUI (Data page, Description); put images into "
           "description_images/ next to it.")
    return 0
