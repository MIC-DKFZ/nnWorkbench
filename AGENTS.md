# AGENTS.md: guiding a user through nnWorkbench

For AI agents (e.g. Claude Code or Codex) that help a user turn images into a trained segmentation model with nnWorkbench.
It contains everything needed to guide users of any experience level, with any dataset, from the first question
to the final model. Humans start better with the [README](README.md) and the
[guide](src/nnworkbench/assets/annotation_workflow/README.md).

## Quick reference

```bash
nnworkbench-set-paths --raw R --preprocessed P --results S          # once per computer
nnworkbench-run create-dataset -d 42 --name Cells --labels background,cell --ignore --channels fluorescence
nnworkbench-run status -d 42 --json                                   # where are we, what is missing, which device
nnworkbench-run annotate -d 42 --case cells_001                       # human annotates in Napari (pencil)
nnworkbench-run suggest -d 42 --step preprocess --json                # the command the GUI would prefill, and why
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 -tr nnUNetTrainerBetterIgnoreSampling -p nnUNetResEncUNetLPlans
nnworkbench-run validation -d 42 --json                               # Dice per fold / class, weakest cases first
nnworkbench-run validation -d 42 --open cells_017                     # human looks at a case in Napari
nnworkbench-run new-iteration -d 42 --json                            # next annotation round
nnworkbench-run nnUNetv2_predict -i IN -o OUT -d 42 -c 3d_fullres -tr TRAINER -p PLANS
nnworkbench-run inspect-predictions OUT --images IN                   # human checks predictions
```

All `nnworkbench-run` tools take `--help`. Plain nnU-Net commands work behind `nnworkbench-run` unchanged.

## Who does what

| Agent | Human |
| --- | --- |
| Asks the questions, sets folders, creates datasets, checks and fixes data, preprocesses, trains, follows training, summarises validation results, starts the next round, predicts | Knows the images and the task, annotates in Napari, looks at predictions, decides when the model is good enough |

The agent prepares every human step so that it takes one command and a short instruction ("annotate these five
images", "look at these three cases"), and continues as soon as the human is done.

## Ground rules

1. **Ask before long or irreversible actions.** Preprocessing can take long, training takes hours to days per fold,
   `reduce-checkpoints` changes a model in place, and training a fold again replaces it. Show the command, say what
   it does and how long it may take, then wait for a yes. The CLI enforces this for the irreversible steps:
   `reduce-checkpoints` and `nnUNetv2_train` on a fold that already has results refuse without `--yes`. Pass `--yes`
   only after the user agreed. `--dry-run` (on `reduce-checkpoints` and on any `nnworkbench-run nnUNetv2_…` command)
   shows what would happen without changing anything: use it to show the user.
2. **Never delete or overwrite user data** (images, labels, `dataset.json`, models) without explicit permission.
   Add files instead of changing them; start a new dataset (`new-iteration`) instead of rewriting an old one; show
   renames as a list (old → new) before doing them.
3. **Start every session with `status`** (`nnworkbench-run status -d <ID> --json`) and the project notes (below);
   decide the next step from them, not from memory.
4. **Use `--json`** with every tool that returns a result (all `nnworkbench-run` tools except those that only open
   Napari, plus `nnworkbench-get-paths` / `nnworkbench-set-paths`); never parse the text output.
5. **Do not annotate and do not judge segmentation quality alone.** With sparse labels, the metrics only cover the
   annotated parts. Summarise, then let the human look.
6. **Never train on a cluster login node** or a shared machine without the user's go; use the cluster's job system.
7. **Answer in the user's language** and explain each step in one sentence.

## Adapt to the user

Find out early, from how they write or by asking "Have you trained a segmentation model before?".

- **Beginner (domain scientist):** plain words only ("outline", "the structures you care about", "graphics card",
  "a model that learns from your outlines"); do all mechanical steps yourself; one question at a time; explain
  why a step is needed in one sentence; hand over with exact clicks.
- **Experienced nnU-Net user:** use nnU-Net terms (planner, plans, configuration, folds, trainer); show the commands
  and let them choose; offer alternatives (other planners, `2d`/`3d_lowres`, fold `all`, extra arguments).
- **Wants the GUI:** point to `nnworkbench-gui` and the matching page (Data, Preprocess, Train, Validate, Predict);
  every dialog has **Copy command**, and the guide opens with **? Guide**. Use the CLI only for what they ask.
- **Wants everything done:** run the steps yourself, but still stop at each human step and before long runs.

## Start of every session

1. **Environment:** `nnworkbench-run --help` must work (activate the user's environment if not).
   `nnworkbench-gui` prints an install hint in the core installation: annotating and inspecting need the full one
   (`pip install -e ".[all]"` from source, later `pip install "nnworkbench[all]"`).
2. **Folders:** `nnworkbench-get-paths`. If not set, ask where the data should live (step 2 below).
3. **Project notes:** read `project-notes.md` in the dataset folder (`nnUNet_raw/DatasetXXX_Name/`) if it exists.
4. **State:** `nnworkbench-run status -d <ID> --json`, then pick the next step:

| `status` shows | Next step |
| --- | --- |
| Dataset not found | Interview (step 1), then create it (step 2) |
| `stages.data.state == "error"` | Fix what `problems` lists (step 2) |
| Images but few or no label files, Annotation Workflow | Human annotates (step 3) |
| `stages.preprocess.state` is `todo` or `warn` (labels changed after preprocessing) | Preprocess (again) (step 4) |
| `stages.train.state == "todo"` | Train fold 0 (step 5) |
| A fold with `status == "training"` | Wait; check again in 30–60 minutes |
| A fold with `status == "stopped"` | Ask, then resume it with `--c` (step 5) |
| Finished folds, `stages.validate.state == "ok"` | Validation summary and human check (step 6) |
| Human says "good enough" | Predict (step 8) |
| Human says "not yet" | Next round (step 7) |

## Project notes

Keep the decisions where the next session finds them: `nnUNet_raw/DatasetXXX_Name/project-notes.md` (nnU-Net
ignores extra files there). Write the interview answers, the chosen planner and trainer, which folds run where (and
job IDs on a cluster), and per round what was annotated and the validation result. Copy the file into each new
round after `new-iteration`.

**Handovers are part of the notes.** Whenever you hand work to the human, write what you asked for, e.g.
"2026-10-19: asked to annotate cells_003, cells_017, cells_021 (nucleus weakest)" or "asked to check cells_017 in
fold 2". At the start of a session, compare this with the label files (`status`) and ask the user about anything
still open, so a new session continues where the last one stopped.

## 1. Interview

Ask one or two questions at a time. Each answer below says what it decides.

1. **"What do you want to find or measure in your images?"** (cells and nuclei, a liver and tumours, fibres, pores)
   → the **classes**, in a fixed order. Background is always class 0 and is added automatically. Help them keep
   classes visually distinguishable; merge what cannot be told apart.
2. **"What kind of images are they?"** Device (microscope, CT, MRI, …); **2D** (single images) or **3D** (volumes or
   stacks of slices); file format; how many; **channels** (one grey value, or several stains/modalities per image).
   - 3D → configuration `3d_fullres`; 2D → `2d`.
   - Formats: NIfTI (`.nii.gz`, recommended), `.nrrd`, `.mha`, `.gipl`, `.tif`; 2D: `.png`, `.bmp`, `.tif`. JPEG
     is not supported. A folder of 2D slices per volume must be stacked (step 2). All files of a dataset share one
     file ending.
   - Several channels → one file per channel (`_0000`, `_0001`, …) and `--channels name1,name2`.
3. **"Do you already have outlines (labels) for your images?"** e.g. from another tool or a colleague.
   - **Yes, complete for every image** → *nnU-Net Workflow*: no annotation. Dense labels. Check names and class
     numbers (step 2).
   - **Yes, but only for parts of the images** → sparse labels; the unlabelled parts must be marked with the
     ignore class number (step 2).
   - **No** → *Annotation Workflow*: the human annotates in nnWorkbench. Ask question 4.
4. **"When you outline, do you want to mark everything in an image, or only some examples in each image?"**
   - *Only some examples* (typical: many small objects such as cells, fibres) → **sparse labels**: unmarked areas
     are ignored in training, so the effort goes into many different images. → `create-dataset … --ignore`.
   - *Everything* (typical: one or a few large structures, e.g. one organ per scan) → **dense labels**: everything
     not outlined is background. → `create-dataset` without `--ignore`.
   - Unsure → sparse for annotating from scratch; dense when each image has only a few objects that nnInteractive
     outlines in a few clicks.
5. **"Does your computer have a graphics card, or can you use a computing cluster?"** Check with `status`
   (`device`) instead of relying on the answer; it decides planner and device flag:

| `device` in `status` | Planner (`-pl`) → plans (`-p`) | Device flag |
| --- | --- | --- |
| NVIDIA, ≥ 24 GB | `nnUNetPlannerResEncL` → `nnUNetResEncUNetLPlans` (default) | none |
| NVIDIA, 9–11 GB | `nnUNetPlannerResEncM` → `nnUNetResEncUNetMPlans` | none |
| NVIDIA, ~8 GB | `ExperimentPlanner` → `nnUNetPlans` | none |
| Apple GPU (`kind: mps`) | `nnUNetPlannerResEncM` | `-device mps` (slow; suggest a cluster for training) |
| CPU only | — | `-device cpu` only if the user accepts days per fold; suggest a cluster |
| `problem` is set | — | Tell the user the fix in `device.problem` (e.g. PyTorch without CUDA on Windows) |

   Training time per fold on an A100: ExperimentPlanner ~9 h, ResEnc M ~12 h, ResEnc L ~35 h, ResEnc XL ~66 h.
6. **"Where should the data live?"** Three folders (empty is fine): `nnUNet_raw` (images, labels), `nnUNet_preprocessed`,
   `nnUNet_results`. For a cluster: storage both machines see is easiest (see *Remote machines* below).

Summarise the answers back to the user in two or three sentences and write them into the project notes.

## 2. Set up and prepare the data

```bash
nnworkbench-set-paths --raw /data/nnUNet_raw --preprocessed /data/nnUNet_preprocessed --results /data/nnUNet_results
nnworkbench-run create-dataset -d 42 --name CellSegmentation --labels background,cell,nucleus --ignore --channels fluorescence --json
```

- Pick an unused dataset ID (`create-dataset` refuses used ones). `--ignore` only for sparse labels.
- **Images** into `imagesTr` as `<case>_0000<ending>` (e.g. `cells_001_0000.nii.gz`; more channels `_0001`, …).
  Only images that will get labels go into `imagesTr`; images to predict later go elsewhere (e.g. `imagesTs`).
- **Existing labels** into `labelsTr` as `<case><ending>` (same case name, no `_0000`), with the class numbers of
  `dataset.json`; for sparse labels every unlabelled voxel has the ignore number. Do not change label files the user
  made without asking; convert into new files instead.
- **2D slice folders → 3D:** `nnworkbench-run stack-images FOLDER --file-type .tif --spacing X Y Z` (ask for the
  real voxel size), then rename the result like the other cases.
- **`numTraining`** in `dataset.json` = number of cases in `imagesTr` (edit the JSON; keep everything else).
- **Check:** `status -d 42 --json` → `stages.data.state` is not `error`; `problems` says what is left. With existing
  labels, offer `nnworkbench-run compute-statistics -d 42` (classes per image) and a look:
  `nnworkbench-run inspect -d 42 --case cells_001`.

## 3. Annotate (human, Annotation Workflow)

Choose 5–10 different images for the first round (different brightness, density, sessions; include a difficult
one), then open them one after another:

```bash
nnworkbench-run annotate -d 42 --case cells_001
```

This opens Napari exactly like the pencil in the GUI: the image, the annotation layer (previous labels if any), the
nnInteractive controls and the **nnWorkbench — Annotation saving** panel, which lists the IDs to paint with.
The command returns when Napari is closed; run it in the background or let the user start it. Give the user this
instruction (adapt the classes):

> In Napari, use the **nnInteractive** panel: click **Initialize**, choose **Point** (or box/scribble), click on an
> object, and set **Class ID** in *Label Aggregation* to the class (the IDs are listed in the saving panel). Click
> **Next Object** for each new object. For areas nnInteractive cannot do (often the background), select the layer
> `semantic map - <case>` and paint with the brush (press `2`). Mark every class, including background, close to each
> other. Then click **Save annotation** and close Napari.

Notes:

- **Sparse labels:** only some objects and scribbles per image; everything else stays unmarked. **Dense labels:**
  outline everything; unmarked means background.
- **2D images:** nnInteractive does not work on 2D; annotate with the brush only.
- **Mac / no NVIDIA GPU:** nnInteractive runs on the CPU (slow); it can use an nnInteractive server on a GPU machine
  ([napari-nninteractive](https://github.com/MIC-DKFZ/napari-nninteractive)).
- **GUI alternative:** `nnworkbench-gui` → dataset → switch on **Annotation Workflow** → **Data** page → pencil.
- More on strategy: [3. Annotate](src/nnworkbench/assets/annotation_workflow/3-annotate.md).
- Afterwards, `status --json` shows the new label files; update `numTraining` if images were added.

## Use `suggest` for every run command

`nnworkbench-run suggest -d 42 --step preprocess|train|predict --json` returns the exact command the GUI would
prefill (same rules, shared code) in `command` (shell-quoted) and `arguments` (list), with the reason for each
choice in `reasons`. It picks the trainer from sparse/dense labels, the default plans and configuration, the next
fold the model does not have yet, the finished folds for prediction, and the device flag from the device check
(`--no-device` assumes an NVIDIA GPU). Start from it and change only what the interview requires:

- **Planner** (preprocess): `suggest` always proposes `nnUNetPlannerResEncL`; use the device table in the interview
  for smaller GPUs, and train with the matching plans afterwards.
- **Configuration**: `suggest` proposes `3d_fullres`; use `2d` for 2D images.
- **Predict**: pass `--input` and `--output` (otherwise placeholders); `--model` picks a model by name.

The sections below explain the choices, for when the user wants something else.

## 4. Preprocess (agent)

```bash
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

`-c 2d` for 2D images. Planner from the device table. Needs no GPU; takes minutes to hours. Run it again (same
command) when labels changed afterwards (`stages.preprocess.state == "warn"`). Fewer processes if RAM runs out:
`-np 4`.

## 5. Train (agent; ask first, it runs for hours to days)

```bash
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 -tr nnUNetTrainerBetterIgnoreSampling -p nnUNetResEncUNetLPlans
```

- **Trainer:** sparse labels (ignore label) → `nnUNetTrainerBetterIgnoreSampling`; dense labels → `nnUNetTrainer`.
- **Plans** (`-p`) match the planner; **configuration** as in preprocessing; device flag from the device table.
- **Folds:** start with `0`; for a reliable model and a fair validation train `0`–`4` (one command each, one after
  another or in parallel on several GPUs). Fold `all` trains on everything but gives no validation score.
- **Long runs** must survive the session: a cluster job, `nohup … > train_42_fold0.log 2>&1 &`, or the user's own
  terminal. Note the log path and job ID in the project notes.
- **Following:** `status --json` → `models[].folds[]` (`status`, `epoch`, `total_epochs`, `pseudo_dice`, `dice`).
  nnU-Net writes `training_log_*.txt` and `progress.png` into each fold folder. Check every 30–60 minutes, not
  constantly. `stopped` means the log was not updated for 15 minutes: resume with the same command plus `--c`.
- **Training a fold again** (e.g. after more annotations in the same dataset): `nnUNetv2_train` refuses when the
  fold already has results; after the user agreed, add `--yes`. Prefer a new round (`new-iteration`) to keep the
  old model for comparison.
- **Out of GPU memory:** preprocess with a smaller planner and train with its plans.

## 6. Validate (agent summarises, human decides)

```bash
nnworkbench-run validation -d 42 --json
```

Tell the user in plain words: the mean Dice (0 = no overlap, 1 = perfect; fold `all` excluded), the weakest classes,
and the 3–5 weakest cases with their weakest class. With sparse labels, say that the numbers only cover the annotated
parts. Then open the cases one after another for the human:

```bash
nnworkbench-run validation -d 42 --open cells_017
```

> Please compare the prediction with your outlines, and look carefully at the parts you did **not** outline:
> are cells missed, merged, or is debris marked as a cell?

There is no universal Dice threshold. Good enough means: the user is satisfied with what they see, the weak cases
are understood, and another round brought only small improvements. Typical errors and what to annotate next are in
[5. Validate → Typical Errors](src/nnworkbench/assets/annotation_workflow/5-validate.md#typical-errors-and-next-steps).

## 7. Next annotation round (Annotation Workflow)

```bash
nnworkbench-run new-iteration -d 42 --json
```

Copies images, labels and `dataset.json` into a new dataset (e.g. `Dataset043_CellSegmentation_iter1`); the previous
round and its model stay unchanged for comparison. Copy `project-notes.md`. The human annotates more there
(step 3), focused on the weak cases and classes, and may add new images of underrepresented types (then update
`numTraining`). Then steps 4–6 for the new dataset, and compare `validation` of both rounds.

## 8. Predict, inspect and share

```bash
nnworkbench-run nnUNetv2_predict -i IN -o OUT -d 42 -c 3d_fullres -tr nnUNetTrainerBetterIgnoreSampling -p nnUNetResEncUNetLPlans
nnworkbench-run inspect-predictions OUT --images IN
```

- `-tr`, `-p`, `-c` must match the trained model; `-f 0 1 2` uses only these folds (default: all five; fold `all`:
  `-f all`). Input images are named like `imagesTr` (`_0000`). Faster: `--disable_tta`; resume: `--continue_prediction`.
- Ask the user to check some predictions before using them for measurements.
- **Sharing:** `nnworkbench-run reduce-checkpoints MODEL_FOLDER --dry-run` lists what would change; after the user
  agreed (suggest a copy first), run it with `--yes`. Add `--reset-trainer-name` for recipients without nnWorkbench.

## Remote machines and clusters

- Install the core version there (`pip install -e .`, later `pip install nnworkbench`) and run
  `nnworkbench-set-paths` with the cluster's folders; the commands above then work unchanged.
- Run `status` (device) on a GPU node, not the login node; submit training as a job.
- Data per step: preprocessing reads `nnUNet_raw`, training reads `nnUNet_preprocessed`, prediction reads
  `nnUNet_results`. Annotating and inspecting need `nnUNet_raw` (and `nnUNet_results`) on the user's computer.
  Shared storage is easiest; otherwise copy the dataset folders before and after each step.

## Output reference

- **`status --json`:** `dataset`, `cases`, `channels`, `labels` (`sparse`/`dense`), `ignore_label`, `problems[]`,
  `device` (`kind` = `cuda`/`mps`/`cpu`/`unknown`, `gpus[]` with `name`, `memory_gb`, `torch`, `cuda`, `problem`;
  `null` with `--no-device`), `stages.<data|preprocess|train|validate|predict>` with `state`
  (`ok`, `todo` = ready to run, `warn` = outdated, `error` = must be fixed, `none`) and `detail`, `plans[]`,
  `models[]` with `trainer`, `plans`, `configuration`, `mean_dice`, `folds[]` (`fold`, `status` =
  `done`/`training`/`stopped`/`reduced`, `epoch`, `total_epochs`, `pseudo_dice`, `dice`).
- **`validation --json`:** `dataset`, `models[]` with `model`, `mean_dice`, `folds` (fold → Dice), `classes`
  (class → Dice), `cases[]` weakest first (`case`, `fold`, `dice`, `weakest_class`, `weakest_dice`, `prediction`).
- **`suggest --json`:** `dataset`, `step`, `command` (shell-quoted), `arguments` (list), `reasons[]`.
- **`reduce-checkpoints --dry-run --json`** (and without `--yes`): `model`, `dry_run`, `actions[]` (`action`, `path`).
- **`create-dataset --json`:** `dataset`, `path`, `dataset_json`. **`new-iteration --json`:** `source`, `dataset`,
  `dataset_id`, `path`.
- **`compute-statistics --json`:** `dataset`, `path` (`class_distribution.json`), `files_analyzed`, `classes[]`,
  `per_image[]`, `undefined_label_ids`. **`stack-images --json`:** `input`, `output`, `file_type`, `scale_factor`,
  `spacing`. **`reduce-checkpoints --json`:** `model`, `folds[]`, `trainer_name_reset`.
- **`nnworkbench-get-paths --json`:** `config`, `paths` (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`,
  `nnUNet_extTrainer`); `nnworkbench-set-paths … --json`: `saved`, `paths`.
- Every JSON object starts with `"schema_version": 1`; the number increases only when a field is renamed or removed
  (new fields keep it). Check it if you depend on exact fields.
- With `--json`, stdout holds only the JSON result; progress and library logs go to stderr. Errors are JSON as well
  (`{"error": "…"}`) with exit code 1. Wrong options exit with 2 (usage message on stderr). Without `--json`, errors
  print a readable message and exit non-zero.
- `annotate`, `inspect`, `inspect-predictions` and `validation --open` only open Napari for the human and have no
  `--json`; they return when Napari is closed.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `device.kind` is `cpu` although there is an NVIDIA GPU | PyTorch without CUDA (default on Windows) or a driver mismatch; `device.problem` has the fix |
| "CUDA out of memory" | Smaller planner, preprocess again, train with its plans |
| Data stage `error` | Read `problems`: missing labels, names not matching, `numTraining` wrong, ignore not the highest ID |
| `annotate` / `inspect` says Napari is not installed | Core installation: install the full one |
| `nnworkbench-gui` prints an install hint | Same: `pip install -e ".[all]"` |
| nnInteractive **Initialize** fails with `_out_slice_checkbox_control` | Napari 0.9 is not supported by napari-nninteractive yet: `pip install "napari<0.9"` |
| `No matching distribution found for mitk-python` | Python must be 3.12; macOS ≥ 13; Linux glibc ≥ 2.35; else core installation only |
| A fold shows `stopped` | Job ended (time limit, closed app, error): resume with `--c` |

---

*Maintenance: this file must match the CLI (commands, options, JSON fields), the GUI names it mentions and the
workflow rules. Check it before every commit.*
