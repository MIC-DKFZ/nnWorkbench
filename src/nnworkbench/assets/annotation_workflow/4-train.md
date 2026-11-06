# 4. Train

***Turn your labels into an automated segmentation model.***

## TL;DR

```bash
# Preprocess the dataset once
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity

# Train fold 0, then repeat for folds 1 to 4 or train all
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
# Use this trainer for the Annotation Workflow
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 -tr nnUNetTrainerBetterIgnoreSampling -p nnUNetResEncUNetLPlans
```

**nnU-Net:** An automated segmentation framework that configures, plans, trains, and applies deep learning models with minimal manual setup.

Links: [Paper](https://www.nature.com/articles/s41592-020-01008-z) | [Code](https://github.com/mic-dkfz/nnunet)

## Prepare and Train

Training uses the images in `imagesTr` and the labels in `labelsTr` (see [Setup](setup.md#create-a-dataset)). **Annotation Workflow:** `labelsTr` is filled by saving your annotations in nnWorkbench (see [3. Annotate](3-annotate.md)).

### 1. Check the Filenames

Images end in a channel suffix, labels do not. Most images have one channel (`_0000`); further channels are stored as separate files (`_0001`, …) and listed in `channel_names` in `dataset.json`.

```text
imagesTr:  cells_001_0000.nii.gz  (+ cells_001_0001.nii.gz, … for more channels)
labelsTr:  cells_001.nii.gz
```

**In nnWorkbench:** The **Data** page marks images without a label with a red ✗ and checks the file names and channels.

### 2. Inspect the Labels (Optional)

Open a few images together with their labels in a viewer such as Napari and check that the labels are correct and lie on the right structures. This opens the dataset in Napari, starting at the first image; use `-i` to start at another one:

```bash
nnworkbench-run inspect -d 42
```

**In nnWorkbench:** On the **Data** page, click the magnifier button (**Open in Napari**) next to a case. Napari opens the image together with its label; **Previous (A)** and **Next (D)** step through the cases.

### 3. Analyze the Labels (Optional)

Count in how many images each class occurs and how many voxels it covers, to spot missing or rare classes:

```bash
nnworkbench-run compute-statistics -d 42
```

**In nnWorkbench:** On the **Data** page, click **Compute statistics**, then switch the case table to **Statistics** to see the results.

### 4. Preprocess

nnU-Net analyzes your dataset, plans the network and training settings, and preprocesses the images:

```bash
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres \
    -pl nnUNetPlannerResEncL --verify_dataset_integrity
```

**In nnWorkbench:** On the **Preprocess** page, click **▶ Preprocess…**, choose the **Planner** and the **Configurations**, then click **▶ Preprocess** (see [Planners and GPU Memory](#planners-and-gpu-memory)). **Processes** sets how many cases are preprocessed in parallel (fewer use less RAM), and **Verify dataset integrity** checks the dataset first. Afterwards, the **Plans** table lists every preprocessed plans and configuration; select one to see its details, open preprocessed cases in Napari with the magnifier button, or click **↻ Run again…**, e.g. after the labels changed (the plans are then marked *outdated*).

nnU-Net offers different model configurations:

- **`2d`:** The model looks at one 2D slice at a time. Use it for 2D images, and for 3D data with only few, thick slices.
- **`3d_fullres`:** The model looks at 3D blocks of the volume at full resolution, so it also sees the context above and below a slice. Use it for 3D data; this is usually the best choice.
- **`3d_lowres`:** The model looks at a downsampled version of the volume. Only useful for very large images, where `3d_fullres` sees too little context.

### 5. Train the Model

The `nnworkbench-run` prefix uses the nnU-Net folders saved in [Setup](setup.md#set-the-nnu-net-folders) and makes the bundled trainers available to nnU-Net.

```bash
nnworkbench-run nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr TRAINER -p PLANS
```

Choose the trainer and fold as explained in [Trainer](#trainer) and [Folds](#folds). For example, train fold 0 of a 3D model, then repeat for folds 1 to 4:

**nnU-Net Workflow:**

```bash
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 \
    -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
```

**Annotation Workflow:**

```bash
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 \
    -tr nnUNetTrainerBetterIgnoreSampling -p nnUNetResEncUNetLPlans
```

To train a single model on all images instead, use fold `all` in place of `0`.

**In nnWorkbench:** On the **Train** page, click **▶ Train…** and choose the **Trainer**, the **Plans** (e.g. `nnUNetResEncUNetLPlans`), the **Configuration** and the **Folds**, then click **▶ Train**. Several folds train one after another; at least one fold stays selected. `nnUNetTrainer` is preselected; with an `ignore` label in `dataset.json` (sparse labels), `nnUNetTrainerBetterIgnoreSampling` is preselected instead. **Device** chooses where the training runs (see [Device](#device-gpu-or-cpu)).

### 6. Follow the Training

Training runs unattended and takes hours to days per fold (see [Planners and GPU Memory](#planners-and-gpu-memory)). nnU-Net writes a log and a `progress.png` plot of the training curves into each fold folder in `nnUNet_results`. `nnworkbench-run status -d 42` shows the state of all folds in a terminal.

**In nnWorkbench:** The **Models** table on the **Train** page lists every model with its **Mean Dice** and **Folds**; click a model to see its folds with **Status**, **Epoch**, **Dice** and **Pseudo Dice**. The status of a fold is *done*, *training* (its log was updated in the last 15 minutes, or within two epochs when epochs take longer, e.g. on the CPU or an Apple GPU), *stopped* (can be continued) or *final checkpoint only* (a shared model). Select a fold for its details and actions:

- **▶ Continue…** resumes a stopped fold from its latest checkpoint (`--c`).
- **↻ Run again…** trains the fold again from the start with the same settings.
- **▶ Next fold…** trains the next missing fold of this model with the same settings.

The console at the bottom shows the training output. Closing nnWorkbench stops a running training (it asks first); resume it later with **▶ Continue…**. To train without nnWorkbench, use **Copy command** and run it in a terminal.

## Options Explained

The steps above use the recommended settings. This section explains the choices behind them.

### Planners and GPU Memory

The planner decides how large the network is, and therefore how much GPU memory it needs and how long training takes. It is chosen during preprocessing (`-pl`) and creates a *plans* file that training and prediction then refer to (`-p`).

| Planner (`-pl`) | Plans (`-p`) | GPU memory | Training time* |
| --- | --- | --- | --- |
| `ExperimentPlanner` | `nnUNetPlans` | ~8 GB | ~9 h |
| `nnUNetPlannerResEncM` | `nnUNetResEncUNetMPlans` | 9–11 GB | ~12 h |
| **`nnUNetPlannerResEncL`** *(recommended)* | **`nnUNetResEncUNetLPlans`** | 24 GB | ~35 h |
| `nnUNetPlannerResEncXL` | `nnUNetResEncUNetXLPlans` | 40 GB | ~66 h |

*\*Per fold on an NVIDIA A100; your times depend on the GPU and the data. On a workstation GPU such as an RTX 3090 or 4090, expect roughly one and a half to two times these times; on older or smaller cards considerably more.*

We recommend **`nnUNetPlannerResEncL`**, which is used in all examples of this guide. If your GPU has less than 24 GB of memory, use `nnUNetPlannerResEncM` or `ExperimentPlanner` instead, and replace the plans name in the commands accordingly.

### Device (GPU or CPU)

nnU-Net trains and predicts on an NVIDIA GPU by default. On a Mac with Apple silicon it can use the Apple GPU (`-device mps`), and every computer can use the CPU (`-device cpu`), but training on the CPU takes days per fold. Training on the Apple GPU works, but is much slower than on an NVIDIA GPU, and the GPU shares its memory with the system: use `nnUNetPlannerResEncM` or `ExperimentPlanner` there.

**In nnWorkbench:** The Train and Predict dialogs have a **Device** choice: **GPU** (NVIDIA, no `-device`), **MPS** (the Apple GPU, `-device mps`) or **CPU** (`-device cpu`). Each choice gives the same command on every computer, so a copied command does what it shows, also on a cluster. The device found on your computer is preselected (MPS on a Mac, otherwise GPU); **Info** (top right) shows what was found.

If **Info** shows `CPU only` although your computer has an NVIDIA GPU, PyTorch was installed without CUDA support (the default on Windows). Reinstall it as described in the [installation instructions](https://pytorch.org/get-started/locally/), or train on a remote machine (see [Work with a Remote Machine](setup.md#work-with-a-remote-machine-optional)).

### Trainer

The trainer defines how the model is trained. Which one to use depends on your labels:

- **Dense labels** (nnU-Net Workflow): nnU-Net's default `nnUNetTrainer`, for complete labels.
- **Sparse labels** (Annotation Workflow): **`nnUNetTrainerBetterIgnoreSampling`**, which is bundled with nnWorkbench and made for sparse labels. Dense labels created in the Annotation Workflow use `nnUNetTrainer`. During training, nnU-Net cuts small blocks (patches) out of your images. This trainer places the annotated structures at varying positions within these patches instead of always centering them, which gives the model more varied examples and reduces errors near patch borders.

The `nnUNetTrainerBetterIgnoreSampling` variants below differ only in *data augmentation*: random changes such as rotations, scaling or mirroring that nnU-Net applies to the images during training so the model sees more variation. Keep the default unless you have a reason to change it:

- `nnUNetTrainerBetterIgnoreSampling`: recommended default.
- `nnUNetTrainerBetterIgnoreSamplingDA5`: stronger augmentation (nnU-Net's DA5 policy).
- `nnUNetTrainerBetterIgnoreSamplingDA5ord0`: DA5 with order-0 interpolation of the augmented images.
- `nnUNetTrainerBetterIgnoreSamplingOnlyMirror01`: mirroring only along axes 0 and 1.
- `nnUNetTrainerBetterIgnoreSamplingNoMirroring`: no mirroring, e.g. if left and right must not be swapped.
- `nnUNetTrainerBetterIgnoreSampling_10epochs`: short test run to check that everything works.

### Folds

By default, nnU-Net uses *5-fold cross-validation*: the training images are split into five parts, and five models (folds 0–4) are trained. Each fold is trained on four parts and tested on the remaining one, so every image is used for validation exactly once (see [5. Validate](5-validate.md)). For prediction, nnU-Net combines all five folds by default.

Alternatively, fold `all` trains a single model on all images. Its validation predictions are made on the training images, so they are only useful for inspection, not for scores (see [5. Validate](5-validate.md)).

**References:**

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
- Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., & Jaeger, P. F. (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. MICCAI 2024. (the residual encoder planners)

---

[← 3. Annotate](3-annotate.md) | [Overview](README.md) | [5. Validate →](5-validate.md)
