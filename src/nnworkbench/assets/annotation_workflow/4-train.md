# 4. Train

***Turn your labels into an automated segmentation model.***

## TL;DR

```bash
# Preprocess the dataset once
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres -pl nnUNetPlannerResEncL

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

Images end in a channel postfix, labels do not. Most images have one channel (`_0000`); further channels are stored as separate files (`_0001`, …) and listed in `channel_names` in `dataset.json`.

```text
imagesTr:  Case001_0000.nii.gz  (+ Case001_0001.nii.gz, … for more channels)
labelsTr:  Case001.nii.gz
```

**In nnWorkbench:** The **Data** page marks images without a label with a red ✗ and checks the file names and channels.

### 2. Inspect the Labels (Optional)

Open a few images together with their labels in a viewer such as Napari and check that the labels are correct and lie on the right structures. This opens the dataset in Napari, starting at the first image; use `-i` to start at another one:

```bash
nnworkbench-run inspect -d 42
```

**In nnWorkbench:** On the **Data** page, click the magnifier button (**Open in Napari**) next to a case. Napari opens the image together with its label.

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

**In nnWorkbench:** On the **Preprocess** page, click **▶ Preprocess…**, choose a planner and the configurations, then click **▶ Preprocess** (see [Planners and GPU Memory](#planners-and-gpu-memory)).

nnU-Net offers different model configurations. Two are relevant here:

- **`2d`:** The model looks at one 2D slice at a time. Use it for 2D images, and for 3D data with only few, thick slices.
- **`3d_fullres`:** The model looks at 3D blocks of the volume at full resolution, so it also sees the context above and below a slice. Use it for 3D data; this is usually the best choice.

### 5. Train the Model

The `nnworkbench-run` prefix uses the folder paths saved in [Setup](setup.md#set-the-folder-paths) and makes the bundled trainers available to nnU-Net.

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

**In nnWorkbench:** On the **Train** page, click **▶ Train…** and choose the plans (e.g. `nnUNetResEncUNetLPlans`), the configuration, the trainer and one fold, then click **▶ Train**. `nnUNetTrainer` is preselected; with an `ignore` label in `dataset.json`, `nnUNetTrainerBetterIgnoreSampling` is preselected instead.

### 6. Follow the Training

Training runs unattended and can take hours. nnU-Net writes a log and a `progress.png` plot of the training curves into each fold folder in `nnUNet_results`.

**In nnWorkbench:** The **Train** page lists every model with its folds and shows the current epoch and pseudo Dice of a running fold. The console at the bottom shows the training output.

## Options Explained

The steps above use the recommended settings. This section explains the choices behind them.

### Planners and GPU Memory

The planner decides how large the network is, and therefore how much GPU memory it needs and how long training takes. It is chosen during preprocessing (`-pl`) and creates a *plans* file that training and prediction then refer to (`-p`).

| Planner (`-pl`) | Plans (`-p`) | GPU memory | Training time* |
| --- | --- | --- | --- |
| `ExperimentPlanner` | `nnUNetPlans` | 8 GB | ~9 h |
| `nnUNetPlannerResEncM` | `nnUNetResEncUNetMPlans` | 9 GB | ~12 h |
| **`nnUNetPlannerResEncL`** *(recommended)* | **`nnUNetResEncUNetLPlans`** | 24 GB | ~35 h |
| `nnUNetPlannerResEncXL` | `nnUNetResEncUNetXLPlans` | 40 GB | ~66 h |

*\*Per fold on an NVIDIA A100; your times depend on the GPU and the data.*

We recommend **`nnUNetPlannerResEncL`**, which is used in all examples of this guide. If your GPU has less than 24 GB of memory, use `nnUNetPlannerResEncM` or `ExperimentPlanner` instead, and replace the plans name in the commands accordingly.

### Device (GPU or CPU)

nnU-Net trains and predicts on an NVIDIA GPU by default. On a Mac with Apple silicon it can use the Apple GPU (`-device mps`), and every computer can use the CPU (`-device cpu`), but training on the CPU takes days per fold. Training on the Apple GPU works, but is much slower than on an NVIDIA GPU, and the GPU shares its memory with the system: use `nnUNetPlannerResEncM` or `ExperimentPlanner` there.

**In nnWorkbench:** The Train and Predict dialogs have a **Device** choice: **GPU** (NVIDIA, no `-device`), **MPS** (the Apple GPU, `-device mps`) or **CPU** (`-device cpu`). Each choice gives the same command on every computer, so a copied command does what it shows, also on a cluster. The device found on your computer is preselected (MPS on a Mac, otherwise GPU); **Info** (top right) shows what was found.

If **Info** shows `CPU only` although your computer has an NVIDIA GPU, PyTorch was installed without CUDA support (the default on Windows). Reinstall it as described in the [installation instructions](https://pytorch.org/get-started/locally/), or train on a remote machine (see [Work with a Remote Machine](setup.md#work-with-a-remote-machine-optional)).

### Trainer

The trainer defines how the model is trained. Which one to use depends on your labels:

- **nnU-Net Workflow:** nnU-Net's default `nnUNetTrainer`, for complete labels.
- **Annotation Workflow:** **`nnUNetTrainerBetterIgnoreSampling`**, which is bundled with nnWorkbench and made for sparse labels. During training, nnU-Net cuts small blocks (patches) out of your images. This trainer places the annotated structures at varying positions within these patches instead of always centering them, which gives the model more varied examples and reduces errors near patch borders.

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

**Reference:** Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.

---

[← 3. Annotate](3-annotate.md) | [Overview](README.md) | [5. Validate →](5-validate.md)
