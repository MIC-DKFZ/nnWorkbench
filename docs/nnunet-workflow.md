# nnU-Net workflow

This guide is for users who already have nnU-Net training labels, or who only want to manage and run nnU-Net experiments.

## 1. Configure your roots

At the top of nnWorkbench, choose the roots that contain `DatasetNNN_Name` folders. You may configure only the roots needed for the current task:

- **Raw** for dataset metadata, images, labels, analysis, and preprocessing.
- **Preprocessed** for plan files and training.
- **Results** for trained models, validation output, and prediction.

The selected locations are remembered locally. The left dataset list merges matching datasets from all configured roots.

## 2. Prepare or review a Raw dataset

A typical Raw dataset has this layout:

```text
nnUNet_raw/
└── Dataset001_MyDataset/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/        # optional
    └── dataset.json
```

Use **New dataset** after selecting a Raw root to create a starter dataset. It creates `imagesTr`, `labelsTr`, `imagesTs`, and a Raw `dataset.json`. The new manifest starts with channel `0` and labels `background: 0` and an empty class at `1`, ready to be completed.

The right panel lets you edit a Raw manifest. `numTraining` is checked against the number of files in `labelsTr`; channels and labels are checked for continuous IDs and missing names. The middle pane also warns if the number of `imagesTr` files does not match the configured channel count multiplied by the number of label cases.

Expand Raw folders only when needed. Folder counts and file lists load lazily. Use **Inspect** to open a case in Napari; matching images and labels are loaded when available.

## 3. Analyze labels

Choose **Analyze Labels** in Raw to calculate, in the background, how many label images contain each class and the total voxel count for each class. Use the gear beside the action to set the shared worker-process count.

The result is saved as `class_distribution.json` in the Raw dataset directory. Select it in the Raw tree to view aggregate and per-image tables, or voxel/image-count plots. The optional background filter applies to tables and plots.

## 4. Plan and preprocess

Choose **Run Preprocessing** in Raw. Select a configuration and planner, or type a custom planner name. nnWorkbench discovers installed planners in the background but never blocks custom names.

Every generated preprocessing command includes `--verify_dataset_integrity`. Use **Additional arguments** for other valid nnU-Net options, for example `--np 8`. **Copy command** copies the exact command; **Run preprocessing** executes it with `nnUNet_raw` and `nnUNet_preprocessed` exported for that process.

## 5. Train

Open Preprocessed and choose **Run Training**. Select a discovered plans file and configuration, choose or type a trainer name, and enter a fold number or `all`. Installed trainers are discovered in the background; custom trainer names remain supported.

Use **Additional arguments** for flags passed directly to `nnUNetv2_train`. Training exports only `nnUNet_preprocessed` and `nnUNet_results`.

## 6. Inspect results and predict

Results shows trained models, their folds, and collapsed validation folders. Select a fold to view `validation/summary.json`. Use **Inspect** on a validation prediction to open it in Napari; when the Raw dataset is available, matching image and ground-truth label are loaded for comparison.

Choose **Run Predict** to select input/output directories and a plan/configuration from discovered trained models. **Additional arguments** are passed to `nnUNetv2_predict`; prediction exports only `nnUNet_results`.

## Commands and console

The preprocessing, training, and prediction dialogs all provide **Copy command** and execute the same displayed command. Their extra-argument fields accept normal shell-style arguments; invalid quoting is reported before copying or running.

Only one nnU-Net command runs at a time. Its output appears in the in-app console and the terminal that launched nnWorkbench. Use **Terminate command** to request graceful termination. The console can be hidden after the command exits.
