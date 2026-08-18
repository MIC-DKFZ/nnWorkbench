# Running the desktop application

Install the package and start the application:

```bash
pip install -e '.[dev]'
nnworkbench-gui
```

The three directory selectors accept the directories that directly contain the `DatasetNNN_Name` folders:

- the `nnUNet_raw` root;
- the `nnUNet_preprocessed` root; and/or
- the `nnUNet_results` root.

They are independent: the application is useful even if only one is selected. Directory choices are remembered locally using Qt settings.

The dataset list combines entries across all chosen roots. The selected dataset header states which of Raw, Preprocessed, and Results is available; the active tab determines which dataset folder **Open folder** opens.

The middle pane has three read-only tabs for the selected dataset:

- **Raw** converts nnInteractive semantic annotations into `labelsTr`, analyzes labels, and runs preprocessing. Conversion requires an `ignore` entry with the highest numeric value in the Raw `dataset.json`; it maps annotation `0` to this ignore value and shifts positive class IDs down by one. Empty or invalid annotation maps are skipped and reported while valid maps continue to convert. Both conversion and label analysis use the shared worker-process count in Raw workspace settings. The preprocessing dialog offers `2d`, `3d_fullres`, `3d_lowres`, and `3d_cascade_fullres`, with `ExperimentPlanner` as the default planner. Raw initially shows only collapsed `imagesTr`, `labelsTr`, and `annotationsTr` folder counts; file rows are created in small batches only after expanding a folder. Image files have Inspect and Annotate actions, annotation files have Inspect and Edit actions, and label files have Inspect.
- **Preprocessed** provides a collapsed tree of its subfolders and files. Select any JSON file to view its parsed contents alongside the tree. Its Run Training dialog lets you select a discovered plans file and one of its configurations, then supply a trainer (default `nnUNetTrainer`) and a fold number or `all`.
- **Results** shows a collapsed trained-model/fold tree alongside a validation-summary viewer. Each fold with validation output contains a collapsed `validation` folder listing all its files. Select a fold to display its `validation/summary.json` when available. NIfTI validation predictions have an **Inspect** action; it opens the prediction in Napari and, when Raw data is available, also loads the matching `imagesTr` image and `labelsTr` ground truth for comparison. Run Predict collects input/output directories and plan/configuration choices from discovered trained models.

The Preprocessing, Training, and Prediction dialogs provide **Copy command**, which copies the assembled nnU-Net command to the clipboard. Confirming a dialog executes the selected nnU-Net command and streams its output to the console and launching terminal.

Each tab loads only when it is opened. Filesystem scans and JSON reads run in the background; a brief loading message is shown while content is being prepared, so selecting datasets and switching tabs stays responsive.

The console below the tabs appears when an nn-U-Net command starts and shows command output and environment exports while mirroring them to the launch terminal. Preprocessing requires Raw and Preprocessed roots, Training requires Preprocessed and Results, and Prediction requires Results. Only one command can run at a time; Run controls stay disabled until it exits. Use **Terminate command** to request graceful termination of the active process.

The middle pane warns when the number of `imagesTr` files does not equal the number of configured channels multiplied by the number of `labelsTr` files.

The metadata pane displays `dataset.json` from Raw first, then Preprocessed, then Results. Only Raw metadata is editable; saving atomically replaces the raw manifest without creating a backup. In a Raw manifest, `numTraining` is editable and receives a red border when it differs from the number of files in `labelsTr`.

Use **New dataset** after choosing a raw root. Creation checks every configured root, rejecting a numeric ID that is already used anywhere. It creates `imagesTr`, `labelsTr`, `imagesTs`, and a starter raw `dataset.json`; it never creates preprocessed or results folders.
