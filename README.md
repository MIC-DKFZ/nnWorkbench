# nnWorkbench

> A GUI for nnU-Net workflows and nnInteractive-powered annotation.

From the first image to a trained segmentation model, nnWorkbench keeps the entire workflow in one place. It provides researchers and domain scientists with a clear workspace to inspect images in Napari, prepare or create labels, and guide an nnU-Net model from dataset preparation to prediction.
Existing labels can be imported directly into an nnU-Net workflow, while unlabeled images can be annotated with nnInteractive before continuing to model training and inference.

## Why nnWorkbench

- **All projects in one place:** Browse all configured nnU-Net datasets and keep each dataset’s images, labels, preprocessing, trained models, and validation results connected.
- **Guided nnU-Net from data to prediction:** Plan, preprocess, train, and predict through clear dialogs, or copy the exact commands when you prefer the command line.
- **Inspect at every stage:** Open Raw images and labels, annotations, predictions, and validation results in Napari, with matching files loaded together whenever available.
- **Connect nnInteractive and nnU-Net:** Use nnInteractive in Napari to create annotations, convert them into training labels, and continue directly to nnU-Net preprocessing and training.
- **Annotation Workflow:** Follow a documented, guided workflow designed to create high-quality labels that support model performance while minimizing manual annotation effort.

![Annotation Workflow](/src/nnworkbench/assets/AnnotationWorkflowHeader.png)

# Installation

Requirements:

- Python 3.12 or newer
- Conda or another suitable Python environment manager

![nnWorkbench local and remote installation overview](src/nnworkbench/assets/AnnotationWorkflowInstall.png)

Create an environment and clone the repository once:

```bash
conda create -n nnworkbench python=3.12
conda activate nnworkbench
git clone <your-repository-url> nnWorkbench
cd nnWorkbench
```

## Local installation — full workbench

GUI, Napari inspection, and optional nnInteractive annotation workflow.

```bash
pip install -e '.[local]'
nnworkbench-gui
```

Local interactive inference additionally needs a compatible PyTorch installation.

## Remote installation — lightweight execution

For clusters or remote machines: nnU-Net, custom trainers, and remote commands only—no GUI, Napari or nnInteractive.

```bash
pip install -e .
```

Save the nnU-Net roots on the remote machine once. `nnworkbench-run` then uses them for nnU-Net commands and dataset tools:

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw --preprocessed /path/to/nnUNet_preprocessed --results /path/to/nnUNet_results
```

You can also create a Remote profile in the GUI and use **Export** to copy this command with its configured paths.

# Quick start

## Set up paths

Start the GUI:

```bash
nnworkbench-gui
```

Then select the directories containing your `DatasetNNN_Name` folders for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. For remote or terminal use, save the same paths with `nnworkbench-set-paths` as shown above.

## nnU-Net Workflow

1. **Prepare and inspect:** Select an existing dataset—or create one from the Raw root—then review its `dataset.json`. Use **Inspect** beside a Raw image to open the case in Napari with its available matching files.
2. **Preprocess:** In **Raw**, choose **Run Preprocessing**, select a configuration and planner, then run the command.
3. **Train:** In **Preprocessed**, choose **Run Training**, select the plans file, configuration, trainer, and fold, then run the command.
4. **Predict:** In **Results**, choose **Run Predict**, select a trained plan/configuration and the input/output directories, then run the command.

Read the full [nnU-Net workflow guide](docs/nnunet-workflow.md).

## Annotation Workflow

1. **Enable the workflow:** Turn on **Annotation workflow** in the header and select a Raw dataset with `imagesTr`.
2. **Prepare label IDs:** In Raw `dataset.json`, add an integer `ignore` label as the highest label ID. The editor highlights this requirement while Annotation Workflow is enabled.
3. **Annotate in Napari:** In `imagesTr`, choose **Annotate** for a new case; choose **Edit** in `annotationsTr` to continue an existing annotation. This opens Napari directly with nnInteractive and a writable semantic annotation layer.
4. **Save:** Use **Save annotation** in Napari to write the semantic map to `annotationsTr`.
5. **Convert:** Back in **Raw**, choose **Convert Annotations** to create matching nnU-Net labels in `labelsTr`.
6. **Train a model:** Continue with **Run Preprocessing**, **Run Training**, and **Run Predict** as described in the nnU-Net workflow above.

Read the full [Annotation Workflow guide](docs/annotation-workflow.md), including label-ID mapping and conversion requirements.

## CLI overview

nnWorkbench can be used entirely through the GUI. If you prefer not to use the GUI, or are working on a remote system, the same workflows are available from the command line.

After configuring the paths with `nnworkbench-set-paths`, use `nnworkbench-run` to run any nnU-Net command with those saved paths:

```bash
nnworkbench-run nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity
nnworkbench-run nnUNetv2_train 1 3d_fullres 0
nnworkbench-run nnUNetv2_predict -i /path/to/input -o /path/to/output -d 1 -c 3d_fullres
```

It also provides the following dataset tools:

```bash
# Analyse labelsTr and save class_distribution.json
nnworkbench-run label-analysis -d 1 --workers 4

# Convert annotationsTr semantic maps into labelsTr training labels
nnworkbench-run convert-annotations -d 1 --workers 4
```

`-d` (or `--dataset`) specifies the numeric nnU-Net dataset ID under the saved `nnUNet_raw` path. For example, use `-d 10` for `Dataset010_MyDataset`. Annotation conversion requires a valid final `ignore` label in the dataset's `dataset.json`. Run `nnworkbench-get-paths` to view the saved path configuration.

### Stack 2D image slices

Create a 3D NIfTI image from naturally ordered 2D slices in a MITK-readable format, such as TIFF or PNG. All selected files must have the same shape and data type. The default output is saved beside the input folder as `<folder-name>.nii.gz`, or `<folder-name>_scale<factor>.nii.gz` when downsampling.

```bash
nnworkbench-run stack-images /path/to/image_slices --file-type .png \
  --spacing 0.5 0.5 1.0 --scale-factor 2 --workers 4
```

`--spacing` uses X Y Z order and defaults to `1 1 1`. `--scale-factor` is an integer downsampling factor applied to all three volume axes. When `--spacing` is supplied, its written value is increased by the same factor; the default unit spacing is unchanged. Use `-o /path/to/output.nii.gz` to choose an output file and `--overwrite` to replace one. MITK/ITK's per-slice messages are hidden by default; pass `--show-mitk-output` for diagnostics.

### Reduce a trained model for sharing

Strip a trained model folder down to what inference needs, for every fold in place. Each fold's `checkpoint_final.pth` has its optimizer state removed, `checkpoint_best.pth` is deleted, and the `validation` folder is deleted.

```bash
nnworkbench-run reduce-checkpoints /path/to/nnUNet_results/Dataset001_.../nnUNetTrainer__nnUNetPlans__3d_fullres
```

Pass `--reset-trainer-name` to also rewrite each checkpoint's `trainer_name` to `nnUNetTrainer`, so a custom trainer class is not required to run inference. This is destructive and edits the model folder in place; keep a copy if you need the optimizer state, best checkpoint, or validation results later.

## Further documentation

- [nnU-Net workflow guide](docs/nnunet-workflow.md)
- [Annotation Workflow guide](docs/annotation-workflow.md)
- [GUI documentation overview](docs/GUI.md)


# Acknowledgements

<img src="src/nnworkbench/assets/HI_Logo.svg" alt="Helmholtz Imaging" height="40"><br>
<img src="src/nnworkbench/assets/DKFZ_Logo.svg" alt="German Cancer Research Center (DKFZ)" height="38">&nbsp;&nbsp;&nbsp;



This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
