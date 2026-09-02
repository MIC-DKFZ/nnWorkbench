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

## Installation

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

### Local installation — full workbench

GUI, Napari inspection, and optional nnInteractive annotation workflow.

```bash
pip install -e '.[local]'
nnworkbench-gui
```

Local interactive inference additionally needs a compatible PyTorch installation.

### Remote installation — lightweight execution

For clusters or remote machines: nnU-Net, custom trainers, and remote commands only—no GUI, Napari or nnInteractive.

```bash
pip install -e .
```

Save the nnU-Net roots on the remote machine once. `nnworkbench-run` then uses them for nnU-Net commands and dataset tools:

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw --preprocessed /path/to/nnUNet_preprocessed --results /path/to/nnUNet_results
```

You can also create a Remote profile in the GUI and use **Export** to copy this command with its configured paths.

## Quick start

### Set up paths

Start the GUI:

```bash
nnworkbench-gui
```

Then select the directories containing your `DatasetNNN_Name` folders for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. For remote or terminal use, save the same paths with `nnworkbench-set-paths` as shown above.

### nnU-Net Workflow

1. **Prepare and inspect:** Select an existing dataset—or create one from the Raw root—then review its `dataset.json`. Use **Inspect** beside a Raw image to open the case in Napari with its available matching files.
2. **Preprocess:** In **Raw**, choose **Run Preprocessing**, select a configuration and planner, then run the command.
3. **Train:** In **Preprocessed**, choose **Run Training**, select the plans file, configuration, trainer, and fold, then run the command.
4. **Predict:** In **Results**, choose **Run Predict**, select a trained plan/configuration and the input/output directories, then run the command.

Read the full [nnU-Net workflow guide](docs/nnunet-workflow.md).

### Annotation Workflow

1. **Enable the workflow:** Turn on **Annotation workflow** in the header and select a Raw dataset with `imagesTr`.
2. **Prepare label IDs:** In Raw `dataset.json`, add an integer `ignore` label as the highest label ID. The editor highlights this requirement while Annotation Workflow is enabled.
3. **Annotate in Napari:** In `imagesTr`, choose **Annotate** for a new case; choose **Edit** in `annotationsTr` to continue an existing annotation. This opens Napari directly with nnInteractive and a writable semantic annotation layer.
4. **Save:** Use **Save annotation** in Napari to write the semantic map to `annotationsTr`.
5. **Convert:** Back in **Raw**, choose **Convert Annotations** to create matching nnU-Net labels in `labelsTr`.
6. **Train a model:** Continue with **Run Preprocessing**, **Run Training**, and **Run Predict** as described in the nnU-Net workflow above.

Read the full [Annotation Workflow guide](docs/annotation-workflow.md), including label-ID mapping and conversion requirements.

## CLI overview

After configuring the paths with `nnworkbench-set-paths`, use `nnworkbench-run` to run any nnU-Net command with those saved paths:

```bash
nnworkbench-run nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity
nnworkbench-run nnUNetv2_train 1 3d_fullres 0
nnworkbench-run nnUNetv2_predict -i /path/to/input -o /path/to/output -d 1 -c 3d_fullres
```

It also provides the following dataset tools:

```bash
# Analyse labelsTr and save class_distribution.json (dataset ID)
nnworkbench-run label-analysis 1 --workers 4

# Convert annotationsTr semantic maps into labelsTr training labels
nnworkbench-run convert-annotations 1 --workers 4
```

Both commands resolve the dataset under the saved `nnUNet_raw` path. Pass its numeric ID or full `DatasetNNN_Name` directory name. Annotation conversion requires a valid final `ignore` label in the dataset's `dataset.json`. Run `nnworkbench-get-paths` to view the saved path configuration.

## Further documentation

- [nnU-Net workflow guide](docs/nnunet-workflow.md)
- [Annotation Workflow guide](docs/annotation-workflow.md)
- [GUI documentation overview](docs/GUI.md)


# Acknowledgements

<img src="src/nnworkbench/assets/HI_Logo.svg" alt="Helmholtz Imaging" height="40"><br>
<img src="src/nnworkbench/assets/DKFZ_Logo.svg" alt="German Cancer Research Center (DKFZ)" height="38">&nbsp;&nbsp;&nbsp;



This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
