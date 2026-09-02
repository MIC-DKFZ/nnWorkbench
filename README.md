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

nnWorkbench’s GUI runs on **Windows, macOS, and Ubuntu**. Compatibility of nnU-Net, nnInteractive, MITK-Python, PyTorch, and their hardware requirements remains determined by those projects and your environment.

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

## Remote control

nnWorkbench distinguishes between **Local** and **Remote** path profiles. **Local** is the default and always supplies the paths used by the GUI to discover datasets, inspect data, and start local nnU-Net commands.

Use the `+` tab in the GUI to create a named **Remote** profile and enter its `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` paths. Remote profiles do not change the Local GUI workspace; they only describe another machine or cluster setup.

Select a Remote profile and click **Export** to copy an `nnworkbench-set-paths` command containing its paths. Paste and run that command on the remote machine after the lightweight installation. The remote setup is then stored there and can be used with `nnworkbench-run` in a shell script or scheduler job.

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw --preprocessed /path/to/nnUNet_preprocessed --results /path/to/nnUNet_results
nnworkbench-run nnUNetv2_train 1 3d_fullres 0
```

For development tools, install the optional test and lint dependencies:

```bash
pip install -e '.[local,dev]'
```

## Quick start: nnU-Net workflow

1. **Configure roots:** Choose the directories containing your `DatasetNNN_Name` folders: `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
2. **Prepare and inspect:** Select an existing dataset—or create one from the Raw root—then review its `dataset.json`. Use **Inspect** beside a Raw image to open the case in Napari with its available matching files.
3. **Preprocess:** In **Raw**, choose **Run Preprocessing**, select a configuration and planner, then run the command.
4. **Train:** In **Preprocessed**, choose **Run Training**, select the plans file, configuration, trainer, and fold, then run the command.
5. **Predict:** In **Results**, choose **Run Predict**, select a trained plan/configuration and the input/output directories, then run the command.

Read the full [nnU-Net workflow guide](docs/nnunet-workflow.md).

## Quick start: Annotation Workflow

1. **Enable the workflow:** Turn on **Annotation workflow** in the header and select a Raw dataset with `imagesTr`.
2. **Prepare label IDs:** In Raw `dataset.json`, add an integer `ignore` label as the highest label ID. The editor highlights this requirement while Annotation Workflow is enabled.
3. **Annotate in Napari:** In `imagesTr`, choose **Annotate** for a new case; choose **Edit** in `annotationsTr` to continue an existing annotation. This opens Napari directly with nnInteractive and a writable semantic annotation layer.
4. **Save:** Use **Save annotation** in Napari to write the semantic map to `annotationsTr`.
5. **Convert:** Back in **Raw**, choose **Convert Annotations** to create matching nnU-Net labels in `labelsTr`.
6. **Train a model:** Continue with **Run Preprocessing**, **Run Training**, and **Run Predict** as described in the nnU-Net workflow above.

Read the full [Annotation Workflow guide](docs/annotation-workflow.md), including label-ID mapping and conversion requirements.

## Further documentation

- [nnU-Net workflow guide](docs/nnunet-workflow.md)
- [Annotation Workflow guide](docs/annotation-workflow.md)
- [GUI documentation overview](docs/GUI.md)


# Acknowledgements

<img src="src/nnworkbench/assets/HI_Logo.svg" alt="Helmholtz Imaging" height="40"><br>
<img src="src/nnworkbench/assets/DKFZ_Logo.svg" alt="German Cancer Research Center (DKFZ)" height="38">&nbsp;&nbsp;&nbsp;



This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
