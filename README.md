# nnWorkbench

> A GUI for nnU-Net workflows and nnInteractive-powered annotation.
> - **nnU-Net Workflow:** Train, validate and apply nnU-Net on images that already have labels.
> - **Annotation Workflow:** Extends the nnU-Net Workflow for images without labels: create the labels efficiently with nnInteractive and scribbles, then train, validate and improve the model in iterative annotation rounds.


From the first image to a trained segmentation model, nnWorkbench keeps the entire workflow in one place. It provides researchers and domain scientists with a clear workspace to inspect images in Napari, prepare or create labels, and guide an nnU-Net model from dataset preparation to prediction.

## Why nnWorkbench

- **All projects in one place:** Browse all configured nnU-Net datasets and keep each dataset’s images, labels, preprocessing, trained models, and validation results connected.
- **Guided nnU-Net from data to prediction:** Plan, preprocess, train, and predict through clear dialogs, or copy the exact commands when you prefer the command line.
- **Inspect at every stage:** Open Raw images and labels, annotations, predictions, and validation results in Napari, with matching files loaded together whenever available.
- **Connect nnInteractive and nnU-Net:** Use nnInteractive in Napari to create annotations, convert them into training labels, and continue directly to nnU-Net preprocessing and training.
- **Annotation Workflow:** Follow a documented, guided workflow designed to create high-quality labels that support model performance while minimizing manual annotation effort.

![Annotation Workflow](src/nnworkbench/assets/AnnotationWorkflowHeader.png)

# Installation

Requirements:

- Python 3.12 or newer
- Conda or another suitable Python environment manager

![nnWorkbench local and remote installation overview](src/nnworkbench/assets/AnnotationWorkflowInstall.png)

Create an environment and clone the repository:

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

For clusters or remote machines: nnU-Net, custom trainers, and remote commands only—no GUI, Napari or nnInteractive. To run nnInteractive on a remote GPU instead, see the [napari-nninteractive documentation](https://github.com/MIC-DKFZ/napari-nninteractive).

```bash
pip install -e .
```

Save the nnU-Net roots on the remote machine once. `nnworkbench-run` then uses them for nnU-Net commands and dataset tools:

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw --preprocessed /path/to/nnUNet_preprocessed --results /path/to/nnUNet_results
```

You can also create a Remote profile in the GUI and use **Export** to copy this command with its configured paths.

# Quick start

nnWorkbench supports two workflows. The **nnU-Net Workflow** trains and applies nnU-Net on images that already have labels. The **Annotation Workflow** extends it for images without labels: you create the labels first, then continue with the same steps.

Start the GUI with `nnworkbench-gui`.

## nnU-Net Workflow

1. **Set the paths:** In the header, select the folders for `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results`. The three folders are independent: nnWorkbench combines the datasets found across all of them. For remote or terminal use, save the same paths with `nnworkbench-set-paths` as shown above.
2. **Create a dataset:** Click **New dataset**, or select an existing one. Copy your images into `imagesTr` and your labels into `labelsTr`, and fill in `dataset.json` in the editor on the right.
3. **Preprocess:** In **Raw**, click **Run Preprocessing** (`nnUNetPlannerResEncL` is preselected).
4. **Train:** In **Preprocessed**, click **Run Training** (`nnUNetTrainer` is preselected).
5. **Validate:** In **Results**, expand a fold's `validation` folder and click **Inspect** to compare the predictions with your labels.
6. **Predict:** In **Results**, click **Run Predict**, then **Inspect predictions**.

## Annotation Workflow

The Annotation Workflow extends the nnU-Net Workflow: it adds the annotation steps and uses a trainer made for sparse labels. Steps marked *(as in the nnU-Net Workflow)* are unchanged.

1. **Set the paths** *(as in the nnU-Net Workflow)*.
2. **Create a dataset:** Turn on the **Annotation workflow** switch in the header first, then click **New dataset**, so that `annotationsTr` is created too. Copy only your images into `imagesTr`, and add an `ignore` class with the highest ID to `dataset.json`.
3. **Annotate:** In **Raw**, click **Annotate** next to an image in `imagesTr` (or **Edit** in `annotationsTr`), annotate in Napari and click **Save annotation**.
4. **Convert:** In **Raw**, click **Convert Annotations** to create the training labels in `labelsTr`.
5. **Preprocess** *(as in the nnU-Net Workflow)*.
6. **Train:** In **Preprocessed**, click **Run Training** (`nnUNetTrainerBetterIgnoreSampling` is preselected).
7. **Validate:** Inspect the predictions as in the nnU-Net Workflow, including the unlabeled regions. For another annotation round, click **Create next iteration** in **Raw** and continue with step 3 in the new dataset.
8. **Predict** *(as in the nnU-Net Workflow)*.

The [nnWorkbench guide](src/nnworkbench/assets/annotation_workflow/README.md) explains both workflows step by step, with command-line alternatives; its overview shows which stages each workflow uses. It is also shown in the GUI via the **ⓘ** button next to the Annotation workflow switch.

## CLI overview

nnWorkbench can be used entirely through the GUI. If you prefer not to use the GUI, or are working on a remote system, the same workflows are available from the command line.

After configuring the paths with `nnworkbench-set-paths`, use `nnworkbench-run` to run any nnU-Net command with those saved paths:

```bash
nnworkbench-run nnUNetv2_plan_and_preprocess -d 42 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity
nnworkbench-run nnUNetv2_train 42 3d_fullres 0 -p nnUNetResEncUNetLPlans
nnworkbench-run nnUNetv2_predict -i /path/to/input -o /path/to/output -d 42 -c 3d_fullres -p nnUNetResEncUNetLPlans
```

These use the recommended `nnUNetPlannerResEncL`, which the GUI also preselects. For sparse annotations from the Annotation Workflow, add `-tr nnUNetTrainerBetterIgnoreSampling` to training and prediction.

It also provides the following dataset tools:

```bash
# Open a dataset's images, labels and annotations in Napari, starting at the 12th image
nnworkbench-run inspect -d 42 -i 12

# Analyse labelsTr and save class_distribution.json
nnworkbench-run label-analysis -d 42 --workers 4

# Convert annotationsTr semantic maps into labelsTr training labels
nnworkbench-run convert-annotations -d 42 --workers 4

# Stack naturally ordered 2D slices (e.g. TIFF or PNG) into one 3D NIfTI volume
nnworkbench-run stack-images /path/to/image_slices --file-type .tif --spacing 1 1 1

# Remove everything a trained model folder does not need for prediction, e.g. before sharing (in place, cannot be undone)
nnworkbench-run reduce-checkpoints /path/to/nnUNet_results/Dataset042_CellSegmentation/MODEL_FOLDER
```

`-d` (or `--dataset`) specifies the numeric nnU-Net dataset ID under the saved `nnUNet_raw` path. For example, use `-d 42` for `Dataset042_CellSegmentation`. Annotation conversion requires a valid final `ignore` label in the dataset's `dataset.json`. Run `nnworkbench-get-paths` to view the saved path configuration, and add `--help` to any tool to see all its options.

## Further documentation

- [nnWorkbench guide](src/nnworkbench/assets/annotation_workflow/README.md): the nnU-Net Workflow and the Annotation Workflow step by step; also shown in the GUI via the **ⓘ** button.


# Acknowledgements

<img src="src/nnworkbench/assets/HI_Logo.svg" alt="Helmholtz Imaging" height="40"><br>
<img src="src/nnworkbench/assets/DKFZ_Logo.svg" alt="German Cancer Research Center (DKFZ)" height="38">&nbsp;&nbsp;&nbsp;



This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
