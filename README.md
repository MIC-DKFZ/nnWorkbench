# nnWorkbench

> A GUI for nnU-Net workflows and nnInteractive-powered annotation.
> - **nnU-Net Workflow:** Train, validate and apply nnU-Net on images that already have labels.
> - **Annotation Workflow:** Extends the nnU-Net Workflow for images without labels: create the labels efficiently with nnInteractive and scribbles, then train, validate and improve the model in iterative annotation rounds.


From the first image to a trained segmentation model, nnWorkbench keeps the entire workflow in one place. It provides researchers and domain scientists with a clear workspace to inspect images in Napari, prepare or create labels, and guide an nnU-Net model from dataset preparation to prediction.

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#cli-overview">CLI</a> ·
  <a href="#qa">Q&amp;A</a> ·
  <a href="src/nnworkbench/assets/annotation_workflow/README.md">Guide</a> ·
  <a href="#license">License</a>
</p>

## Why nnWorkbench

- **All projects in one place:** Browse all configured nnU-Net datasets and keep each dataset’s images, labels, preprocessing, trained models, and validation results connected.
- **Guided nnU-Net from data to prediction:** Each dataset follows the nnU-Net stages Data, Preprocess, Train, Validate and Predict. Run them through short dialogs with sensible defaults, or copy the exact commands for a terminal or a cluster.
- **See what is ready and what needs attention:** Coloured stage tabs and status badges show what is done, what can run next and what must be fixed first, for example an incomplete `dataset.json` before preprocessing.
- **Inspect at every stage:** Open Raw images and labels, predictions, and validation results in Napari, with matching files loaded together whenever available.
- **Connect nnInteractive and nnU-Net:** Use nnInteractive in Napari to create annotations, save them directly as training labels in `labelsTr`, and continue with nnU-Net preprocessing and training.
- **Annotation Workflow:** Follow a documented, guided workflow designed to create high-quality labels that support model performance while minimizing manual annotation effort.

![Annotation Workflow](src/nnworkbench/assets/images/annotation-workflow-header.png)

# Installation

**Requirements:**

- **Python 3.12**
- Windows 10/11, macOS 13 or newer, or Linux with glibc 2.35 or newer (e.g. Ubuntu 22.04+)
- Conda or another Python environment manager

**From PyPI** *(coming soon: nnWorkbench is not on PyPI yet, so this does not work at the moment)*:

```bash
conda create -n nnworkbench python=3.12
conda activate nnworkbench
pip install "nnworkbench[all]"
nnworkbench-gui
```

**From source** (works now):

```bash
conda create -n nnworkbench python=3.12
conda activate nnworkbench
git clone <your-repository-url> nnWorkbench
cd nnWorkbench
pip install -e ".[all]"
nnworkbench-gui
```

Depending on your system, you may need to install PyTorch manually before installing nnWorkbench, as described on [pytorch.org](https://pytorch.org/get-started/locally/). This is required on Windows to use an NVIDIA GPU.
For a lightweight installation without the GUI on a remote machine, and for notes on using nnWorkbench on macOS, see the instructions below.


<details>
<summary><b>Windows</b></summary>

pip installs a CPU-only PyTorch on Windows, so an NVIDIA GPU would not be used. Install the CUDA build of PyTorch **before** nnWorkbench, in the activated environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install nnWorkbench from PyPI *(coming soon: does not work at the moment)*:

```bash
pip install "nnworkbench[all]"
```

or from source (works now, inside the cloned repository):

```bash
pip install -e ".[all]"
```

The right command for your CUDA version is on [pytorch.org](https://pytorch.org/get-started/locally/). Everything works in PowerShell and the Windows command prompt.

</details>

<details>
<summary><b>macOS</b></summary>

nnWorkbench itself runs fully on macOS (13 or newer). For the GPU-heavy parts, it relies on nnU-Net and nnInteractive, which differ on Apple silicon:

- **nnU-Net (training and prediction):** runs on the Apple GPU with `-device mps` (choose **MPS** under **Device** in the Train and Predict dialogs). It works, but is much slower than an NVIDIA GPU, and the GPU shares its memory with the system: prefer `nnUNetPlannerResEncM` or `ExperimentPlanner` over `nnUNetPlannerResEncL`. A Mac is fine for predicting and short test runs; train on a GPU machine or cluster.
- **nnInteractive (annotation):** the napari-nninteractive plugin uses only CUDA GPUs, so on a Mac it runs on the CPU, which is slow. For a smooth experience, connect the plugin to an nnInteractive server on a GPU machine (see the [napari-nninteractive documentation](https://github.com/MIC-DKFZ/napari-nninteractive)).

</details>

<details>
<summary><b>Remote machines and clusters</b></summary>

On a cluster, the core installation is enough: nnU-Net, the bundled trainers and the command-line tools, without the nnWorkbench GUI, Napari or nnInteractive.

```bash
# from PyPI (coming soon, does not work at the moment)
pip install nnworkbench

# from source (works now, inside the cloned repository)
pip install -e .
```

Save the nnU-Net folders on the remote machine once: 

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw --preprocessed /path/to/nnUNet_preprocessed --results /path/to/nnUNet_results
```

`nnworkbench-run` then uses them for nnU-Net commands and dataset tools. 
Optionally, you can prepare this in the nnWorkbench GUI: on the overview, add a remote path profile with the folders of the remote machine; **Export** then copies the `nnworkbench-set-paths` command above with these paths. The commands themselves can also be configured in the GUI: **Copy command** in the run dialogs gives the matching `nnworkbench-run` command, which uses the paths saved on the remote machine. All commands and dataset tools for the remote machine are listed in the [CLI overview](#cli-overview).

</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **`No matching distribution found for mitk-python`:** the environment does not use Python 3.12, or the system is too old (macOS below 13, Linux with glibc below 2.35). Create the environment with `python=3.12`; on older Linux systems, use the core installation (`pip install -e .`, later `pip install nnworkbench`) without the GUI.
- **Info shows *CPU only* although there is an NVIDIA GPU:** PyTorch was installed without CUDA support or does not match the driver; Info shows the cause and the fix.
- **Qt 5 instead of Qt 6:** install the GUI without a Qt binding and add PyQt5: `pip install -e ".[gui]" PyQt5` (from PyPI, coming soon: `pip install "nnworkbench[gui]" PyQt5`).

</details>

# Quick start

Start the GUI with `nnworkbench-gui`.

> *[Screenshot placeholder: the start screen with the three nnU-Net folders and the dataset list. To be added with dummy data.]*

1. **Set your folders.** On the start screen, select three folders: `nnUNet_raw` for your datasets (images, labels and `dataset.json`), `nnUNet_preprocessed` for the preprocessed data used in training, and `nnUNet_results` for the trained models and their validation results. nnU-Net fills the last two. Empty folders are fine. nnWorkbench lists every dataset it finds in them.
2. **Create a dataset or select an existing one.** Click **＋ New dataset**, or open a dataset from the list. Put your images into `imagesTr` (and, if you have them, your labels into `labelsTr`), and describe the classes with **Edit** next to `dataset.json`. Red badges on the **Data** page tell you what is still missing.
3. **Annotate** *(Annotation Workflow only)*. Turn on **Annotation Workflow** in the top bar. On the **Data** page, click the pencil next to an image, annotate in Napari and click **Save annotation**.
4. **Preprocess.** On **Preprocess**, click **▶ Preprocess…** and then **▶ Preprocess**, or **Copy command** to run it on a remote machine. The defaults are a good start.
5. **Train.** On **Train**, click **▶ Train…**, choose the folds and click **▶ Train**, or **Copy command** to run it on a remote machine. The console at the bottom shows the progress.
6. **Validate.** On **Validate**, click a fold of your model to see its Dice score per case, sort by **Dice** to see the weakest cases first, and open any case in Napari to compare the prediction with your label. *Annotation Workflow:* for another round, select the dataset on the start screen, click **＋ New iteration** and continue with step 3.
7. **Predict.** On **Predict**, choose a folder of new images and an output folder and click **▶ Predict**, or **Copy command** to run it on a remote machine.

> *[Screenshot placeholder: the Data page of a dataset with sparse labels: cases, dataset.json and the how-to. To be added with dummy data.]*

Every page has a short how-to on the left, and **? Guide** in the top bar opens the full [nnWorkbench guide](src/nnworkbench/assets/annotation_workflow/README.md). It explains both workflows step by step: how to understand your images, define classes, annotate efficiently, read the validation results, and decide when a model is good enough.

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
# Show what is done and what is next for a dataset (the stage states of the GUI; add --json for scripts)
nnworkbench-run status -d 42

# Open a dataset's images and labels in Napari, starting at the 12th image
nnworkbench-run inspect -d 42 -i 12

# Compute statistics (as on the Data page): classes per image and voxels per class, from labelsTr
nnworkbench-run compute-statistics -d 42 --workers 4


# Stack naturally ordered 2D slices (e.g. TIFF or PNG) into one 3D NIfTI volume
nnworkbench-run stack-images /path/to/image_slices --file-type .tif --spacing 1 1 1

# Remove everything a trained model folder does not need for prediction, e.g. before sharing (in place, cannot be undone)
nnworkbench-run reduce-checkpoints /path/to/nnUNet_results/Dataset042_CellSegmentation/MODEL_FOLDER
```

`-d` (or `--dataset`) specifies the numeric nnU-Net dataset ID under the saved `nnUNet_raw` path. For example, use `-d 42` for `Dataset042_CellSegmentation`. Run `nnworkbench-get-paths` to view the saved path configuration, and add `--help` to any tool to see all its options.

## Q&A

**Do I need a GPU?**
For training, an NVIDIA GPU is strongly recommended; on the CPU, training takes days per fold. Everything else (managing datasets, annotating, inspecting, validating, predicting) also works without one. **Info** (top right in the GUI) shows which device was found.

**Can I train on a cluster and keep working in the nnWorkbench GUI on my computer?**
Yes. Use the core installation on the cluster (see *Remote machines and clusters* above) and copy the commands from the GUI (see the [CLI overview](#cli-overview)). If the cluster's `nnUNet_results` folder is mounted on your computer, the GUI shows the training progress and the validation results as if they ran locally.

**Where does my data have to be for a remote setup?**
Each step needs the output of the previous one, in the nnU-Net folders saved on the machine that runs it:

| Step | Needs | Writes |
| --- | --- | --- |
| Preprocessing | `nnUNet_raw` (images, labels, `dataset.json`) | `nnUNet_preprocessed` |
| Training | `nnUNet_preprocessed` | `nnUNet_results` |
| Prediction | `nnUNet_results` (the trained model) and the images to predict | the output folder you choose |

On your computer, the GUI uses `nnUNet_raw` for annotating and inspecting, and `nnUNet_results` for validation. The simplest setup is storage that both machines can access (e.g. a mounted network drive). Otherwise, copy `nnUNet_raw` to the remote machine after each annotation round, preprocess and train there, and copy `nnUNet_results` back.

**Can I annotate with nnInteractive without a local GPU?**
Yes. nnInteractive can run on a server with a GPU, while you annotate in Napari on your computer. Set up the server and connect the plugin as described in the [napari-nninteractive documentation](https://github.com/MIC-DKFZ/napari-nninteractive). Without a GPU or a server, nnInteractive runs on the CPU, which is slow (on a Mac always, see *macOS* above).

## Further documentation

- [nnWorkbench guide](src/nnworkbench/assets/annotation_workflow/README.md): the nnU-Net Workflow and the Annotation Workflow step by step; also shown in the GUI via **? Guide** in the top bar.


# License

nnWorkbench is licensed under the [Apache License 2.0](../../LICENSE).

> **Note:** The [nnInteractive model weights](https://huggingface.co/MIC-DKFZ/nnInteractive), downloaded on first use
> for interactive annotation, are licensed under
> [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): **non-commercial use only**. nnU-Net and the
> rest of nnWorkbench are not affected.
> 
# Acknowledgements

<img src="src/nnworkbench/assets/images/helmholtz-imaging-logo.svg" alt="Helmholtz Imaging" height="40"><br>
<img src="src/nnworkbench/assets/images/dkfz-logo.svg" alt="German Cancer Research Center (DKFZ)" height="38">&nbsp;&nbsp;&nbsp;



This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
