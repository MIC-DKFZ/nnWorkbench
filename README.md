# nnWorkbench

> A GUI for nnU-Net workflows and nnInteractive-powered annotation

## Your workflow, one workbench

nnWorkbench brings nnU-Net and nnInteractive together in one desktop app. Start with images and labels to train a model, or start with images only and create the labels you need in Napari. From the first image to a trained model ready for inference, your workflow stays connected.

The nnWorkbench GUI supports **Windows, macOS, and Ubuntu**. Its own interface, file browsing, settings, documentation viewer, and background loading are platform-neutral; availability of nnU-Net, nnInteractive, MITK-Python, and their hardware dependencies is determined by those projects and your environment.

## Setup

- Python 3.12 or newer
- nn-U-Net data roots for the workflows you intend to use

### Optional: create a Conda environment

If you do not already have a suitable Python environment, create one with Conda (for example, [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda):

```bash
conda create -n nnworkbench python=3.12
conda activate nnworkbench
```

### Install and start

Install nnWorkbench and its nnU-Net, Napari, MITK-Python, and local nnInteractive dependencies:

```bash
git clone <your-repository-url> nnWorkbench
cd nnWorkbench
pip install -e '.[dev]'
nnworkbench-gui
```

`nnworkbench-gui` is the recommended way to start the installed application. Local nnInteractive inference additionally requires a compatible PyTorch installation and an nnInteractive model; configure these only if you plan to use interactive annotation.

For development, you can alternatively launch the application directly from the repository:

```bash
python -m nnworkbench.app
```

Press `Ctrl+C` in the launching terminal to close the GUI and its child Napari windows.

## Configure data roots

Select the directories that contain your `DatasetNNN_Name` folders:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

The roots are independent. nnWorkbench works with any available subset and remembers the paths locally. The **Open folder** button opens the selected dataset folder for the active Raw, Preprocessed, or Results tab.

## Annotation workflow

In the **Raw** tab, choose **Annotate** for an image or **Edit** for an existing `annotationsTr` map. Napari opens with nnInteractive, semantic-map aggregation enabled, and nnWorkbench’s saving controls above the nnInteractive controls.

An annotation session always has a writable semantic layer named `semantic map - <image-layer-name>`. A missing annotation starts as an empty map with the source image’s shape and geometry. Saving writes it to `annotationsTr` using the nnU-Net case name—for example, `Case_001_0000.nii.gz` saves as `annotationsTr/Case_001.nii.gz`.

## Convert annotations and analyze labels

**Convert Annotations** turns NIfTI semantic maps in `annotationsTr` into training labels in `labelsTr`. The Raw `dataset.json` must contain an integer `ignore` label with the highest ID. During conversion:

- annotation value `0` becomes the nnU-Net ignore ID;
- positive IDs are shifted down by one;
- valid output replaces the matching `labelsTr` file;
- empty or out-of-range maps are skipped and reported without stopping the batch.

Conversion and **Analyze Labels** run in the background using the shared worker-process count configured in Raw workspace settings. Analyze Labels stores `class_distribution.json` in the raw dataset folder; select it in the Raw tree to view the saved table and plots.

## nnU-Net commands

The GUI runs `nnUNetv2_*` commands asynchronously and streams their output to its console and the launching terminal. It exports only the environment roots needed by each operation:

- preprocessing: `nnUNet_raw`, `nnUNet_preprocessed`
- training: `nnUNet_preprocessed`, `nnUNet_results`
- prediction: `nnUNet_results`

Only one nnU-Net command can run at a time. The console includes a terminate action while a command is active.

In the **Results** tab, each trained fold exposes a collapsed `validation` folder. Use **Inspect** beside a validation NIfTI prediction to open it in Napari. When the matching Raw dataset is available, nnWorkbench also loads the corresponding training image and ground-truth label for comparison; otherwise it opens the validation prediction on its own.

## Dataset layout

```text
nnUNet_raw/
└── Dataset001_MyDataset/
    ├── imagesTr/        # image cases, e.g. Case_001_0000.nii.gz
    ├── labelsTr/        # nnU-Net training labels, e.g. Case_001.nii.gz
    ├── annotationsTr/   # nnInteractive semantic annotations (optional)
    ├── imagesTs/        # optional test images
    └── dataset.json
```

Only reviewed labels should be used for training. Validate geometry and class definitions before preprocessing or training.

## Optional CLI

`nnworkbench` provides `init`, `status`, `next`, `accept`, and `run` commands for configuration-driven workflows. Copy [config/example.yaml](config/example.yaml) to `config/local.yaml` and adapt it before using these commands.

## Development

```bash
pytest -q
ruff check .
```

See [docs/GUI.md](docs/GUI.md) for the GUI reference and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.
