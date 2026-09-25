# Setup

***Organize your data for a smooth and structured workflow.***

We use the standard nnU-Net folder structure to stay compatible with the existing nnU-Net workflow. This keeps the overall workflow simple, avoids redundant conversions or file movements, and helps to keep the project clear and reproducible. We need three main storage folders:

- **`nnUNet_raw/`**: This folder contains the original datasets in the format expected by nnU-Net: images, labels and, with the Annotation Workflow, your annotations.
- **`nnUNet_preprocessed/`**: This folder is filled automatically when running nnU-Net planning and preprocessing and is used during model training. You don't have to do anything here manually.
- **`nnUNet_results/`**: This folder stores trained models and training outputs.

## Set the Folder Paths

nnWorkbench needs to know where these three folders are. You only have to set them once; they are remembered.

Run once on the command line:

```bash
nnworkbench-set-paths --raw /path/to/nnUNet_raw \
    --preprocessed /path/to/nnUNet_preprocessed \
    --results /path/to/nnUNet_results
```

Check the saved paths with `nnworkbench-get-paths`. The GUI and the command line share these settings, so setting them in one place is enough on the same computer.

**In nnWorkbench:** In the header at the top, click **Browse…** next to `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` and select each folder.

To train on a remote machine or cluster, see [Work with a Remote Machine](#work-with-a-remote-machine-optional).

## Work with a Remote Machine (Optional)

If your computer has no suitable GPU, you can keep working in nnWorkbench on your computer (annotating, validating, inspecting) and run the GPU-heavy commands, training and prediction, on a remote machine or cluster.

**Set up the remote machine once:** Install the lightweight remote version of nnWorkbench there (see the nnWorkbench README) and save the remote folder paths:

```bash
nnworkbench-set-paths --raw /remote/path/to/nnUNet_raw \
    --preprocessed /remote/path/to/nnUNet_preprocessed \
    --results /remote/path/to/nnUNet_results
```

**In nnWorkbench:** Add a remote profile in the header (**+**), enter the remote paths, and click **Export** to copy this command.

**Run commands:** In the nnWorkbench dialogs, click **Copy command** and run the command in a terminal on the remote machine. The `nnworkbench-run` prefix makes it use the remote paths.

**Sync the folders:** If your computer and the remote machine do not share the same storage, copy the dataset's folders (e.g. `Dataset042_CellSegmentation`) before and after each command. Each command only needs some of them:

| Step | Needs | Creates |
| --- | --- | --- |
| Preprocess | `nnUNet_raw` | `nnUNet_preprocessed` |
| Train | `nnUNet_preprocessed` | `nnUNet_results` |
| Predict | `nnUNet_results` (the trained model) and the input folder | the output folder |

- Preprocessing needs no GPU. You can run it on your computer and copy only `nnUNet_preprocessed` to the remote machine; `nnUNet_raw` is not needed there for training.
- Copy `nnUNet_results` back to your computer after training, so you can review the results in [5. Validate](5-validate.md).
- Keep `nnUNet_raw` on your computer: annotating, converting and inspecting need it.

nnInteractive can also run on a remote GPU; see the [napari-nninteractive documentation](https://github.com/MIC-DKFZ/napari-nninteractive).

## Create a Dataset

Each dataset is a folder in `nnUNet_raw` named `DatasetXXX_DatasetName`, where XXX is a unique three-digit dataset ID, for example `Dataset042_CellSegmentation`. It contains the following subfolders and files:

```text
nnUNet_raw/
└── Dataset042_CellSegmentation/
    ├── imagesTr/                # Training images
    │   ├── Case001_0000.nii.gz  # _0000 = channel 0 (more channels: _0001, …)
    │   └── ...
    ├── annotationsTr/           # Annotation Workflow only:
    │   ├── Case001.nii.gz       # your manual annotations
    │   └── ...
    ├── labelsTr/                # Training labels, one per image
    │   ├── Case001.nii.gz
    │   └── ...
    ├── imagesTs/                # Optional: images to predict later
    └── dataset.json             # Describes channels, classes and file format
```

**Annotation Workflow:** The format that works best for manual annotation differs slightly from the format nnU-Net requires for training. Therefore, human-created annotations are stored in `annotationsTr` and converted before training. The converted labels are saved in `labelsTr`, which is the folder nnU-Net uses during training.

**nnU-Net Workflow:** Copy your labels directly into `labelsTr`; `annotationsTr` is not needed.

> **Note:** In the end, `imagesTr` and `labelsTr` (and `annotationsTr` in the Annotation Workflow) must contain the same cases. Each image, its corresponding annotation and label must also have the same filename. Images need a `_0000` postfix: it is the channel number (see [4. Train](4-train.md#1-check-the-filenames)).

> **Note:** Put only images with labels, or images you are going to annotate, into `imagesTr`. An image without label file makes the counts no longer match. Keep all other images in a separate folder, e.g. `imagesTs`, to predict them later in [6. Predict](6-predict.md).

> **Note:** We recommend storing images, annotations and labels as NIfTI files (`.nii.gz`). This format is supported by nnU-Net and by most nnInteractive integrations. But in general, other formats are also possible. See [Prepare Your Images](#prepare-your-images-optional) for supported formats and conversion.

Create the dataset folder with the subfolders `imagesTr` and `labelsTr` (plus `annotationsTr` for the Annotation Workflow) in `nnUNet_raw` yourself, copy your images (and, in the nnU-Net Workflow, your labels) into them, and add a `dataset.json` as described [below](#create-datasetjson). Make sure no other dataset uses the same ID.

**In nnWorkbench:** Click **New dataset** in the Datasets panel and enter a dataset ID and a name. nnWorkbench creates the correctly named folder with `imagesTr`, `labelsTr` and a `dataset.json` template, and warns you if the ID is already in use. For the Annotation Workflow, turn on the **Annotation workflow** switch in the header first, so that `annotationsTr` is created too. Copy your images (and, in the nnU-Net Workflow, your labels) into the folders, and use **Inspect** next to a file in **Raw** to open the case in Napari.

## Create dataset.json

The `dataset.json` file describes the channels, the classes and the file format; see the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson). For the Dataset042 example:

```json
{
    "channel_names": {
        "0": "fluorescence"
    },
    "labels": {
        "background": 0,
        "cell": 1,
        "nucleus": 2,
        "ignore": 3
    },
    "numTraining": 20,
    "file_ending": ".nii.gz"
}
```

- **`labels`:** Class IDs start at 0 for the background and are numbered continuously.
  - **Annotation Workflow:** Use the classes from [2. Define](2-define.md). The `ignore` class is required and must have the highest ID.
  - **nnU-Net Workflow:** Use the IDs in your label files. Add `ignore` only if some voxels are not labeled.
- **`numTraining`:** The number of cases in `labelsTr`.

**In nnWorkbench:** **New dataset** creates a `dataset.json` template. Fill it in with the editor on the right. It highlights missing names, non-continuous IDs and a `numTraining` that does not match `labelsTr`; with the **Annotation workflow** switch on, also a missing or misplaced `ignore` class. The workspace also warns if the number of images in `imagesTr` does not match the number of labels and channels.

## Prepare Your Images (Optional)

Skip this if your images are already in a supported format, one file per case:

- **3D:** `.nii.gz` / `.nii` (recommended), `.nrrd`, `.mha`, `.gipl`, `.tif` / `.tiff`
- **2D:** `.png`, `.bmp`, `.tif`

JPEG is not supported because its lossy compression alters the images.

**2D slices → 3D volume.** If your 3D data is a folder of 2D slices (e.g. TIFF or PNG), combine them into one NIfTI file:

```bash
nnworkbench-run stack-images /path/to/slices --file-type .tif --spacing 1.0 1.0 1.0
```

`--spacing` is the voxel size in X Y Z; use your real acquisition values. The result is saved next to the folder; rename it to `CaseXXX_0000.nii.gz` and move it into `imagesTr`.

**Custom preprocessing** such as cropping or downsampling can be done with your usual tools (e.g. Fiji/ImageJ or Python). Apply it identically to all training images and to every image you predict later. Do not normalize intensities; nnU-Net does this itself.

## Commands and Console

The preprocessing, training and prediction dialogs in nnWorkbench all offer **Copy command** and run exactly the displayed command. Use **Additional arguments** for further nnU-Net options, e.g. `--np 8`. Only one command runs at a time; its output appears in the console at the bottom and in the terminal that started nnWorkbench. Use **Terminate command** to stop it.

---

[← Overview](README.md) | [1. Describe →](1-describe.md) | nnU-Net Workflow: [4. Train →](4-train.md)
