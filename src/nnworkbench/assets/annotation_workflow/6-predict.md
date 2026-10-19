# 6. Predict

***Release the model and use it in the wild.***

## Run the Prediction

To predict new images, or to pre-label images for the next annotation round, place the images in an input folder using the same file format and naming convention as the training images, including channel postfixes such as `_0000`. Then run:

```bash
nnworkbench-run nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
    -d DATASET_ID -c CONFIGURATION -tr TRAINER -p PLANS
```

Example for a model trained in the nnU-Net Workflow:

```bash
nnworkbench-run nnUNetv2_predict -i imagesTs/ -o predictions/ \
    -d 42 -c 3d_fullres -tr nnUNetTrainer \
    -p nnUNetResEncUNetLPlans
```

Example for a model trained with the Annotation Workflow:

```bash
nnworkbench-run nnUNetv2_predict -i imagesTs/ -o predictions/ \
    -d 42 -c 3d_fullres -tr nnUNetTrainerBetterIgnoreSampling \
    -p nnUNetResEncUNetLPlans
```

`-tr` and `-p` must name the trainer and plans the model was trained with, e.g. `nnUNetTrainerBetterIgnoreSampling` for the Annotation Workflow or `nnUNetTrainer` for the nnU-Net Workflow. Without them, nnU-Net looks for a model trained with its defaults and may not find yours.

By default, nnU-Net combines the five cross-validation folds during prediction, so all five must be trained. To use only some folds, list them with `-f`, e.g. `-f 0 1 2`. For a model trained on all images with fold `all`, add `-f all`:

```bash
nnworkbench-run nnUNetv2_predict -i imagesTs/ -o predictions/ \
    -d 42 -c 3d_fullres -tr nnUNetTrainerBetterIgnoreSampling \
    -p nnUNetResEncUNetLPlans -f all
```

**In nnWorkbench:** On the **Predict** page, select a model in the **Models** table, choose the **Input** and **Output** folders, and set the options in the panel on the right:

- **Folds:** the finished folds to combine (by default the cross-validation folds).
- **Mirroring (TTA):** test-time augmentation; **On · best** is more accurate, **Off · faster** is considerably faster.
- **Continue:** **Skip already predicted cases** predicts only cases without a result in the output folder.
- **Processes:** for preprocessing (`-npp`) and for writing the results (`-nps`); fewer use less RAM.
- **Device:** where the prediction runs (see [Device](4-train.md#device-gpu-or-cpu)).

Then click **▶ Predict**, or **Copy command** to run it on a remote machine. The **Input cases** table shows which cases are already predicted.

The output folder contains one segmentation file for each input case. Review the predictions before using them for further analysis; in nnWorkbench, the magnifier button (**Open in Napari**) next to a case opens the image and its prediction.

## Share the Model

A trained model folder contains much more than prediction needs. To share a model with colleagues, reduce it first:

```bash
nnworkbench-run reduce-checkpoints MODEL_FOLDER --dry-run   # shows what would be changed
nnworkbench-run reduce-checkpoints MODEL_FOLDER --yes       # reduces the model
```

`MODEL_FOLDER` is the trained model's folder in `nnUNet_results`, named after trainer, plans and configuration, e.g. `Dataset042_CellSegmentation/nnUNetTrainerBetterIgnoreSampling__nnUNetResEncUNetLPlans__3d_fullres`.

This removes everything that is not needed for prediction, which makes the model considerably smaller.

Add `--reset-trainer-name` if the recipient does not use nnWorkbench. The model then refers to nnU-Net's default trainer, so the model runs in a standard nnU-Net installation without the bundled trainer classes.

> **Note:** `reduce-checkpoints` changes the model folder in place and cannot be undone. Run it on a copy if you still need the validation results or want to continue training.

Send the reduced model folder, e.g. as a zip file. The recipient can predict with it directly, without setting up any nnU-Net folders:

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

---

[← 5. Validate](5-validate.md) | [Overview](README.md)
