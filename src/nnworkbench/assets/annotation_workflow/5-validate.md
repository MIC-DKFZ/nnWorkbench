# 5. Validate

***Decide if your model is ready or not.***

After training finishes, nnU-Net predicts every image assigned to the validation fold. Review these predictions carefully to decide whether the model quality is sufficient or whether it needs more or better training data.

```text
nnUNet_results/
└── Dataset042_CellSegmentation/
    └── nnUNetTrainerBetterIgnoreSampling__nnUNetResEncUNetLPlans__3d_fullres/
        ├── fold_0/
        │   └── validation/
        │       ├── cells_003.nii.gz
        │       ├── cells_008.nii.gz
        │       ├── ...
        │       └── summary.json
        ├── fold_1/
        ├── fold_2/
        ├── fold_3/
        └── fold_4/
```

When using cross-validation, repeat this review for all folds. Each image is used for validation in one fold, allowing you to inspect predictions across the complete training dataset. During later prediction, nnU-Net combines the models from all trained folds by default. This often produces more robust and accurate results than a single fold model, so the final quality may be slightly better than the predictions in an individual fold's validation folder.

> **Note:** With fold `all`, every image is used for training, so its `validation` folder contains predictions of the training images themselves. The scores in `summary.json` are therefore not meaningful. With sparse labels, these predictions are still useful for inspection: most of each image was never labeled, so you can check how the model performs in the unlabeled regions (see [Check the Unlabeled Regions](#check-the-unlabeled-regions)).

## Inspect the Predictions

Open each validation prediction together with its image and your labels from `labelsTr`, overlay them, and compare the prediction with your labels.

**In nnWorkbench:** On the **Validate** page, click a finished fold of a model in the **Models** table. The **Cases** table lists its validation cases with their **Dice** and **Weakest class**; click **Dice** to sort the weakest cases first. The magnifier button (**Open in Napari**) next to a case opens the image, your labels and the prediction together.

## What to Look For

- **Look at many images.** Check several images from every fold, especially difficult ones, e.g. noisy or low-contrast images.
- **Inspect each class.** Go through the classes one by one; errors in a rare or small class are easy to overlook.
- **Compare with your labels.** Does the prediction match the labeled structures, and are the boundaries in the right place?

## Validation Metrics

The `summary.json` file contains validation metrics for each case and class, together with average values across the validation set. The most common metric is the Dice score, which measures the overlap between prediction and labels (0 = no overlap, 1 = perfect). These values are useful for tracking progress and comparing training runs.

**In nnWorkbench:** On the **Validate** page, the right panel shows the **Mean Dice** of the selected model over its folds (fold `all` excluded), the **Dice per fold** and the **Dice per class**. Click a case to see its values from `summary.json`.

## Next Step

Once the results are satisfying, continue to [6. Predict](6-predict.md). Otherwise, improve the training data and train again.

**nnU-Net Workflow:** Add more labeled images, especially of the cases where the model fails, and check your labels for errors or inconsistencies.

## Annotation Workflow

With sparse annotations, the validation needs some additional checks, and improving the training data means another iteration.

### Check the Unlabeled Regions

Most of each image was never annotated (Ignore). Check whether objects there are predicted correctly too; this shows how well the model generalizes. Also include the difficult images you identified in [1. Describe](1-describe.md).

### Metrics with Sparse Labels

With sparse annotations, the metrics in `summary.json` are only computed on the labeled voxels. Treat them mainly as an indicator and rely on inspecting the predictions.

### Typical Errors and Next Steps

| Error | Next step |
| --- | --- |
| **Wrong class:** an object is predicted as another class. | Check your annotations of both classes for consistency with the [class definitions](2-define.md#class-descriptions), then add more annotations of both classes, ideally next to each other. |
| **Oversegmentation:** a prediction extends beyond the object. | Add background annotations directly next to the object's boundary. |
| **Undersegmentation:** a prediction covers only part of the object or has holes. | Add more annotations of such objects, up to their boundaries. |
| **Split prediction:** one object is predicted partly as one class and partly as another. | Add more annotations of such objects. |
| **Missed objects:** objects are not predicted at all. | Add more annotations of such objects, e.g. small, faint or rare ones. |
| **Blurry boundaries:** the boundary between two touching classes is imprecise. | Add annotations on both sides of the boundary, close to it. |
| **Errors near the image border:** e.g. at the edges or in out-of-focus slices at the top and bottom. | Add annotations in these regions too. |
| **False detections:** artifacts, debris or noise are predicted as a class. | Add background annotations on these structures. |
| **Errors in specific images:** e.g. only in noisy or blurry images or in one imaging session. | Annotate more in these images to cover their variability. |

Based on these results, either return to [3. Annotate](3-annotate.md) for another iteration or, once the results are satisfying, continue to [6. Predict](6-predict.md).

### Start the Next Iteration

Create a new dataset for each iteration. The previous dataset and its trained model stay unchanged, so you can compare the iterations later.

1. **Create a new dataset** with a new, unused dataset ID, e.g. `Dataset043_CellSegmentation_iter1`, and copy `imagesTr`, `labelsTr` and `dataset.json` from the previous iteration into it.
2. **Improve the training data** based on the validation outcome (see [Typical Errors and Next Steps](#typical-errors-and-next-steps)):
   - Add annotations to existing images where the model makes errors.
   - Add new images to `imagesTr` and annotate them if the errors occur in image types that are underrepresented, e.g. noisy images or one imaging session.
3. **Train again:** update `numTraining` in `dataset.json` if you added images, then preprocess and train the new dataset as described in [4. Train](4-train.md).

**In nnWorkbench:** On the overview, select the dataset and click **＋ New iteration**. The **Source dataset** is the selected dataset and can be changed; nnWorkbench suggests the next free **Dataset ID** and a **Name** ending in `_iter1` (then `_iter2`, …), and copies `imagesTr`, `labelsTr` and `dataset.json` into the new dataset; its `dataset.json` names the source in `inherits_from`. With **Link images** checked, the images stay in the source dataset and the new `dataset.json` lists them (saves space; keep `nnUNet_raw` together, and add new images to that list as well, since nnU-Net then reads only the listed files); the labels are always copied. In the new dataset, use the pencil button (**Annotate**) to extend existing annotations and to annotate newly added images, then preprocess and train.

---

[← 4. Train](4-train.md) | [Overview](README.md) | [6. Predict →](6-predict.md)
