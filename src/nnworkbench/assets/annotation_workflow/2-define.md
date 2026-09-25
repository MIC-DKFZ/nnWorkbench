# 2. Define

***Define your task precisely and resolve all ambiguities.***

## Dense or Sparse Labeling

Decide first how completely you want to label your images:

- **Dense labeling:** Every pixel or voxel of an image is assigned a class. This gives the model complete information, but it is very time-consuming, especially for large 3D volumes.
- **Sparse labeling:** Only parts of each image are labeled, for example with scribbles or a few objects segmented with nnInteractive. Everything else stays unlabeled (Ignore), and the model learns only from the labeled voxels.

Sparse labeling is the default in this workflow. It needs much less effort per image, so you can spread your annotations over many different images and cover more of the variability in your data. Dense labeling only makes sense if each image contains very few objects, for example a single liver in an abdominal CT, and these can be labeled quickly, e.g. in 2D or with nnInteractive. It is also an option if complete labels already exist.

## Class Definitions

Define all classes required for the segmentation task and assign each class a unique ID. Use continuous numbering starting at 0.

| ID | Class |
| --- | --- |
| 0 | Background |
| 1 | *First class* |
| 2 | *Second class* |
| … | … |
| last | Ignore |

> **Note:** Background is a class and must be explicitly defined and labeled.

> **Note:** Ignore is always required and must have the highest ID. It refers to all voxels that are not labeled and for which we provide no information to the model. With dense labeling, it simply stays unused.

These are the IDs stored in [`dataset.json`](setup.md#create-datasetjson). While annotating, they are shifted by one so that unlabeled voxels (Ignore) can be 0; see [Annotation IDs](3-annotate.md#annotation-ids).

## Class Descriptions

For each class document:

- **Description:** What is the structure?
- **Visual Description:** What do objects in this class look like? Define what should be included or excluded, where the boundaries are, and how uncertain or unusual cases should be handled.
- **Example images:** Add representative examples.

## Example Definition

*Continuing the fictional `Dataset042_CellSegmentation` example from [1. Describe](1-describe.md#example-description), annotated sparsely with scribbles.*

| ID | Class |
| --- | --- |
| 0 | Background |
| 1 | Cell |
| 2 | Nucleus |
| 3 | Ignore |

**0. Background**

- **Description:** Culture medium and the dish surface outside the cells.
- **Visual Description:** Dark. Out-of-focus blur at the top and bottom of a volume is labeled as background only where no cell is visible behind it.

**1. Cell**

- **Description:** The cell body, from the membrane inwards, without the nucleus.
- **Visual Description:** Outlined by the bright membrane signal. The boundary is the outer edge of the membrane signal. Where two cells touch, both are labeled as cell. Cells cut off at the image border are labeled as well.

**2. Nucleus**

- **Description:** The cell nucleus.
- **Visual Description:** Bright, round to oval region inside a cell. In dividing cells, the condensed or split chromatin is labeled as nucleus. Where it is unclear whether a region belongs to the nucleus, leave it unlabeled (Ignore) instead of guessing.

*Example images: add one or two screenshots per class, including a difficult case.*

---

[← 1. Describe](1-describe.md) | [Overview](README.md) | [3. Annotate →](3-annotate.md)
