# 1. Describe

***Understand the full range of your image data and the structures you are interested in.***

Answer and document the following questions:

## Images

- What type of images do I have? e.g. soil-core CT or bone marrow microscopy.
- How many different images are available?
- How much do the images vary in contrast, brightness, noise, resolution, artifacts, background, and acquisition conditions?
- What's a representative image?
- What is the most difficult image I want the model to work on?

## Target Structures

- Which structures am I interested in?
- Why am I interested in them? What do I want to measure, count or compare, and which question should this answer?
  The answer decides which classes are needed (a structure that is never measured may not need a class of its own)
  and how exact the annotations must be (counting needs separated objects, measuring volumes needs precise boundaries).
- How can I separate these structures into distinguishable classes?
- How much do individual objects of a class vary in size, shape, appearance, orientation, position, and context?
- How often does each class occur, and in how many images or volumes is it represented? Are some classes much more common or rarer than others?

**In nnWorkbench:** Write the answers into `dataset-description.md` in the `nnworkbench` folder of the dataset (next
to `imagesTr`; nnU-Net ignores it): a template with these questions (sections 1. Describe Images and 2. Describe Target Structures), the class definitions of
[2. Define](2-define.md) (section 3. Define Classes) and the facts nnWorkbench already knows (number of images,
file ending). **＋ New dataset** creates it (checkbox **Add dataset-description.md**); for an existing
dataset, open **Description** on the **Data** page (next to Files and Statistics) and click **Create description**.
There, **Edit** opens a form with one card per section (click a title to fold it in or out), and **Add image…** for
example images, e.g. a representative and a difficult one. Section 1 has a channel table (`_0000`, `_0001`, … and their
names); like the class table, it is compared with `dataset.json` and can be imported from or pushed to it. **Save** writes the file; experienced users can use **Edit as text**. On the
command line: `nnworkbench-run describe -d 42`. Each new annotation round gets a copy.

## Example Description

*A fictional example for the `Dataset042_CellSegmentation` dataset used throughout this guide.*

**Images**

- **Type:** 3D confocal fluorescence microscopy of cultured cells, with one fluorescence channel in which membranes and nuclei appear bright.
- **Number:** 40 volumes from 8 imaging sessions.
- **Variation:** Brightness differs between sessions, and the last two sessions are noticeably noisier. Some volumes have blurry, out-of-focus slices at the top and bottom. Cell density ranges from sparse to fully confluent.
- **Representative image:** A medium-density volume from session 3.
- **Most difficult image:** A confluent culture from session 8 with low contrast and many touching cells.

**Target Structures**

- **Structures of interest:** Cells and their nuclei.
- **Why these structures:** To measure cell and nucleus volume per cell. Cell and nucleus are therefore separate
  classes, and their boundaries must be annotated precisely.
- **Classes:** *background* (medium outside the cells), *cell* (cell body without nucleus), *nucleus*.
- **Variation:** Cells range from small and round to large and flat, spread-out shapes. Dividing cells look very different: they are rounded, and their nuclei appear condensed or split in two.
- **Frequency:** Background dominates in sparse cultures, cells in confluent ones. Nuclei occur in every volume but cover only a small fraction of the voxels. Dividing cells are rare, only a few per volume.

---

[← Setup](setup.md) | [Overview](README.md) | [2. Define →](2-define.md)
