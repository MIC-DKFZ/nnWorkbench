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
- How can I separate these structures into distinguishable classes?
- How much do individual objects of a class vary in size, shape, appearance, orientation, position, and context?
- How often does each class occur, and in how many images or volumes is it represented? Are some classes much more common or rarer than others?

## Example Description

*A fictional example for the `Dataset042_CellSegmentation` dataset used throughout this guide.*

**Images**

- **Type:** 3D confocal fluorescence microscopy of cultured cells, with one fluorescence channel in which membranes and nuclei appear bright.
- **Number:** 40 volumes from 8 imaging sessions.
- **Variation:** Brightness differs between sessions, and the last two sessions are noticeably noisier. Some volumes have blurry, out-of-focus slices at the top and bottom. Cell density ranges from sparse to fully confluent.
- **Representative image:** A medium-density volume from session 3.
- **Most difficult image:** A confluent culture from session 8 with low contrast and many touching cells.

**Target Structures**

- **Structures of interest:** Cells and their nuclei, to measure cell and nucleus volume.
- **Classes:** *background* (medium outside the cells), *cell* (cell body without nucleus), *nucleus*.
- **Variation:** Cells range from small and round to large and flat, spread-out shapes. Dividing cells look very different: they are rounded, and their nuclei appear condensed or split in two.
- **Frequency:** Background dominates in sparse cultures, cells in confluent ones. Nuclei occur in every volume but cover only a small fraction of the voxels. Dividing cells are rare, only a few per volume.

---

[← Setup](setup.md) | [Overview](README.md) | [2. Define →](2-define.md)
