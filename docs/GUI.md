# nnWorkbench documentation

nnWorkbench supports two connected, but independently useful, workflows:

- [nnU-Net workflow](nnunet-workflow.md): manage existing labels and run planning, preprocessing, training, prediction, and validation inspection.
- [Annotation Workflow](annotation-workflow.md): create semantic annotations with nnInteractive in Napari, save them to `annotationsTr`, and convert them into nnU-Net training labels.

## Shared GUI concepts

The header stores the paths to the directories that directly contain `DatasetNNN_Name` folders:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

The roots are independent. nnWorkbench combines datasets found across all selected roots and indicates which locations are available for the selected dataset. **Open folder** opens the selected dataset in the active tab’s root.

Raw, Preprocessed, and Results load only when their tab is opened. Filesystem scans and JSON reads run in the background to keep navigation responsive. The right-hand metadata panel takes `dataset.json` from Raw first, then Preprocessed, then Results; only a Raw manifest can be edited.

The command console appears when nnWorkbench runs an nnU-Net command. It mirrors command output to the launching terminal, allows graceful termination, and prevents multiple nnU-Net commands from running in parallel.
