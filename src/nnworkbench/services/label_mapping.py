"""Map labelsTr values to the Napari annotation layer and back.

Napari and nnInteractive treat ``0`` as "not annotated". For sparse labels (dataset.json has an
``ignore`` label) the label is therefore shifted on load: ``ignore`` becomes ``0`` and every other
class ``n`` becomes ``n + 1``. Saving reverses this. Dense labels (no ``ignore``) are used as they
are, so ``0`` stays background. In both modes an empty layer (all zeros) is the default for a new
label: all ignore for sparse labels, all background for dense labels.
"""

from __future__ import annotations


def label_to_annotation(label, ignore_id: int | None):
    """Return the Napari layer values for a labelsTr array."""
    import numpy as np

    data = np.asarray(label)
    if ignore_id is None:
        return data.astype(np.uint16, copy=True)
    annotation = data.astype(np.uint16) + 1
    annotation[data == ignore_id] = 0
    return annotation


def annotation_to_label(annotation, ignore_id: int | None):
    """Return labelsTr values for a Napari layer, using the smallest fitting integer type."""
    import numpy as np

    data = np.asarray(annotation).astype(np.int64)
    if ignore_id is None:
        label = data
    else:
        label = data - 1
        label[data == 0] = ignore_id
    label = np.clip(label, 0, None)
    return label.astype(np.uint8 if label.max(initial=0) < 256 else np.uint16)
