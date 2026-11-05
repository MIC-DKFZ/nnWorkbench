"""Open a raw nnU-Net case and its related files in a dedicated Napari process."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path

from qtpy.QtCore import QObject, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QKeySequence, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .image_files import color_components, has_image_suffix, image_suffix, strip_image_suffix

_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")
_B2ND_SUFFIX = ".b2nd"
_B2ND_SEGMENTATION_SUFFIX = "_seg.b2nd"


@dataclass(frozen=True)
class InspectionFiles:
    images: tuple[Path, ...]
    labels: tuple[Path, ...]


@dataclass(frozen=True)
class _LayerData:
    name: str
    kind: str
    data: object
    affine: object | None = None


@dataclass(frozen=True)
class _CaseData:
    layers: tuple[_LayerData, ...]


class _PrefetchSignals(QObject):
    completed = Signal(str, object)
    failed = Signal(str, str)


class PrefetchStatusBar(QWidget):
    """Dot indicator for the cached state of the navigator's requested prefetches."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._states: list[tuple[str, bool | None]] = []
        self.setFixedHeight(24)

    def set_cases(self, previous: list[tuple[str, bool]], current: str, following: list[tuple[str, bool]]) -> None:
        self._states = [*previous, (current, None), *following]
        neighbours = [ready for _, ready in [*previous, *following]]
        ready = sum(neighbours)
        self.setToolTip(f"Prefetch: {ready}/{len(neighbours)} neighbouring cases ready")
        self.setFixedWidth(max(18, 18 * len(self._states)))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(max(72, 18 * len(self._states)), 24)

    def paintEvent(self, event) -> None:
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        diameter, spacing = 10, 8
        x = 0
        for _identifier, ready in self._states:
            painter.setBrush(QColor("#ffffff" if ready is None else "#5b9137" if ready else "#ed6a05"))
            painter.setPen(QColor("#777777" if ready is None else "#457329" if ready else "#b84c00"))
            painter.drawEllipse(x, (self.height() - diameter) // 2, diameter, diameter)
            x += diameter + spacing
        painter.end()


# Files of datasets whose dataset.json lists its cases ("dataset" key; they may lie outside imagesTr/labelsTr):
# file → (dataset folder, case ID), filled by use_listed_cases when such a dataset is opened.
_LISTED_FILES: dict[Path, tuple[Path, str]] = {}


def use_listed_cases(dataset: Path) -> dict[str, tuple[tuple[Path, ...], Path]] | None:
    """The dataset's "dataset" file list (None without one); its files are remembered for case_id & co."""
    from .dataset_json import listed_cases

    listed = listed_cases(dataset)
    for case, (images, label) in (listed or {}).items():
        for path in (*images, label):
            _LISTED_FILES[path] = (dataset, case)
    return listed


def _dataset_of(path: Path) -> Path:
    """The raw dataset folder of an image or label file."""
    return _LISTED_FILES.get(path, (path.parent.parent, ""))[0]


def case_id(path: Path) -> str:
    """Derive the nnU-Net case ID, stripping an image-channel suffix when present."""
    if path in _LISTED_FILES:
        return _LISTED_FILES[path][1]
    return _CHANNEL_SUFFIX.sub("", strip_image_suffix(path))


def related_files(selected: Path) -> InspectionFiles:
    """Find the raw image and label files belonging to ``selected``'s case."""
    return related_files_for_case(_dataset_of(selected), case_id(selected))


def related_files_for_case(dataset: Path, identifier: str) -> InspectionFiles:
    """Find all raw files for a known nnU-Net case ID."""
    listed = use_listed_cases(dataset)
    if listed is not None:
        images, label = listed.get(identifier, ((), None))
        return InspectionFiles(tuple(path for path in images if path.is_file()),
                               (label,) if label is not None and label.is_file() else ())

    def matching(folder: str, images_only: bool = False) -> tuple[Path, ...]:
        directory = dataset / folder
        if not directory.is_dir():
            return ()
        paths = (path for path in directory.iterdir() if path.is_file() and case_id(path) == identifier
                 and (not images_only or has_image_suffix(path)))
        return tuple(sorted(paths, key=lambda path: path.name.casefold()))

    return InspectionFiles(matching("imagesTr", True), matching("labelsTr", True))


def image_case_ids(dataset: Path) -> tuple[str, ...]:
    """Return the ordered case IDs represented by training image files."""
    listed = use_listed_cases(dataset)
    if listed is not None:
        return tuple(sorted((case for case, (paths, _label) in listed.items() if any(path.is_file() for path in paths)),
                            key=str.casefold))
    images = dataset / "imagesTr"
    if not images.is_dir():
        return ()
    return tuple(sorted({case_id(path) for path in images.iterdir()
                         if path.is_file() and has_image_suffix(path)}, key=str.casefold))


def prediction_case_ids(prediction_folder: Path) -> tuple[str, ...]:
    """Return the cases available for inspection, using predictions as the source of truth."""
    if not prediction_folder.is_dir():
        return ()
    return tuple(sorted({case_id(path) for path in prediction_folder.iterdir()
                         if path.is_file() and has_image_suffix(path)}, key=str.casefold))


def preprocessed_case_id(path: Path) -> str:
    """Return an nnU-Net preprocessed case ID from a data or segmentation B2ND file."""
    if path.name.endswith(_B2ND_SEGMENTATION_SUFFIX):
        return path.name[:-len(_B2ND_SEGMENTATION_SUFFIX)]
    return path.name[:-len(_B2ND_SUFFIX)] if path.name.endswith(_B2ND_SUFFIX) else path.stem


def preprocessed_case_ids(session_folder: Path) -> tuple[str, ...]:
    """Return cases represented by data B2ND files, excluding their ``_seg`` companions."""
    if not session_folder.is_dir():
        return ()
    return tuple(sorted({preprocessed_case_id(path) for path in session_folder.iterdir()
                         if path.is_file() and path.name.endswith(_B2ND_SUFFIX)
                         and not path.name.endswith(_B2ND_SEGMENTATION_SUFFIX)}, key=str.casefold))


def _matching_case_files(directory: Path | None, identifier: str) -> tuple[Path, ...]:
    if directory is None or not directory.is_dir():
        return ()
    return tuple(sorted((path for path in directory.iterdir()
                         if path.is_file() and has_image_suffix(path) and case_id(path) == identifier),
                        key=lambda path: path.name.casefold()))


def prefetch_indices(index: int, count: int, previous: bool, following: bool, radius: int) -> tuple[int, ...]:
    """Return valid prefetch positions in nearest-first alternating direction order."""
    result: list[int] = []
    for offset in range(1, radius + 1):
        if following and index + offset < count:
            result.append(index + offset)
        if previous and index - offset >= 0:
            result.append(index - offset)
    return tuple(result)


def image_layer_name(path: Path) -> str:
    """Return the image-layer name expected by napari-nninteractive."""
    return strip_image_suffix(path)


def file_ending(path: Path) -> str:
    """Keep a compound suffix such as ``.nii.gz`` intact, or use the image's normal suffix."""
    return image_suffix(path) or path.suffix


def annotation_target_path(image_path: Path) -> Path:
    """Return the labelsTr file that annotations of an image channel are saved to (the listed label file for a
    "dataset" file list in dataset.json)."""
    if image_path in _LISTED_FILES:
        dataset, case = _LISTED_FILES[image_path]
        listed = use_listed_cases(dataset) or {}
        if case in listed:
            return listed[case][1]
    return image_path.parent.parent / "labelsTr" / f"{case_id(image_path)}{file_ending(image_path)}"


def annotation_source_image(path: Path) -> Path:
    """Return an imagesTr source for a label, preferring channel 0000."""
    listed = use_listed_cases(_LISTED_FILES[path][0]) if path in _LISTED_FILES else None
    listed_label = listed is not None and listed.get(_LISTED_FILES[path][1], ((), None))[1] == path
    if path.parent.name != "labelsTr" and not listed_label:
        return path
    images = related_files(path).images
    if not images:
        raise ValueError(f"No imagesTr file exists for label case '{case_id(path)}'.")
    return next((image for image in images if image_layer_name(image).endswith("_0000")), images[0])


def semantic_map_name(image_path: Path) -> str:
    return f"semantic map - {image_layer_name(image_path)}"


def _read_image(path: Path) -> tuple[object, object]:
    """Read image data and a Napari affine transform from MITK-Python."""
    import mitk
    import numpy as np

    image = mitk.Image(path)
    data = np.asarray(image)
    # A colour 2D image (RGB/RGBA PNG, BMP, TIFF) comes as (colour, Y, X): colour last, so Napari shows one RGB
    # image (not three slices) and the annotation is 2D, as nnU-Net reads such images.
    components = color_components(path) or 1
    rgb = components in (3, 4) and data.ndim == 3 and data.shape[0] == components
    if rgb:
        data = np.moveaxis(data, 0, -1)
    # MITK geometry is expressed in x, y, z while its NumPy view is z, y, x.
    # Reverse both coordinate axes to produce Napari's data-coordinate affine.
    reverse = np.flipud(np.eye(3))
    linear = reverse @ np.asarray(image.direction) @ reverse @ np.diag(np.asarray(image.spacing)[::-1])
    # MITK always reports 3D geometry; a 2D image uses only its last two (Y, X) axes.
    spatial = 2 if rgb else min(data.ndim, 3)
    affine = np.eye((2 if rgb else data.ndim) + 1)
    affine[-spatial - 1:-1, -spatial - 1:-1] = linear[-spatial:, -spatial:]
    affine[-spatial - 1:-1, -1] = np.asarray(image.origin)[::-1][-spatial:]
    return data, affine


def _layer_name(role: str, path: Path) -> str:
    """Name a layer by what it shows, e.g. "Label — case_001.nii.gz", so layers are easy to tell apart.

    Annotation sessions keep plain names: napari-nninteractive requires its image layer and the
    matching "semantic map - <image>" layer to be named exactly like that (see open_annotation).
    """
    return f"{role} — {path.name}"


def _read_case(dataset: Path, identifier: str) -> _CaseData:
    """Load a complete case off the Napari GUI thread for display or prefetching."""
    files = related_files_for_case(dataset, identifier)
    layers: list[_LayerData] = []
    for path in files.images:
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Image", path), "image", data, affine))
    for path in files.labels:
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Label", path), "labels", data, affine))
    return _CaseData(tuple(layers))


def _read_prediction_case(prediction_folder: Path, image_folder: Path | None, reference_folder: Path | None,
                           identifier: str) -> _CaseData:
    """Load one prediction-led case and any matching optional comparison files."""
    layers: list[_LayerData] = []
    for path in _matching_case_files(image_folder, identifier):
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Image", path), "image", data, affine))
    for path in _matching_case_files(prediction_folder, identifier):
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Prediction", path), "prediction", data, affine))
    for path in _matching_case_files(reference_folder, identifier):
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Label", path), "reference", data, affine))
    return _CaseData(tuple(layers))


def _read_validation_case(validation_folder: Path, raw_dataset: Path | None, identifier: str) -> _CaseData:
    """Load one validation prediction with its matching raw image and reference label."""
    layers: list[_LayerData] = []
    if raw_dataset is not None and raw_dataset.is_dir():
        files = related_files_for_case(raw_dataset, identifier)
        for path in files.images:
            data, affine = _read_image(path)
            layers.append(_LayerData(_layer_name("Image", path), "image", data, affine))
        for path in files.labels:
            data, affine = _read_image(path)
            layers.append(_LayerData(_layer_name("Label", path), "labels", data, affine))
    for path in _matching_case_files(validation_folder, identifier):
        data, affine = _read_image(path)
        layers.append(_LayerData(_layer_name("Prediction", path), "prediction", data, affine))
    return _CaseData(tuple(layers))


def _read_preprocessed_case(session_folder: Path, identifier: str) -> _CaseData:
    """Load nnU-Net's preprocessed B2ND data through nnU-Net's Blosc2 dataset reader."""
    import numpy as np
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    data_path = session_folder / f"{identifier}{_B2ND_SUFFIX}"
    if not data_path.is_file():
        raise ValueError(f"Preprocessed data file not found: {data_path}")
    segmentation_path = session_folder / f"{identifier}{_B2ND_SEGMENTATION_SUFFIX}"
    if not segmentation_path.is_file():
        raise ValueError(f"Preprocessed segmentation file not found: {segmentation_path}")
    dataset = nnUNetDatasetBlosc2(str(session_folder), identifiers=[identifier])
    data, segmentation, _previous_stage_segmentation, _properties = dataset[identifier]
    data_array = np.asarray(data)
    if data_array.ndim < 3:
        raise ValueError(f"Unexpected preprocessed data shape for {data_path.name}: {data_array.shape}")
    channels = data_array if data_array.ndim > 3 else (data_array,)
    # Name the channels only if there are several, so their layers stay distinguishable.
    layers = [
        _LayerData(_layer_name("Image", data_path) + (f" (channel {index})" if len(channels) > 1 else ""),
                   "image", channel)
        for index, channel in enumerate(channels)
    ]
    segmentation_array = np.asarray(segmentation)
    if segmentation_array.ndim > 3 and segmentation_array.shape[0] == 1:
        segmentation_array = segmentation_array[0]
    layers.append(_LayerData(_layer_name("Label", segmentation_path), "labels", segmentation_array))
    return _CaseData(tuple(layers))


def _display_case(viewer, case_data: _CaseData) -> None:
    """Update reusable Napari layers with already loaded case data and geometry."""
    reusable = getattr(viewer, "_nnworkbench_case_layers", {})
    used: set[tuple[str, int]] = set()
    positions: dict[str, int] = {}
    for layer_data in case_data.layers:
        position = positions.get(layer_data.kind, 0)
        positions[layer_data.kind] = position + 1
        slot = (layer_data.kind, position); used.add(slot)
        layer = reusable.get(slot)
        if layer is None:
            if layer_data.kind in {"labels", "prediction", "reference"}:
                layer = viewer.add_labels(layer_data.data, name=layer_data.name, affine=layer_data.affine)
            else:
                layer = viewer.add_image(layer_data.data, name=layer_data.name, affine=layer_data.affine)
            reusable[slot] = layer
        else:
            layer.data = layer_data.data
            layer.name = layer_data.name
            if layer_data.affine is not None:
                layer.affine = layer_data.affine
            layer.visible = True
        layer.metadata["nnworkbench_source"] = layer_data.name
    for slot, layer in list(reusable.items()):
        if slot not in used:
            layer.visible = False
    viewer._nnworkbench_case_layers = reusable
    _reset_middle_slice(viewer, case_data)
    if hasattr(viewer, "reset_view"):
        viewer.reset_view()


def _reset_middle_slice(viewer, case_data: _CaseData) -> None:
    """Place Napari's z slider at the middle slice of the newly displayed volume."""
    if not hasattr(viewer, "dims"):
        return
    volume = next((layer.data for layer in case_data.layers if layer.kind == "image"), None)
    if volume is None:
        volume = next((layer.data for layer in case_data.layers), None)
    shape = getattr(volume, "shape", ())
    if len(shape) < 3:
        return
    axis = viewer.dims.ndim - 3
    if axis >= 0:
        viewer.dims.set_current_step(axis, shape[-3] // 2)


def _dataset_ignore_id(image_path: Path) -> int | None:
    """Return the ignore label of the dataset an image belongs to (``None`` for dense labels)."""
    from .workflow_status import ignore_label_id

    try:
        with (_dataset_of(image_path) / "dataset.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return ignore_label_id(manifest if isinstance(manifest, dict) else None)


def save_semantic_annotation(viewer, image_path: Path, target: Path) -> None:
    """Save nnInteractive's semantic-map layer to labelsTr with the source image geometry.

    The layer values are mapped back to nnU-Net label IDs (see ``label_mapping``).
    """
    import mitk
    import numpy as np

    from .label_mapping import annotation_to_label

    layer_name = semantic_map_name(image_path)
    try:
        layer = viewer.layers[layer_name]
    except (KeyError, IndexError) as error:
        raise ValueError(f"No nnInteractive semantic-map layer named '{layer_name}' is available.") from error
    source = mitk.Image(image_path)
    values = annotation_to_label(np.asarray(layer.data), _dataset_ignore_id(image_path))
    annotation = mitk.Image(values, spacing=source.spacing, origin=source.origin, direction=source.direction)
    target.parent.mkdir(parents=True, exist_ok=True)
    annotation.save(target)


def annotation_label_summary(image_path: Path) -> tuple[tuple[int, str], ...]:
    """Return dataset labels translated to the IDs used while annotating.

    Sparse labels (with an ``ignore`` label): ``ignore`` is shown as ``0`` and all other label IDs are
    shifted by one. Dense labels (no ``ignore``): the IDs are unchanged and ``0`` is background.
    """
    from .workflow_status import ignore_label_id

    try:
        with (_dataset_of(image_path) / "dataset.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return ()
    labels = manifest.get("labels", {}) if isinstance(manifest, dict) else {}
    if not isinstance(labels, dict):
        return ()
    sparse = ignore_label_id(manifest) is not None  # the same rule as loading and saving
    translated: list[tuple[int, str]] = []
    for name, value in labels.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int):
            continue
        if not sparse:
            translated.append((value, name))
        else:
            translated.append((0 if name == "ignore" else value + 1, name))
    return tuple(sorted(translated, key=lambda entry: (entry[0], entry[1].casefold())))


LEGEND_ROWS = 10  # rows of the annotation legend shown at once; more classes scroll


def annotation_legend(labels: tuple[tuple[int, str], ...], parent=None) -> QTableWidget:
    """The annotation IDs as a two-column table that scrolls beyond ``LEGEND_ROWS`` rows (a dataset can have
    hundreds of classes, and the dock must keep the save buttons visible)."""
    table = QTableWidget(len(labels), 2, parent)
    table.setHorizontalHeaderLabels(("ID", "Label"))
    table.verticalHeader().hide()
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setWordWrap(False)
    table.setShowGrid(False)
    for row, (label_id, name) in enumerate(labels):
        identifier = QTableWidgetItem(str(label_id))
        identifier.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title = QTableWidgetItem(name)
        title.setToolTip(f"{label_id}: {name}")
        table.setItem(row, 0, identifier)
        table.setItem(row, 1, title)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    row_height = table.verticalHeader().defaultSectionSize()
    shown = max(1, min(len(labels), LEGEND_ROWS))
    table.setFixedHeight(header.sizeHint().height() + shown * row_height + 2 * table.frameWidth())
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return table


class AnnotationSaveWidget(QWidget):
    """Napari dock widget that saves the current nnInteractive semantic map."""

    def __init__(self, viewer, image_path: Path, target: Path) -> None:
        super().__init__()
        self._viewer, self._image_path, self._target = viewer, image_path, target
        self._semantic_layer_name = semantic_map_name(image_path)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(3)
        self._layer_info = QLabel()
        layout.addWidget(self._layer_info)
        labels = annotation_label_summary(image_path)
        self._label_summary = QLabel("<b>Annotation IDs</b>" if labels else "<b>Annotation IDs: </b> unavailable")
        layout.addWidget(self._label_summary)
        self._label_list = annotation_legend(labels)
        self._label_list.setVisible(bool(labels))
        layout.addWidget(self._label_list)
        note = ("<i>Sparse labels: 0 is ignore (not annotated); all other label IDs are shifted by +1. "
                "Saving writes labelsTr with the original IDs.</i>" if _dataset_ignore_id(image_path) is not None
                else "<i>Dense labels: 0 is background. Saving writes labelsTr.</i>")
        self._label_note = QLabel(note)
        self._label_note.setWordWrap(True)
        layout.addWidget(self._label_note)
        self._status = QLabel(); self._status.hide(); layout.addWidget(self._status)
        save = QPushButton("Save annotation")
        save.clicked.connect(self._save); layout.addWidget(save)
        # Napari's own Close Window shortcut (Ctrl+W, Cmd+W); Ctrl+C closed it and lost unsaved annotation.
        close = QPushButton(f"Close Napari ({_close_keys()})")
        close.clicked.connect(self._viewer.close); layout.addWidget(close)
        layout.addStretch()  # a tall dock keeps the rows together at the top
        self._viewer.layers.events.inserted.connect(self._update_layer_info)
        self._viewer.layers.events.removed.connect(self._update_layer_info)
        self._update_layer_info()

    def _update_layer_info(self, event=None) -> None:
        try:
            layer = self._viewer.layers[self._semantic_layer_name]
        except (KeyError, IndexError):
            self._layer_info.setText("<b>Annotation layer: </b>")
        else:
            self._layer_info.setText(f"<b>Annotation layer: </b>&nbsp;{escape(layer.name)}")

    def _save(self) -> None:
        try:
            save_semantic_annotation(self._viewer, self._image_path, self._target)
        except Exception as error:  # noqa: BLE001  (e.g. mitk.Exception for an RGB .png: say so, the window stays)
            self._status.setText(f"Could not save: {error}"); self._status.show()
            QMessageBox.warning(self, "Annotation not saved", str(error))
            return
        self._status.setText(f"Saved to {self._target.name}"); self._status.show()


def _activate_nninteractive(viewer):
    """Show napari-nninteractive's dock widget when the plugin is installed."""
    try:
        dock_widget, widget = viewer.window.add_plugin_dock_widget("napari-nninteractive")
        dock_widget.setWindowTitle("nnInteractive controls")
        # nnWorkbench saves the shared semantic map, rather than individual object layers.
        output = getattr(widget, "aggregation_output_combo", None)
        semantic_output = getattr(widget, "OUT_SEMANTIC", None)
        if output is not None and semantic_output is not None:
            output.setCurrentText(semantic_output)
        return dock_widget
    except (KeyError, ModuleNotFoundError, ValueError) as error:
        print(f"Could not activate napari-nninteractive: {error}")
        return None


def _activate_layer_statistics(viewer):
    """Show napari-layer-statistics when it is installed."""
    dock_widget = None
    try:
        dock_widget, _widget = viewer.window.add_plugin_dock_widget("napari-layer-statistics")
        dock_widget.setWindowTitle("Layer statistics")
        return dock_widget
    except Exception as error:  # noqa: BLE001  (optional Napari plugin; a failure only disables it)
        if dock_widget is not None:
            try:
                dock_widget.close()
            except Exception:  # noqa: BLE001, S110
                pass
        print(f"Could not activate napari-layer-statistics: {error}")
        return None


def _expand_nninteractive_dock(viewer, save_dock, nninteractive_dock) -> None:
    """Let nnInteractive fill the right dock area after Napari has laid it out."""
    try:
        widget = nninteractive_dock.widget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setMinimumSize(0, 0)
        for scroll_area in widget.findChildren(QScrollArea):
            scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            scroll_area.setWidgetResizable(True)
        # The save panel remains compact; nnInteractive receives the rest of the dock stack.
        viewer.window._qt_window.resizeDocks(
            [save_dock, nninteractive_dock],
            [save_dock.height(), max(1, viewer.window._qt_window.height())],
            Qt.Vertical,
        )
    except Exception as error:  # noqa: BLE001  (optional nnInteractive plugin)
        print(f"Could not expand nnInteractive controls: {error}")


def _close_keys() -> str:
    """Napari's Close Window shortcut (Ctrl+W, Cmd+W on macOS) as shown in menus."""
    return QKeySequence("Ctrl+W").toString(QKeySequence.SequenceFormat.NativeText)


def _show_message(viewer, text: str) -> None:
    """Show a short message in the middle of the canvas, e.g. while data is loading."""
    overlay = viewer.text_overlay
    overlay.text, overlay.position, overlay.font_size, overlay.visible = text, "middle_center", 14, True
    viewer.status = text
    # An empty viewer shows napari's welcome screen on top of the canvas; hide it so the message is visible.
    hide_welcome = getattr(getattr(viewer.window, "_qt_viewer", None), "_set_welcome_visible", None)
    if callable(hide_welcome):
        hide_welcome(False)


def _hide_message(viewer) -> None:
    viewer.text_overlay.visible = False
    viewer.status = "Ready"


def _load_in_background(viewer, description: str, load: Callable[[], object],
                        on_loaded: Callable[[object], None]) -> None:
    """Load data in a worker thread so the viewer appears immediately and shows the progress."""
    from napari.qt.threading import create_worker

    _show_message(viewer, f"Loading {description} …")
    worker = create_worker(load)

    def loaded(data: object) -> None:
        _hide_message(viewer)
        on_loaded(data)

    def failed(error: Exception) -> None:
        _show_message(viewer, f"Could not load {description}: {error}")
        print(f"Could not load {description}: {error}")

    worker.returned.connect(loaded)
    worker.errored.connect(failed)
    viewer._nnworkbench_loader = worker  # Keep the worker alive until it finishes.
    worker.start()


def open_annotation(selected_path: Path) -> None:
    """Open an image in Napari for nnInteractive annotation and optional semantic-map reuse."""
    import napari

    image_path = annotation_source_image(selected_path)
    # a selected label file is saved back to itself; an image to its case's label file
    target = annotation_target_path(image_path)
    if selected_path.parent.name == "labelsTr" and has_image_suffix(selected_path):
        target = selected_path
    viewer = napari.Viewer(title=f"nnWorkbench annotation — {case_id(image_path)}")

    _load_in_background(viewer, case_id(image_path), lambda: load_annotation_case(image_path, target),
                        lambda loaded: _set_up_annotation(viewer, image_path, target, *loaded))
    napari.run()


def load_annotation_case(image_path: Path, target: Path) -> tuple[tuple, tuple | None]:
    """Load an image and its saved label for annotating (the pencil), in annotation IDs.

    Sparse labels (dataset.json has an ignore label) are shifted: ignore becomes 0 and every class n
    becomes n + 1. Dense labels are used as they are, so 0 stays background. Inspection never calls this
    and always shows the original label IDs.
    """
    from .label_mapping import label_to_annotation

    image = _read_image(image_path)
    if not target.is_file():
        return image, None
    label, affine = _read_image(target)
    return image, (label_to_annotation(label, _dataset_ignore_id(image_path)), affine)


def _set_up_annotation(viewer, image_path: Path, target: Path, image: tuple, annotation: tuple | None) -> None:
    """Add the image, its semantic map and the annotation panels once the data is loaded."""
    import numpy as np

    data, affine = image
    image_layer = viewer.add_image(data, name=image_layer_name(image_path), affine=affine)
    shape = data.shape[:-1] if getattr(image_layer, "rgb", False) else data.shape  # RGB: a 2D annotation
    if annotation is not None:
        annotation_data, annotation_affine = annotation
        # MITK exposes an efficient, read-only NumPy view. nnInteractive updates the
        # semantic map in place, so an existing annotation must be copied first.
        viewer.add_labels(annotation_data.copy(), name=semantic_map_name(image_path), affine=annotation_affine)
    else:
        # Always provide nnInteractive with a writable semantic map matching the
        # source image's dimensions and geometry, even before its first save.
        viewer.add_labels(np.zeros(shape, dtype=np.uint16), name=semantic_map_name(image_path), affine=affine)
    _reset_middle_slice(viewer, _CaseData((_LayerData(image_layer_name(image_path), "image", data, affine),)))
    viewer.reset_view()
    save_widget = AnnotationSaveWidget(viewer, image_path, target)
    save_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    save_dock = viewer.window.add_dock_widget(
        save_widget,
        name="nnWorkbench — Annotation saving",
        area="right",
        add_vertical_stretch=False,
    )
    nninteractive_dock = _activate_nninteractive(viewer)
    statistics_dock = _activate_layer_statistics(viewer)
    if nninteractive_dock is not None:
        nninteractive_widget = nninteractive_dock.widget()
        save_widget.setMinimumWidth(nninteractive_widget.minimumSizeHint().width())
        # Keep nnWorkbench's save controls above the nnInteractive controls.
        viewer.window._qt_window.splitDockWidget(save_dock, nninteractive_dock, Qt.Vertical)
        if statistics_dock is not None:
            try:
                # Layer statistics shares nnInteractive's dock area as a selectable tab.
                viewer.window._qt_window.tabifyDockWidget(nninteractive_dock, statistics_dock)
                nninteractive_dock.raise_()
            except Exception as error:  # noqa: BLE001  (optional Napari plugin; a failure only disables it)
                try:
                    statistics_dock.close()
                except Exception:  # noqa: BLE001, S110
                    pass
                print(f"Could not dock napari-layer-statistics: {error}")
    # Keep the compact panel large enough for its two label lines. Without a
    # fixed height the vertical dock splitter can collapse them when Napari is
    # resized.
    save_widget.adjustSize()
    save_dock.setFixedHeight(save_widget.sizeHint().height() + 36)
    if nninteractive_dock is not None:
        QTimer.singleShot(100, lambda: _expand_nninteractive_dock(viewer, save_dock, nninteractive_dock))


class CaseNavigator(QWidget):
    """A compact right-side Napari control for switching between image cases."""

    def __init__(self, viewer, case_loader: Callable[[str], _CaseData], cases: tuple[str, ...], current_case: str,
                 current_data: _CaseData,
                 prefetch_previous: bool = True, prefetch_next: bool = True, prefetch_radius: int = 1) -> None:
        super().__init__()
        self._viewer, self._case_loader, self._cases = viewer, case_loader, cases
        self._index = cases.index(current_case) if current_case in cases else 0
        self._prefetch_previous = prefetch_previous
        self._prefetch_next = prefetch_next
        self._prefetch_radius = prefetch_radius
        self._cache: dict[str, _CaseData] = {current_case: current_data}
        self._prefetching: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="napari-prefetch")
        self._signals = _PrefetchSignals(self)
        self._signals.completed.connect(self._prefetch_completed)
        self._signals.failed.connect(self._prefetch_failed)
        self.destroyed.connect(self._shutdown_prefetching)
        layout = QVBoxLayout(self)
        self._current = QLabel(); layout.addWidget(self._current)
        buttons = QHBoxLayout()
        self._previous = QPushButton("Previous (A)"); self._previous.clicked.connect(self.previous)
        self._next = QPushButton("Next (D)"); self._next.clicked.connect(self.next)
        buttons.addWidget(self._previous); buttons.addWidget(self._next); layout.addLayout(buttons)
        self._prefetch_status = PrefetchStatusBar(self)
        prefetch_line = QHBoxLayout(); prefetch_line.setContentsMargins(0, 0, 0, 0)
        prefetch_line.addWidget(QLabel("Prefetch")); prefetch_line.addWidget(self._prefetch_status)
        prefetch_line.addStretch(); layout.addLayout(prefetch_line)
        self._close = QPushButton(f"Close Napari ({_close_keys()})"); self._close.clicked.connect(self.close_viewer)
        layout.addWidget(self._close)
        self._update_controls()
        self._prefetch_neighbours()

    def previous(self) -> None:
        if self._index > 0:
            self._navigate(-1)

    def next(self) -> None:
        if self._index < len(self._cases) - 1:
            self._navigate(1)

    def close_viewer(self) -> None:
        self._viewer.close()

    def _navigate(self, step: int) -> None:
        self._index += step
        identifier = self._cases[self._index]
        case_data = self._cache.get(identifier)
        if case_data is None:
            # Not prefetched yet: show that something is happening while the case loads.
            _show_message(self._viewer, f"Loading {identifier} …")
            QApplication.processEvents()
            try:
                case_data = self._case_loader(identifier)
            finally:
                _hide_message(self._viewer)
            self._cache[identifier] = case_data
        _display_case(self._viewer, case_data)
        self._update_controls()
        self._prefetch_neighbours()

    def _update_controls(self) -> None:
        self._current.setText(f"Image case {self._index + 1}/{len(self._cases)}")
        self._previous.setEnabled(self._index > 0)
        self._next.setEnabled(self._index < len(self._cases) - 1)

    def _prefetch_neighbours(self) -> None:
        indices = prefetch_indices(self._index, len(self._cases), self._prefetch_previous,
                                   self._prefetch_next, self._prefetch_radius)
        for index in indices:
            identifier = self._cases[index]
            if identifier not in self._cache and identifier not in self._prefetching:
                self._prefetching.add(identifier)
                future = self._executor.submit(self._case_loader, identifier)
                future.add_done_callback(lambda completed, case=identifier: self._finish_prefetch(case, completed))
        keep = {self._cases[self._index]}
        keep.update(self._cases[index] for index in indices)
        self._cache = {identifier: data for identifier, data in self._cache.items() if identifier in keep}
        self._refresh_prefetch_status(indices)

    def _finish_prefetch(self, identifier: str, future: Future) -> None:
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001  (reported to the viewer instead of crashing the loader thread)
            signal, value = "failed", str(error)
        else:
            signal, value = "completed", result
        try:
            getattr(self._signals, signal).emit(identifier, value)
        except RuntimeError:
            pass  # The viewer was closed while the case was still being prefetched.

    def _prefetch_completed(self, identifier: str, data: object) -> None:
        self._prefetching.discard(identifier)
        if isinstance(data, _CaseData):
            self._cache[identifier] = data
        self._refresh_prefetch_status()

    def _prefetch_failed(self, identifier: str, error: str) -> None:
        self._prefetching.discard(identifier)
        print(f"Could not prefetch {identifier}: {error}")
        self._refresh_prefetch_status()

    def _refresh_prefetch_status(self, indices: tuple[int, ...] | None = None) -> None:
        previous = [(self._cases[index], self._cases[index] in self._cache)
                    for index in range(max(0, self._index - self._prefetch_radius), self._index)] if self._prefetch_previous else []
        following = [(self._cases[index], self._cases[index] in self._cache)
                     for index in range(self._index + 1,
                                        min(len(self._cases), self._index + self._prefetch_radius + 1))] if self._prefetch_next else []
        self._prefetch_status.set_cases(previous, self._cases[self._index], following)

    def _shutdown_prefetching(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _open_navigated_viewer(title: str, identifier: str, cases: tuple[str, ...],
                           loader: Callable[[str], _CaseData], prefetch_previous: bool, prefetch_next: bool,
                           prefetch_radius: int) -> None:
    """Open Napari immediately, load the selected case in the background, then add case navigation."""
    import napari

    viewer = napari.Viewer(title=title)

    def show(current_data: _CaseData) -> None:
        _display_case(viewer, current_data)
        if not cases:
            return
        navigator = CaseNavigator(viewer, loader, cases, identifier, current_data,
                                  prefetch_previous, prefetch_next, prefetch_radius)
        viewer.window.add_dock_widget(navigator, name="Case navigation", area="right")
        viewer.bind_key("a", lambda *_: navigator.previous(), overwrite=True)
        viewer.bind_key("d", lambda *_: navigator.next(), overwrite=True)

    _load_in_background(viewer, identifier, lambda: loader(identifier), show)
    napari.run()


def open_case(selected: Path, prefetch_previous: bool = True, prefetch_next: bool = True,
              prefetch_radius: int = 1) -> None:
    """Start Napari and add all supported raw files related to the selected case."""
    dataset, identifier = _dataset_of(selected), case_id(selected)
    _open_navigated_viewer(f"nnWorkbench — {identifier}", identifier, image_case_ids(dataset),
                           lambda case: _read_case(dataset, case), prefetch_previous, prefetch_next, prefetch_radius)


def open_prediction_folder(prediction_folder: Path, image_folder: Path | None = None,
                           reference_folder: Path | None = None, prefetch_previous: bool = True,
                           prefetch_next: bool = True, prefetch_radius: int = 1) -> None:
    """Inspect prediction cases, with optional matching images and reference labels."""
    cases = prediction_case_ids(prediction_folder)
    if not cases:
        raise ValueError(f"No prediction images found in {prediction_folder}.")
    identifier = cases[0]
    loader = lambda case: _read_prediction_case(prediction_folder, image_folder, reference_folder, case)
    _open_navigated_viewer(f"nnWorkbench predictions — {identifier}", identifier, cases, loader,
                           prefetch_previous, prefetch_next, prefetch_radius)


def open_preprocessed_case(selected: Path, prefetch_previous: bool = True, prefetch_next: bool = True,
                           prefetch_radius: int = 1) -> None:
    """Inspect a preprocessed nnU-Net B2ND session with the standard case navigator."""
    session_folder, identifier = selected.parent, preprocessed_case_id(selected)
    cases = preprocessed_case_ids(session_folder)
    if identifier not in cases:
        raise ValueError(f"No preprocessed data case found for {selected.name}.")
    loader = lambda case: _read_preprocessed_case(session_folder, case)
    _open_navigated_viewer(f"nnWorkbench preprocessed — {identifier}", identifier, cases, loader,
                           prefetch_previous, prefetch_next, prefetch_radius)


def open_validation_prediction(prediction_path: Path, raw_dataset: Path | None = None,
                               prefetch_previous: bool = True, prefetch_next: bool = True,
                               prefetch_radius: int = 1) -> None:
    """Inspect and navigate validation predictions with matching raw images and labels."""
    validation_folder, identifier = prediction_path.parent, case_id(prediction_path)
    cases = prediction_case_ids(validation_folder)
    if identifier not in cases:
        raise ValueError(f"No validation prediction found for {prediction_path.name}.")
    loader = lambda case: _read_validation_case(validation_folder, raw_dataset, case)
    _open_navigated_viewer(f"nnWorkbench validation — {identifier}", identifier, cases, loader,
                           prefetch_previous, prefetch_next, prefetch_radius)


def main() -> None:
    if os.name == "nt":
        # started by the GUI, this window shares the GUI's console: Ctrl+C there must not interrupt Napari (and a
        # half-saved annotation); the window closes with its close button or with the GUI
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    parser = argparse.ArgumentParser(description="Open an nnU-Net case in Napari.")
    parser.add_argument("file", type=Path, nargs="?")
    parser.add_argument("--prefetch-radius", type=int, default=1, choices=range(1, 21))
    parser.add_argument("--no-prefetch-previous", action="store_true")
    parser.add_argument("--no-prefetch-next", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--preprocessed-case", action="store_true")
    parser.add_argument("--validation-prediction", action="store_true")
    parser.add_argument("--raw-dataset", type=Path)
    parser.add_argument("--prediction-folder", type=Path)
    parser.add_argument("--image-folder", type=Path)
    parser.add_argument("--reference-folder", type=Path)
    arguments = parser.parse_args()
    selected = arguments.file
    if arguments.raw_dataset is not None and selected is not None:  # its dataset.json may list files elsewhere
        use_listed_cases(arguments.raw_dataset)
        selected = selected.absolute()
    if arguments.prediction_folder is None and (selected is None or not selected.is_file()):
        raise SystemExit(f"File not found: {selected}")
    if arguments.prediction_folder is not None and not arguments.prediction_folder.is_dir():
        raise SystemExit(f"Prediction folder not found: {arguments.prediction_folder}")
    try:
        if arguments.prediction_folder is not None:
            open_prediction_folder(arguments.prediction_folder, arguments.image_folder, arguments.reference_folder,
                                   not arguments.no_prefetch_previous, not arguments.no_prefetch_next,
                                   arguments.prefetch_radius)
        elif arguments.preprocessed_case:
            open_preprocessed_case(selected, not arguments.no_prefetch_previous, not arguments.no_prefetch_next,
                                   arguments.prefetch_radius)
        elif arguments.annotate:
            open_annotation(selected)
        elif arguments.validation_prediction:
            open_validation_prediction(selected, arguments.raw_dataset,
                                       not arguments.no_prefetch_previous, not arguments.no_prefetch_next,
                                       arguments.prefetch_radius)
        else:
            open_case(selected, not arguments.no_prefetch_previous, not arguments.no_prefetch_next,
                      arguments.prefetch_radius)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    except ImportError as error:
        raise SystemExit("Napari is not installed. Reinstall nnWorkbench to install the inspection dependency.") from error


if __name__ == "__main__":
    main()
