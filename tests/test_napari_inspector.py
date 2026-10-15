from pathlib import Path
from types import SimpleNamespace

import pytest

from nnworkbench.services.napari_inspector import (
    _CaseData,
    _display_case,
    _LayerData,
    _read_case,
    annotation_label_summary,
    annotation_source_image,
    annotation_target_path,
    case_id,
    image_case_ids,
    prediction_case_ids,
    prefetch_indices,
    preprocessed_case_id,
    preprocessed_case_ids,
    related_files,
    save_semantic_annotation,
    semantic_map_name,
)


def test_related_files_groups_raw_case_across_folders(tmp_path: Path) -> None:
    for folder, names in {
        "imagesTr": ("case_001_0000.nii.gz", "case_001_0001.nii.gz", "case_002_0000.nii.gz"),
        "labelsTr": ("case_001.nii.gz", "case_002.nii.gz"),
    }.items():
        directory = tmp_path / folder; directory.mkdir()
        for name in names:
            (directory / name).touch()

    files = related_files(tmp_path / "labelsTr" / "case_001.nii.gz")

    assert case_id(tmp_path / "imagesTr" / "case_001_0000.nii.gz") == "case_001"
    assert [path.name for path in files.images] == ["case_001_0000.nii.gz", "case_001_0001.nii.gz"]
    assert [path.name for path in files.labels] == ["case_001.nii.gz"]
    assert image_case_ids(tmp_path) == ("case_001", "case_002")


def test_prediction_case_ids_uses_predictions_not_the_number_of_input_images(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"; images = tmp_path / "images"
    predictions.mkdir(); images.mkdir()
    for name in ("case_001.nii.gz", "case_002.nii.gz"):
        (predictions / name).touch()
    for name in ("case_001_0000.nii.gz", "case_002_0000.nii.gz", "case_003_0000.nii.gz"):
        (images / name).touch()

    assert prediction_case_ids(predictions) == ("case_001", "case_002")


def test_preprocessed_case_ids_excludes_segmentation_companions(tmp_path: Path) -> None:
    session = tmp_path / "nnUNetPlans_3d_fullres"; session.mkdir()
    for name in ("001.b2nd", "001_seg.b2nd", "002.b2nd", "dataset.json"):
        (session / name).touch()

    assert preprocessed_case_id(session / "001_seg.b2nd") == "001"
    assert preprocessed_case_ids(session) == ("001", "002")


def test_annotation_source_image_uses_the_first_image_channel(tmp_path: Path) -> None:
    images = tmp_path / "imagesTr"; labels = tmp_path / "labelsTr"
    images.mkdir(); labels.mkdir()
    (images / "case_001_0001.nii.gz").touch()
    primary = images / "case_001_0000.nii.gz"; primary.touch()
    label = labels / "case_001.nii.gz"; label.touch()

    assert annotation_source_image(label) == primary


def test_annotation_label_summary_reserves_zero_for_ignore(tmp_path: Path) -> None:
    images = tmp_path / "imagesTr"; images.mkdir()
    image = images / "case_001_0000.nii.gz"; image.touch()
    (tmp_path / "dataset.json").write_text(
        '{"labels": {"background": 0, "tumour": 1, "ignore": 2}}', encoding="utf-8"
    )

    assert annotation_label_summary(image) == ((0, "ignore"), (1, "background"), (2, "tumour"))


def test_prefetch_indices_prioritise_nearest_next_then_previous() -> None:
    assert prefetch_indices(5, 10, True, True, 2) == (6, 4, 7, 3)
    assert prefetch_indices(0, 10, True, True, 2) == (1, 2)
    assert prefetch_indices(5, 10, False, True, 2) == (6, 7)


def test_read_case_uses_mitk_for_nifti_layers(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    images = tmp_path / "imagesTr"; images.mkdir()
    mitk.Image(np.ones((2, 2, 2), dtype=np.float32)).save(images / "case_001_0000.nii.gz")

    case = _read_case(tmp_path, "case_001")

    assert [(layer.name, layer.kind) for layer in case.layers] == [("Image — case_001_0000.nii.gz", "image")]


def test_read_case_names_layers_by_role(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    for folder, name in (("imagesTr", "case_001_0000.nii.gz"), ("labelsTr", "case_001.nii.gz")):
        (tmp_path / folder).mkdir()
        mitk.Image(np.ones((2, 2, 2), dtype=np.uint8)).save(tmp_path / folder / name)

    case = _read_case(tmp_path, "case_001")

    assert [layer.name for layer in case.layers] == [
        "Image — case_001_0000.nii.gz", "Label — case_001.nii.gz",
    ]


def test_save_semantic_annotation_uses_case_name_and_source_geometry(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    images = tmp_path / "imagesTr"; images.mkdir()
    image_path = images / "Case_001_0000.nii.gz"
    mitk.Image(np.zeros((2, 3, 4), dtype=np.uint8), spacing=(1.1, 2.2, 3.3), origin=(4, 5, 6)).save(image_path)
    target = annotation_target_path(image_path)
    annotation = np.ones((2, 3, 4), dtype=np.uint8)
    viewer = SimpleNamespace(layers={semantic_map_name(image_path): SimpleNamespace(data=annotation)})

    save_semantic_annotation(viewer, image_path, target)

    saved = mitk.Image(target)
    assert target.name == "Case_001.nii.gz"
    assert np.array_equal(np.asarray(saved), annotation)
    assert saved.spacing == pytest.approx((1.1, 2.2, 3.3))
    assert saved.origin == pytest.approx((4.0, 5.0, 6.0))


def test_case_display_reuses_layers_and_updates_data_and_geometry() -> None:
    class Layer:
        def __init__(self, data, name, affine=None) -> None:
            self.data, self.name, self.affine = data, name, affine
            self.visible, self.metadata = True, {}

    class Viewer:
        def __init__(self) -> None:
            self.layers = []
            self.added = 0

        def add_image(self, data, name, affine=None):
            self.added += 1; layer = Layer(data, name, affine); self.layers.append(layer); return layer

        def add_labels(self, data, name, affine=None):
            return self.add_image(data, name, affine)

    viewer = Viewer()
    first = _CaseData((_LayerData("case_001_0000.nii.gz", "image", "first", "first-affine"),))
    second = _CaseData((_LayerData("case_002_0000.nii.gz", "image", "second", "second-affine"),))

    _display_case(viewer, first)
    first_layer = viewer.layers[0]
    _display_case(viewer, second)

    assert viewer.added == 1
    assert viewer.layers[0] is first_layer
    assert (first_layer.name, first_layer.data, first_layer.affine) == ("case_002_0000.nii.gz", "second", "second-affine")


def test_case_display_resets_z_slider_to_middle_slice() -> None:
    np = pytest.importorskip("numpy")

    class Dims:
        ndim = 3

        def set_current_step(self, axis, value) -> None:
            self.axis, self.value = axis, value

    class Viewer:
        def __init__(self) -> None:
            self.layers, self.dims = [], Dims()

        def add_image(self, data, name, affine=None):
            layer = type("Layer", (), {"data": data, "name": name, "affine": affine, "visible": True,
                                         "metadata": {}})()
            self.layers.append(layer); return layer

        add_labels = add_image

    viewer = Viewer()
    _display_case(viewer, _CaseData((_LayerData("case.nii.gz", "image", np.zeros((9, 4, 4))),)))

    assert (viewer.dims.axis, viewer.dims.value) == (0, 4)


def test_read_image_supports_2d_images(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    import numpy as np

    from nnworkbench.services.napari_inspector import _read_image

    path = tmp_path / "case_001_0000.nii.gz"
    mitk.Image(np.zeros((3, 4), dtype=np.float32), spacing=(0.5, 0.25, 1.0), origin=(3, 4, 0)).save(path)

    data, affine = _read_image(path)

    assert data.shape == (3, 4)
    assert affine.shape == (3, 3)
    assert np.allclose(np.diag(affine)[:2], (0.25, 0.5))
    assert np.allclose(affine[:2, -1], (4, 3))


def test_related_files_accept_non_nifti_images(tmp_path: Path) -> None:
    images = tmp_path / "imagesTr"; images.mkdir()
    labels = tmp_path / "labelsTr"; labels.mkdir()
    (images / "case_001_0000.png").touch()
    (labels / "case_001.png").touch()

    files = related_files(images / "case_001_0000.png")

    assert case_id(images / "case_001_0000.png") == "case_001"
    assert [path.name for path in files.images] == ["case_001_0000.png"]
    assert [path.name for path in files.labels] == ["case_001.png"]
