import warnings

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
    nnUNetTrainerDA5ord0,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01,
    nnUNetTrainerNoMirroring,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetDataLoaderBetterIgnoreSampling(nnUNetDataLoader):
    """Sample annotated areas with randomly offset patch centres.

    For datasets with an ignore label, regular samples are drawn from annotated
    voxels rather than arbitrary unannotated background. The selected location
    is randomly offset within the image bounds to increase patch diversity.
    """
    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: dict | None,
        overwrite_class = None,
        verbose: bool = False,
    ):
        # Increase padding for images smaller than the requested patch, then
        # compute the valid lower and upper crop bounds.
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]

        # Without an ignore label, regular samples can be cropped anywhere.
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            # With ignore labels, regular samples start in annotated image
            # content. annotated_classes_key represents all non-ignore labels.
            if class_locations is None:
                selected_class = None
            elif not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations.get(selected_class, ())) == 0:
                    warnings.warn("Warning! No annotated pixels in image!")
                    selected_class = None
            elif force_fg:
                # Choose a foreground class or region present in this case. When
                # real foreground classes are available, do not oversample the
                # aggregate annotated-area key instead.
                assert class_locations is not None, "if force_fg is set class_locations cannot be None"
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), (
                        'desired class ("overwrite_class") does not have class_locations (missing key)'
                    )

                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                annotated_key_matches = [
                    i == self.annotated_classes_key if isinstance(i, tuple) else False
                    for i in eligible_classes_or_regions
                ]
                if any(annotated_key_matches) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(annotated_key_matches)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print("case does not contain any foreground classes")
                else:
                    selected_class = (
                        eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
                        if overwrite_class is None or overwrite_class not in eligible_classes_or_regions
                        else overwrite_class
                    )
            else:
                raise RuntimeError("unexpected foreground sampling state")

            voxels_of_that_class = (
                class_locations[selected_class]
                if class_locations is not None and selected_class is not None
                else None
            )

            if voxels_of_that_class is not None:
                # Copy before applying offsets so cached class locations remain
                # unchanged for subsequent samples.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))].copy()

                if self.has_ignore and not force_fg:
                    # Offset regular annotated samples by up to half a patch in
                    # each dimension. This avoids repeatedly placing the same
                    # annotation in the patch centre.
                    allowed_max_neg_offset = [
                        min(s, p // 2) for s, p in zip(selected_voxel[1:], self.patch_size)
                    ]
                    allowed_max_pos_offset = [
                        min(d - 1 - s, p // 2)
                        for s, p, d in zip(selected_voxel[1:], self.patch_size, data_shape)
                    ]
                    for d in range(len(self.patch_size)):
                        offset_low = -allowed_max_neg_offset[d]
                        offset_high = allowed_max_pos_offset[d]
                        selected_voxel[d + 1] += np.random.randint(offset_low, offset_high + 1)

                # Convert the selected centre to lower crop coordinates.
                # selected_voxel[0] is the class identifier; spatial coordinates
                # therefore start at index 1.
                bbox_lbs = [
                    max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
            else:
                # Fall back to a random crop when no usable location is present.
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs


class nnUNetTrainerBetterIgnoreSampling(nnUNetTrainer):
    def get_dataloaders(self):
        """Create standard nnU-Net augmentation pipelines with custom sampling."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, tr_transforms, val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(
        self,
        initial_patch_size: tuple[int, ...],
        tr_transforms: BasicTransform,
        val_transforms: BasicTransform,
    ):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Preserve nnU-Net's usual transform and oversampling configuration while
        # using the custom patch-location strategy.
        dl_tr = nnUNetDataLoaderBetterIgnoreSampling(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        # Validation does not require the larger augmentation crop.
        dl_val = nnUNetDataLoaderBetterIgnoreSampling(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        return dl_tr, dl_val


class nnUNetTrainerBetterIgnoreSamplingDA5(nnUNetTrainerBetterIgnoreSampling, nnUNetTrainerDA5):
    """DA5 augmentation with better-ignore sampling."""


class nnUNetTrainerBetterIgnoreSamplingDA5ord0(nnUNetTrainerBetterIgnoreSampling, nnUNetTrainerDA5ord0):
    """Order-0 DA5 augmentation with better-ignore sampling."""


class nnUNetTrainerBetterIgnoreSamplingOnlyMirror01(
    nnUNetTrainerBetterIgnoreSampling, nnUNetTrainer_onlyMirror01
):
    """Mirror only axes 0 and 1 with better-ignore sampling."""


class nnUNetTrainerBetterIgnoreSamplingNoMirroring(
    nnUNetTrainerBetterIgnoreSampling, nnUNetTrainerNoMirroring
):
    """Disable mirroring augmentation while retaining better-ignore sampling."""


class nnUNetTrainerBetterIgnoreSampling_10epochs(nnUNetTrainerBetterIgnoreSampling):
    """Ten-epoch variant for quick debugging runs."""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10
