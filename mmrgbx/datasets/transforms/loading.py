# Copyright (c) Open-CD. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from mmrgbx.registry import TRANSFORMS
from osgeo import gdal_array


@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile(MMCV_LoadImageFromFile):
    """Load an image pair from files.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filenames = results["img_path"]
        imgs = []
        imgs_bands = []
        try:
            for filename in filenames:
                img = gdal_array.LoadFile(filename)
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                imgs_bands.append(img.shape[0])
                img = np.transpose(img, (1, 2, 0))
                imgs.append(img)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results["img"] = imgs
        results["img_shape"] = imgs[0].shape[:2]
        results["ori_shape"] = imgs[0].shape[:2]
        results["imgs_bands"] = imgs_bands
        return results


@TRANSFORMS.register_module()
class MultiImgLoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for change detection provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of change detection ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of change detection ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 255. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend="pillow",
        with_ps=False,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn(
                "`reduce_zero_label` will be deprecated, "
                "if you would like to ignore the zero label, please "
                "set `reduce_zero_label=True` when dataset "
                "initialized"
            )
        self.imdecode_backend = imdecode_backend
        self.with_ps = with_ps

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        gt_semantic_seg = (
            gdal_array.LoadFile(results["seg_map_path"]).squeeze().astype(np.uint8)
        )
        # import pdb

        # pdb.set_trace()
        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results["reduce_zero_label"]
        assert self.reduce_zero_label == results["reduce_zero_label"], (
            "Initialize dataset with `reduce_zero_label` as "
            f"{results['reduce_zero_label']} but when load annotation "
            f"the `reduce_zero_label` is {self.reduce_zero_label}"
        )
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg
        results["seg_fields"].append("gt_seg_map")

    def _load_ps_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        ps_maps = []
        for path in results["ps_path"]:
            gt_semantic_seg = gdal_array.LoadFile(path).squeeze().astype(np.uint8)
            # import pdb

            # pdb.set_trace()
            # reduce zero_label
            if self.reduce_zero_label is None:
                self.reduce_zero_label = results["reduce_zero_label"]
            assert self.reduce_zero_label == results["reduce_zero_label"], (
                "Initialize dataset with `reduce_zero_label` as "
                f"{results['reduce_zero_label']} but when load annotation "
                f"the `reduce_zero_label` is {self.reduce_zero_label}"
            )
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_semantic_seg[gt_semantic_seg == 0] = 255
                gt_semantic_seg = gt_semantic_seg - 1
                gt_semantic_seg[gt_semantic_seg == 254] = 255
            # modify if custom classes
            if results.get("label_map", None) is not None:
                # Add deep copy to solve bug of repeatedly
                # replace `gt_semantic_seg`, which is reported in
                # https://github.com/open-mmlab/mmsegmentation/pull/1445/
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                for old_id, new_id in results["label_map"].items():
                    gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
            ps_maps.append(gt_semantic_seg)
        # results["ps_seg_map"] = ps_maps
        for i in range(len(ps_maps)):
            results["seg_fields"].append(f"ps_seg_map_{i}")
            results[f"ps_seg_map_{i}"] = ps_maps[i]

        results["ps_seg_map_count"] = len(ps_maps)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """
        if "seg_fields" not in results:
            results["seg_fields"] = []
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        if self.with_ps:
            self._load_ps_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str
