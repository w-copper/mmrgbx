# Copyright (c) Open-CD. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
from mmrgbx.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AnyImageToRGB(BaseTransform):
    """Convert any image type to byte.

    Args:
        channel_order (str): Order of channel. Options are 'bgr' and 'rgb'.
            Default: 'bgr'.
    """

    def to_uint8(self, oimg):
        if len(oimg.shape) == 2:
            oimg = oimg[:, :, np.newaxis]
            oimg = np.repeat(oimg, 3, axis=2)

        if oimg.shape[2] == 1:
            oimg = np.repeat(oimg, 3, axis=2)
        if oimg.shape[2] == 2:
            oimg = np.concatenate([oimg, np.mean(oimg, axis=2, keepdims=True)], axis=2)
        if oimg.dtype == np.uint8:
            return oimg
        if oimg.shape[2] > 3:
            oimg = oimg[:, :, :3]
        ori_shape = oimg.shape
        img = oimg.flatten()
        nan_mask = np.isnan(img)
        img = img.astype(np.float32)
        valid_value = img[~nan_mask]
        if len(valid_value) <= 100:
            return np.zeros(ori_shape, dtype=np.uint8)
        else:
            _vmin, _vmax = np.percentile(valid_value, [1, 99])
            img[img < _vmin] = _vmin
            img[img > _vmax] = _vmax
            img[nan_mask] = _vmin
            img = (img - _vmin) / ((_vmax - _vmin) + 1e-5)
            img = img.reshape(ori_shape)
            img = (img * 255).astype(np.uint8)
            return img

    def transform(self, results):
        imgs = results["img"]
        for i in range(len(imgs)):
            results["img"][i] = self.to_uint8(imgs[i])

        return results


@TRANSFORMS.register_module()
class MultiImgPackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        # print(results["img_path"])
        if "img" in results:

            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img

            imgs = [_transform_img(img) for img in results["img"]]

            imgs = torch.cat(imgs, axis=0)  # -> (3*N, H, W)
            packed_results["inputs"] = imgs

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results["gt_seg_map"][None, ...].astype(np.int64))
            )
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        ps_count = results.get("ps_seg_map_count", 0)
        if ps_count > 0:
            ps_segs = []
            for i in range(ps_count):
                ps_segs.append(
                    to_tensor(
                        results["ps_seg_map_" + str(i)][None, ...].astype(np.int64)
                    )
                )

            ps_segs_data = dict(data=torch.cat(ps_segs, axis=0))
            data_sample.set_data(dict(ps_seg_map=PixelData(**ps_segs_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
