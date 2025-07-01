import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import tqdm
import os
from skimage.segmentation import slic
from skimage.util import img_as_float
from osgeo import gdal, gdal_array


def correct_masks(noisy_masks: list[np.ndarray]):
    """
    将加噪后的 masks 转换为互不重叠、覆盖全图的 mask
    """
    N, H, W = len(noisy_masks), noisy_masks[0].shape[0], noisy_masks[0].shape[1]
    final_mask = np.zeros((H, W), dtype=int)  # 每个像素属于哪个类别（从 1 开始）

    for idx in range(N):
        mask = noisy_masks[idx]
        final_mask[mask] = idx + 1  # 类别从 1 开始编号

    # 填补未分配区域（可选）
    final_mask = fill_unassigned_pixels(final_mask)

    # 转换回 one-hot 格式
    corrected_masks = np.zeros((N, H, W), dtype=bool)
    for idx in range(N):
        corrected_masks[idx] = final_mask == (idx + 1)

    return corrected_masks


def fill_unassigned_pixels(final_mask: np.ndarray):
    """
    使用最近邻插值填补未被任何 mask 覆盖的空白区域
    """
    unassigned = final_mask == 0
    if not np.any(unassigned):
        return final_mask

    distances, indices = distance_transform_edt(
        unassigned, return_distances=True, return_indices=True
    )
    nearest_labels = final_mask[tuple(indices[:, unassigned])]
    final_mask[unassigned] = nearest_labels
    return final_mask


def masks_to_binary_array(masks):
    N = len(masks)
    H, W = masks[0]["segmentation"].shape
    binary_masks = np.zeros((N, H, W), dtype=bool)
    for i, mask in enumerate(masks):
        binary_masks[i] = mask["segmentation"]
    return binary_masks


def assign_classes_to_masks(binary_masks, ground_truth):
    """
    binary_masks: (N, H, W) bool array
    ground_truth: (H, W) int array，每个像素代表类别
    返回：(H, W) 的预测图 F
    """
    N, H, W = binary_masks.shape
    F = np.zeros((H, W), dtype=np.int64)

    for idx in range(N):
        mask = binary_masks[idx]
        if not np.any(mask):
            continue
        # 统计 mask 内部的类别直方图
        class_ids, counts = np.unique(ground_truth[mask], return_counts=True)
        dominant_class = class_ids[np.argmax(counts)]
        F[mask] = dominant_class

    return F


def binary_cls_gt_pred(gt, pred):
    result = np.zeros_like(gt)
    result[(gt == 1) & (pred == 1)] = 0
    result[(gt == 1) & (pred == 0)] = 1
    result[(gt == 0) & (pred == 1)] = 2
    result[(gt == 0) & (pred == 0)] = 3
    return result


def superpixel_segmentation(image, n_segments=500, compactness=10):
    """
    使用 SLIC 超像素分割图像
    image: HxWxC numpy array (RGB/IRRG)
    返回：HxW 的整数标签图，表示每个像素属于哪个对象
    """
    if image.dtype == np.uint8:
        image = img_as_float(image)
    segments = slic(
        image, n_segments=n_segments, compactness=compactness, start_label=1
    )
    return segments


def generate_binary_masks_from_segments(segments):
    """
    输入：segments - HxW 的整数标签图
    输出：N x H x W 的 bool 类型二值 mask 数组
    """
    unique_labels = np.unique(segments)
    masks = [(segments == label) for label in unique_labels if label != 0]
    return np.stack(masks)


def visual_sam_mask(masks, image):
    mask_p = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int16)
    for i, mask in enumerate(masks):
        mask_p[mask] = i + 1
    plt.imshow(mask_p, cmap="tab20")
    plt.colorbar()
    # plt.imshow(image, alpha=0.5)
    plt.show()


def merge_masks(masks, image_shape):
    merged_mask = np.zeros(image_shape[:2], dtype=np.int32)
    for idx, mask in enumerate(masks):
        merged_mask[mask["segmentation"]] = idx + 1  # 给每个 mask 分配唯一 ID
    return merged_mask


def fill_gaps(mask_image, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filled_mask = cv2.dilate(mask_image, kernel, iterations=iterations)
    filled_mask = cv2.erode(filled_mask, kernel, iterations=iterations)
    return filled_mask


def generate_mask(image, model: SamAutomaticMaskGenerator, gt=None):
    masks = model.generate(image)
    # masks = remove_small_regions(masks, 100)
    masks.sort(key=lambda x: np.sum(x["segmentation"]), reverse=True)
    for mask in masks:
        mask["segmentation"] = (
            fill_gaps(mask["segmentation"].astype(np.uint8), 3, 2) > 0
        )
    # masks = fill_gaps(masks, kernel_size=3, iterations=1)
    masks = merge_masks(masks, image.shape)
    masks = generate_binary_masks_from_segments(masks)
    # visual_sam_mask(masks, image)
    non_zero_masks = masks.sum(axis=0) > 0
    zero_mask = ~non_zero_masks
    if zero_mask.sum() > 100:
        slic_mask = slic(
            image,
            n_segments=100,
            compactness=10.0,
            max_num_iter=10,
            sigma=0,
            spacing=None,
            convert2lab=None,
            enforce_connectivity=True,
            min_size_factor=0.5,
            max_size_factor=3,
            slic_zero=False,
            start_label=1,
            mask=zero_mask,
            channel_axis=-1,
        )
        slic_mask = generate_binary_masks_from_segments(slic_mask)
        masks = np.concatenate([masks, slic_mask], axis=0)
    masks = remove_small_regions(masks, 2)

    return masks


def remove_small_regions(mask, area_threshold=20):
    masks = []
    idx = 0
    for i in range(mask.shape[0]):
        if np.sum(mask[i]) < area_threshold:
            continue
        m = mask[i].copy().astype(np.uint8)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        m = m.astype(np.int32)
        masks.append(m * (idx + 1))
        idx += 1
    masks = np.array(masks)
    masks = np.sum(masks, axis=0)
    masks = fill_unassigned_pixels(masks)
    masks = generate_binary_masks_from_segments(masks)
    return masks


def generate_psmask(img_list, gt_list, outdir, root=None):
    sam = sam_model_registry["vit_h"](
        checkpoint="/project/wangtong/segment-anything/notebooks/sam_vit_h_4b8939.pth"
    ).cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    from mmrgbx.datasets.transforms import AnyImageToRGB

    trans = AnyImageToRGB()

    for imgpth, gtpath in tqdm.tqdm(zip(img_list, gt_list), total=len(img_list)):
        # imgname = os.path.basename(imgpth)
        if root is not None:
            imgname = os.path.relpath(imgpth, root)
        else:
            imgname = os.path.basename(imgpth)
        imgname = os.path.splitext(imgname)[0]
        img = gdal_array.LoadFile(imgpth).squeeze()
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        img = trans.to_uint8(img)
        masks = generate_mask(img, mask_generator)
        gt = gdal_array.LoadFile(gtpath)
        # print(gt.shape)
        psmask = assign_classes_to_masks(masks, gt)
        outpth = os.path.join(outdir, imgname + ".png")
        os.makedirs(os.path.dirname(outpth), exist_ok=True)
        cv2.imwrite(os.path.join(outdir, imgname + ".png"), psmask)


def dfc23():
    img_list = os.listdir("/DFC23_IEEE-DataPort/track1/train/rgb")
    img_list = [fn for fn in img_list if fn.endswith("tif")]
    img_list = [
        os.path.join("/DFC23_IEEE-DataPort/track1/train/rgb", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("rgb", "mask") for fn in img_list]
    out_dir = "/DFC23_IEEE-DataPort/track1/train/rgbps"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)


def yesg_opt_sar():
    import glob

    img_list = glob.glob(
        "/YESeg-OPT-SAR/optical/**/*.png", recursive=True
    )
    ann_list = [fn.replace("optical", "label") for fn in img_list]
    out_dir = "/YESeg-OPT-SAR/opticalps"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(
        img_list, ann_list, out_dir, root="/YESeg-OPT-SAR/optical"
    )


import glob


def c2seg():
    root = "/C2SegClip/"
    for tag in ["train", "val"]:
        for suffix in [
            "_msi.tif",
        ]:
            img_list = glob.glob(os.path.join(root, tag, "*" + suffix))
            ann_list = [fn.replace(suffix, "_label.tif") for fn in img_list]
            out_dir = os.path.join(root, tag + "ps", suffix.replace(".tif", "")[1:])
            os.makedirs(out_dir, exist_ok=True)
            generate_psmask(img_list, ann_list, out_dir)


def potsdam():
    img_list = os.listdir("/PotsdamClip/img_dir/train")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("/PotsdamClip/img_dir/train", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("img_dir", "ann_dir") for fn in img_list]
    out_dir = "/PotsdamClip/imgps_dir/train"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)

    img_list = os.listdir("/PotsdamClip/img_dir/val")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("/PotsdamClip/img_dir/val", fn) for fn in img_list
    ]
    ann_list = [fn.replace("img_dir", "ann_dir") for fn in img_list]
    out_dir = "/PotsdamClip/imgps_dir/val"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)


def vaihingen():
    img_list = os.listdir("/VaihingenClip/img_dir/train")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("/VaihingenClip/img_dir/train", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("img_dir", "ann_dir") for fn in img_list]
    out_dir = "/VaihingenClip/imgps_dir/train"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)

    img_list = os.listdir("/VaihingenClip/img_dir/val")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("/VaihingenClip/img_dir/val", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("img_dir", "ann_dir") for fn in img_list]
    out_dir = "/VaihingenClip/imgps_dir/val"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c2seg")
    args = parser.parse_args()
    if args.dataset == "c2seg":
        c2seg()
    elif args.dataset == "dfc23":
        dfc23()
    elif args.dataset == "yesg_opt_sar":
        yesg_opt_sar()
    elif args.dataset == "potsdam":
        potsdam()
    elif args.dataset == "vaihingen":
        vaihingen()
