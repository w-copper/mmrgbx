import torch
import torch.nn as nn
import slidingwindow as sw
import numpy as np
import scipy.io as sio
import os
import h5py
from sklearn.decomposition import PCA
import cv2

from mmrgbx.datasets.transforms import AnyImageToRGB


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def band_normalization(data):
    """normalize the matrix to (0,1), r.s.t A axis (Default=0)
    return normalized matrix and a record matrix for normalize back
    """
    size = data.shape
    if len(size) != 3:
        raise ValueError("Unknown dataset")
    for i in range(size[-1]):
        _range = np.max(data[:, :, i]) - np.min(data[:, :, i])
        data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / _range
    return data


def read_data(dataset, pca_flag=False, band_norm=False):
    if dataset == "augsburg":
        num_classes, band = 14, 242
        train_file = "D:/project/mmrgbx-v1.0/data/data1/augsburg_multimodal.mat"
        col_train, row_train = 1360, 886
        valid_file = "D:/project/mmrgbx-v1.0/data/data1/berlin_multimodal.mat"
        col_valid, row_valid = 811, 2465
        input_data = sio.loadmat(train_file)
        valid_data = sio.loadmat(valid_file)

        hsi = input_data["HSI"]  # 886 1360 242
        hsi = hsi.astype(np.float32)
        msi = input_data["MSI"]
        msi = msi.astype(np.float32)
        sar = input_data["SAR"]
        sar = sar.astype(np.float32)
        label = input_data["label"]

        hsi_valid = valid_data["HSI"][:, :, 0:band]  # 2456 811 242
        hsi_valid = hsi_valid.astype(np.float32)
        msi_valid = valid_data["MSI"]
        msi_valid = msi_valid.astype(np.float32)
        sar_valid = valid_data["SAR"]
        sar_valid = sar_valid.astype(np.float32)
        label_valid = valid_data["label"]

        # a = np.min(label_valid)
        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(
                hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2])
            )  # 2456*811 242
            pca = PCA(n_components=10)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(
                hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_)
            )

            hsi_matrix = np.reshape(
                hsi_valid, (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2])
            )  # 2456*811 242
            pca = PCA(n_components=10)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(
                hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_)
            )

            band = 10

            del hsi_matrix

    elif dataset == "beijing":
        num_classes = 14
        band = 10  # 116
        ## beijing is training, wuhan is testing
        train_file = "D:/project/mmrgbx-v1.0/data/data2/beijing.mat"
        train_file_label = "D:/project/mmrgbx-v1.0/data/data2/beijing_label.mat"
        col_train, row_train = 13474, 8706
        valid_file = "D:/project/mmrgbx-v1.0/data/data2/wuhan.mat"
        valid_file_label = "D:/project/mmrgbx-v1.0/data/data2/wuhan_label.mat"
        col_valid, row_valid = 6225, 8670

        with h5py.File(train_file, "r") as f:
            f = h5py.File(train_file, "r")
        hsi = np.array(f["HSI"])
        msi = np.transpose(f["MSI"])
        sar = np.transpose(f["SAR"])
        # idx = np.where(np.isnan(sar))
        sar[1097, 8105, 1] = (
            sum(
                [
                    sar[1096, 8105, 1],
                    sar[1098, 8105, 1],
                    sar[1097, 8104, 1],
                    sar[1097, 8106, 1],
                ]
            )
            / 4
        )

        # cut beijing suburb region
        cut_length = 0
        col_train = col_train - cut_length
        hsi = hsi[:, :, cut_length // 3 :]
        msi = msi[cut_length:, :, :]
        sar = sar[cut_length:, :, :]

        # applying PCA for HSI # hsi (116, 2903, 4492)
        hsi_matrix = np.reshape(
            np.transpose(hsi), (hsi.shape[1] * hsi.shape[2], hsi.shape[0])
        )  # 2903*4492 116
        pca = PCA(n_components=10)
        pca.fit_transform(hsi_matrix)
        newspace = pca.components_
        newspace = newspace.transpose()  # 116*10
        hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2903*4492 10
        hsi_cube = np.transpose(
            np.reshape(hsi_matrix, (hsi.shape[2], hsi.shape[1], pca.n_components_))
        )
        del hsi

        mm = nn.Upsample(scale_factor=3, mode="nearest", align_corners=None)
        # upsample from 30m to 10m
        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi1 = np.transpose(hsi1)
        # remove extra pixels
        hsi = hsi1[:col_train, :row_train, :]
        del hsi1

        with h5py.File(train_file_label, "r") as f:
            f = h5py.File(train_file_label, "r")
        label = np.transpose(f["label"])
        # cut beijing label
        label = label[cut_length:, :]

        with h5py.File(valid_file, "r") as f:
            f = h5py.File(valid_file, "r")
        hsi_valid = np.array(f["HSI"])
        msi_valid = np.transpose(f["MSI"])
        sar_valid = np.transpose(f["SAR"])

        ## applying PCA for valid HSI
        hsi_matrix = np.reshape(
            np.transpose(hsi_valid),
            (hsi_valid.shape[1] * hsi_valid.shape[2], hsi_valid.shape[0]),
        )
        pca = PCA(n_components=10)
        pca.fit_transform(hsi_matrix)
        newspace = pca.components_
        newspace = newspace.transpose()
        hsi_matrix = np.matmul(hsi_matrix, newspace)
        hsi_cube = np.transpose(
            np.reshape(
                hsi_matrix, (hsi_valid.shape[2], hsi_valid.shape[1], pca.n_components_)
            )
        )
        del hsi_valid

        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi_valid = np.transpose(hsi1)
        del hsi1

        with h5py.File(valid_file_label, "r") as f:
            f = h5py.File(valid_file_label, "r")
        label_valid = np.transpose(f["label"])

    else:
        raise ValueError("Unknown dataset")

    # normalize data
    if band_norm:
        norm = band_normalization
    else:
        norm = normalization

    hsi = norm(hsi)
    msi = norm(msi)
    sar = norm(sar)
    hsi_valid = norm(hsi_valid)
    msi_valid = norm(msi_valid)
    sar_valid = norm(sar_valid)

    return (
        hsi,
        msi,
        sar,
        label,
        hsi_valid,
        msi_valid,
        sar_valid,
        label_valid,
        num_classes,
        band,
    )


def slide_crop_all_modalities(dataset, patch, overlay, outdir):
    (
        hsi,
        msi,
        sar,
        label,
        hsi_valid,
        msi_valid,
        sar_valid,
        label_valid,
        num_classes,
        band,
    ) = read_data(dataset)

    if dataset == "augsburg":
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == "beijing":
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")
    os.makedirs(os.path.join(outdir, "train"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "val"), exist_ok=True)
    # slide crop for train data
    # img = np.concatenate([hsi, msi, sar], axis=2)  # w, h, c
    # del hsi, msi, sar
    window_set_train = sw.generate(msi, sw.DimOrder.HeightWidthChannel, patch, overlay)
    msi = msi[:, :, :3]
    hsi = hsi[:, :, :3]
    trans = AnyImageToRGB()
    for window in enumerate(window_set_train):
        out_fname = f"{dataset}_{window[1].x}_{window[1].y}_{window[1].w}_{window[1].h}"

        sub_msi = msi[window[1].indices()]
        sub_msi = trans.to_uint8(sub_msi)
        sub_sar = sar[window[1].indices()]
        sub_sar = trans.to_uint8(sub_sar)
        sub_label = label[window[1].indices()]
        sub_hsi = hsi[window[1].indices()]
        sub_hsi = trans.to_uint8(sub_hsi)
        # print(sub_msi.shape)
        # print(sub_msi.dtype)
        cv2.imwrite(os.path.join(outdir, "train", f"{out_fname}_msi.tif"), sub_msi)
        cv2.imwrite(os.path.join(outdir, "train", f"{out_fname}_sar.tif"), sub_sar)
        cv2.imwrite(os.path.join(outdir, "train", f"{out_fname}_hsi.tif"), sub_hsi)
        cv2.imwrite(os.path.join(outdir, "train", f"{out_fname}_label.tif"), sub_label)

    window_set_valid = sw.generate(msi_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    msi_valid = msi_valid[:, :, [3, 1, 2]]
    hsi_valid = hsi_valid[:, :, :3]

    for window in enumerate(window_set_valid):
        out_fname = f"{dataset}_{window[1].x}_{window[1].y}_{window[1].w}_{window[1].h}"
        sub_msi = msi_valid[window[1].indices()]
        sub_msi = trans.to_uint8(sub_msi)
        sub_sar = sar_valid[window[1].indices()]
        sub_sar = trans.to_uint8(sub_sar)
        sub_label = label_valid[window[1].indices()]
        sub_hsi = hsi_valid[window[1].indices()]
        sub_hsi = trans.to_uint8(sub_hsi)
        cv2.imwrite(os.path.join(outdir, "val", f"{out_fname}_msi.tif"), sub_msi)
        cv2.imwrite(os.path.join(outdir, "val", f"{out_fname}_sar.tif"), sub_sar)
        cv2.imwrite(os.path.join(outdir, "val", f"{out_fname}_hsi.tif"), sub_hsi)
        cv2.imwrite(os.path.join(outdir, "val", f"{out_fname}_label.tif"), sub_label)


slide_crop_all_modalities("beijing", 512, 200, "D:/C2SegClip")
slide_crop_all_modalities("augsburg", 512, 200, "D:/C2SegClip")
