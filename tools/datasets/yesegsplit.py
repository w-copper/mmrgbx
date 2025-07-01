import os
import random


def yesegsplit(root):
    img_dir = os.path.join(root, "label")

    subdirs = os.listdir(img_dir)

    train_dir = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "14",
        "15",
        "16",
        "39_A",
        "GY_A",
        "HY_A",
        "HY_C",
    ]
    val_dir = []
    for subdir in subdirs:
        if subdir not in train_dir:
            val_dir.append(subdir)

    train_list = []
    val_list = []
    for subdir in train_dir:
        subdir_path = os.path.join(img_dir, subdir)
        files = os.listdir(subdir_path)
        for file in files:
            if file.endswith(".png"):
                train_list.append(
                    os.path.join(subdir, file).removesuffix(".png").replace("\\", "/")
                )

    for subdir in val_dir:
        subdir_path = os.path.join(img_dir, subdir)
        files = os.listdir(subdir_path)
        for file in files:
            if file.endswith(".png"):
                val_list.append(
                    os.path.join(subdir, file).removesuffix(".png").replace("\\", "/")
                )

    with open(os.path.join(root, "train.txt"), "w") as f:
        for item in train_list:
            f.write("%s\n" % item)

    with open(os.path.join(root, "val.txt"), "w") as f:
        for item in val_list:
            f.write("%s\n" % item)


yesegsplit("D:/YESeg-OPT-SAR")
