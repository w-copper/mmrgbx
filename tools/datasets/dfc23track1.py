from pycocotools import coco
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tqdm


def run_mask():
    COCO = coco.COCO(
        "D:/DFC23_IEEE-DataPort/track1/roof_fine_train.json"
    )

    train_image_root = "D:/DFC23_IEEE-DataPort/track1/train/rgb"
    mask_root = "D:/DFC23_IEEE-DataPort/track1/train/mask"

    os.makedirs(mask_root, exist_ok=True)

    for imgid in tqdm.tqdm(COCO.getImgIds()):
        img_info = COCO.loadImgs(imgid)
        filename = img_info[0]["file_name"]
        annids = COCO.getAnnIds(imgIds=imgid)
        anns = COCO.loadAnns(annids)
        mask = np.zeros((img_info[0]["height"], img_info[0]["width"]), dtype=np.uint8)

        for ann in anns:
            bmask = COCO.annToMask(ann)
            mask[bmask == 1] = ann["category_id"]

        mask = mask.astype(np.uint8)
        cv2.imwrite(os.path.join(mask_root, filename), mask)


def run_split(root):
    img_dir = os.path.join(root, "rgb")
    img_list = os.listdir(img_dir)
    img_list = [f[:-4] for f in img_list if f.endswith(".tif")]
    train_part = img_list[: int(len(img_list) * 0.7)]
    val_part = img_list[int(len(img_list) * 0.7) :]

    with open(os.path.join(root, "train.txt"), "w") as f:
        for img in train_part:
            f.write(img + "\n")
    with open(os.path.join(root, "val.txt"), "w") as f:
        for img in val_part:
            f.write(img + "\n")


if __name__ == "__main__":
    run_mask()
    # run_split("D:/DFC23_IEEE-DataPort/track1/train")
    # COCO = coco.COCO(
    #     "D:/DFC23_IEEE-DataPort/track1/roof_fine_train.json"
    # )

    # ids = COCO.getCatIds()
    # print(ids)
    # print(COCO.loadCats(ids))
    # import matplotlib.pyplot as plt

    # cmap = plt.get_cmap("tab20")

    # for cat in COCO.loadCats(ids):
    #     rgb = cmap(cat["id"] - 1)
    #     rgb = rgb[:3]
    #     rgb = [int(c * 255) for c in rgb]
    #     rgb = tuple(rgb)

    #     print(cat["name"], rgb)
