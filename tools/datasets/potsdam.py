import os
from .generate_mask import generate_psmask


def potsdam():
    img_list = os.listdir("D:/VaihingenClip/dep_dir/train")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("D:/VaihingenClip/dep_dir/train", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("dep_dir", "ann_dir") for fn in img_list]
    out_dir = "D:/VaihingenClip/ps_dir/train"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)

    img_list = os.listdir("D:/VaihingenClip/dep_dir/val")
    img_list = [fn for fn in img_list if fn.endswith("png")]
    img_list = [
        os.path.join("D:/VaihingenClip/dep_dir/val", fn)
        for fn in img_list
    ]
    ann_list = [fn.replace("dep_dir", "ann_dir") for fn in img_list]
    out_dir = "D:/VaihingenClip/ps_dir/val"
    os.makedirs(out_dir, exist_ok=True)
    generate_psmask(img_list, ann_list, out_dir)
