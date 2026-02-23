import numpy as np

import torch
from datasets import load_dataset  # type: ignore
from os import makedirs
from PIL import Image

from dinosaw.datasets.translate_featurise import translate_featurise
from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST
from dinosaw.utils import (
    closest_crop,
    convert_image,
    closest_resize,
    to_img,
    unnormalize,
)

from typing import cast
from time import time

FOLDER_NAME = "data"
DEVICE = "cuda:0"
NAME = "IN_reduced_dv2_dropped"
IMG_L = 518
N_VAL = 2600
HOMOGENISE = False
STRIDE = 14

np.random.seed(10001)
torch.random.manual_seed(10001)

ds = load_dataset("richwardle/reduced-imagenet", split="train")
ds = ds.shuffle()

torch.cuda.empty_cache()


for split in ("train", "val"):
    for which in ("imgs", "embeddings"):
        makedirs(f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/{which}/", exist_ok=True)

dv2 = PretrainedViTWrapper(
    MODEL_LIST[1],
    stride=STRIDE,
    add_flash_attn=False,
    device=DEVICE,
    # chk_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
)
dv2 = dv2.eval()

tr = closest_resize(IMG_L, IMG_L, STRIDE)


with torch.no_grad():
    N = len(ds)
    start_t = time()
    for i, dct in enumerate(ds):
        pil_img = dct["image"]
        pil_img = cast(Image.Image, pil_img).convert("RGB")

        img_tensor = convert_image(pil_img, tr, True, False, device_str=DEVICE)
        img_to_save = to_img(unnormalize(img_tensor.squeeze(0)))

        if HOMOGENISE:
            translated_feats = translate_featurise(
                img_tensor, dv2, step=STRIDE, mult=1, max_batch_size=128, device=DEVICE
            )
            translated_feats = translated_feats.to("cpu")
        else:
            with torch.no_grad():
                translated_feats = dv2.forward_features(img_tensor, make_2D=True)
                translated_feats[:, [47, 113, 117, 359], :, :]
                translated_feats = translated_feats.to("cpu")

        split = "val" if i < N_VAL else "train"

        img_to_save.save(f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/imgs/{i:05d}.png")
        torch.save(
            translated_feats,
            f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/embeddings/{i:05d}.pt",
        )

        if i % 50 == 0:
            end_t = time()
            print(f"[{i:05d}/{N}] in {end_t - start_t:03f}s")
            start_t = end_t
