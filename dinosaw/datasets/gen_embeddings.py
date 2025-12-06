import numpy as np

import torch
from datasets import load_dataset  # type: ignore
from os import makedirs
from PIL import Image

from dinosaw.datasets.translate_featurise import translate_featurise
from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST
from dinosaw.utils import closest_crop, convert_image

from typing import cast
from time import time

FOLDER_NAME = "data"
DEVICE = "cuda:1"
NAME = "IN_reduced"
IMG_L = 224
N_VAL = 2600

np.random.seed(10001)
torch.random.manual_seed(10001)

ds = load_dataset("richwardle/reduced-imagenet", split="train")
ds = ds.shuffle()

torch.cuda.empty_cache()


for split in ("train", "val"):
    for which in ("imgs", "embeddings"):
        makedirs(f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/{which}/", exist_ok=True)

dv2 = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=True, device=DEVICE)
dv2 = dv2.eval()

tr = closest_crop(IMG_L, IMG_L, 14)


N = len(ds)
start_t = time()
for i, dct in enumerate(ds):
    pil_img = dct["image"]
    pil_img = cast(Image.Image, pil_img).convert("RGB")

    img_tensor = convert_image(pil_img, tr, True, True, device_str=DEVICE)

    translated_feats = translate_featurise(img_tensor, dv2, step=14, mult=1, max_batch_size=128, device=DEVICE)
    translated_feats = translated_feats.to("cpu")

    split = "val" if i < N_VAL else "train"

    pil_img.save(f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/imgs/{i:05d}.png")
    torch.save(translated_feats, f"{FOLDER_NAME}/{NAME}_{IMG_L}/{split}/embeddings/{i:05d}.pt")

    if i % 50 == 0:
        end_t = time()
        print(f"[{i:05d}/{N}] in {end_t - start_t:03f}s")
        start_t = end_t
