import numpy as np

import torch
from datasets import load_dataset
from os import makedirs
from PIL import Image

from dinosaw.datasets.translate_featurise import translate_featurise
from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST
from dinosaw.utils import closest_crop, convert_image

from typing import cast, Literal
from time import time

DEVICE = "cuda:1"
NAME = "IN_reduced"
IMG_L = 224
SPLIT: Literal["train", "val"] = "train"

np.random.seed(10001)
torch.random.manual_seed(10001)

ds = load_dataset("richwardle/reduced-imagenet", split=SPLIT)
ds = ds.shuffle()

torch.cuda.empty_cache()


for split in ("train", "val"):
    for which in ("imgs", "embeddings"):
        makedirs(f"Dataset/{split}/{which}/", exist_ok=True)

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

    pil_img.save(f"Dataset/{NAME}_{IMG_L}/{SPLIT}/imgs/{i:05d}.png")
    torch.save(translated_feats, f"Dataset/{NAME}_{IMG_L}/{SPLIT}/embeddings/{i:05d}.pt")

    if i % 50 == 0:
        end_t = time()
        print(f"[{i:05d}/{N}] in {end_t - start_t:03f}s")
        start_t = end_t
