import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper
from dinosaw.comaprisons.denoising_vits import DenoisingViTWrapper
from dinosaw.utils import to_numpy, closest_resize, convert_image

from typing import Literal, get_args

import matplotlib.pyplot as plt
from matplotlib import font_manager


ModelTypes = Literal[
    "dv2",
    "dv2_b",
    "dv2_cb",
    "dvt",
    "alibi_dv2",
    "alibi_dv2_h",
    "alibi_dv2_cb",
    "alibi_dv2_cb_s_l",
    "alibi_dv2_cb_nr_l",
    "alibi_dv2_cb_l_j",
    "alibi_dv2_coco",
    "nope",
    "dv",
    "dv_b",
    "dv3",
    "vit_b",
    "clip_b",
    "eva02_b",
    "sam_b",
    "convnext",
    "vit_b_in",
    "deit",
    "vit_t_in",
]
TimmModels = Literal[
    "dv2",
    "dv2_b",
    "dv2_cb",
    "dv",
    "dv_b",
    "dv3",
    "vit_b",
    "clip_b",
    "eva02_b",
    "sam_b",
    "convnext",
    "vit_b_in",
    "deit",
    "vit_t_in",
]
timm_models = get_args(TimmModels)
model_types: tuple[ModelTypes] = get_args(ModelTypes)
model_names: dict[ModelTypes, str] = {
    "dv2": "DINOv2",
    "dv2_b": "DINOv2-B",
    "dvt": "DVT",
    "alibi_dv2": "ALiBi-Dv2",
    "alibi_dv2_h": "ALiBi(H)-Dv2",
    "alibi_dv2_cb": "ALiBi(CB)-Dv2",
    "alibi_dv2_cb_s_l": "ALiBi(CB-S-L)-Dv2",
    "alibi_dv2_cb_nr_l": "ALiBi(CB-NR-L)-Dv2",
    "alibi_dv2_cb_l_j": "ALiBi(CB-L-J)-Dv2",
    "alibi_dv2_coco": "ALiBi(COCO)-Dv2",
    "nope": "NoPE",
    "dv": "DINO",
    "dv_b": "DINO-B",
    "dv3": "DINOv3",
    "clip_b": "CLIP-B",
    "eva02_b": "EVA02-B",
    "sam_b": "SAM-B",
    "convnext": "",
    "vit_b": "ViT-B",
    "vit_b_in": "ViT-B-INet",
    "deit": "",
    "vit_t_in": "",
}
model_chkpoints: dict[ModelTypes, str] = {
    "dv2": "",
    "dv2_b": "",
    "dvt": "dvt.pth",
    "alibi_dv2": "alibi_dv2_vits14_reg.pth",
    "alibi_dv2_h": "alibi_homog_dv2_vits14_reg.pth",
    "alibi_dv2_cb": "alibi_cb_dv2_vits14_reg.pth",
    "alibi_dv2_cb_s_l": "alibi_scratch_cb_dv2_vits14_reg_learned_m.pth",
    "alibi_dv2_cb_nr_l": "alibi_cb_dv2_vits14_noreg.pth",
    "alibi_dv2_cb_l_j": "alibi_cb_dv2_vits14_j_ms.pth",
    "alibi_dv2_coco": "alibi_coco_dv2_vits14_reg_ms.pth",
    "nope": "nope_dv2_vits14_reg.pth",
    "dv": "",
    "dv_b": "",
    "dv3": "dinov3_vits_patch16_plus_reg4.pth",
    "vit_b": "",
    "clip_b": "",
    "eva02_b": "",
    "sam_b": "",
    "convnext": "",
    "vit_b_in": "",
    "deit": "",
    "vit_t_in": "",
}
model_name_to_timm: dict[TimmModels, str] = {
    "dv2": MODEL_LIST[1],
    "dv2_b": MODEL_LIST[3],
    "dv": MODEL_LIST[5],
    "dv_b": MODEL_LIST[14],
    "dv3": MODEL_LIST[4],
    "vit_b": MODEL_LIST[6],
    "clip_b": MODEL_LIST[7],
    "eva02_b": MODEL_LIST[8],
    "sam_b": MODEL_LIST[9],
    "convnext": MODEL_LIST[10],
    "vit_b_in": MODEL_LIST[11],
    "deit": MODEL_LIST[12],
    "vit_t_in": MODEL_LIST[13],
    "dv2_cb": MODEL_LIST[1],
}


def get_model(
    model_type: ModelTypes,
    model_dir: str,
    device: str,
    to_half: bool = False,
    conf_path: str | None = None,
    remove_final_norm: bool = False,
) -> PretrainedViTWrapper:
    S = 14
    model: PretrainedViTWrapper | None = None
    conf_path = conf_path if "dv3" in model_type else None

    if model_type in timm_models:
        model_id = model_name_to_timm[model_type]
        model_chk = model_chkpoints[model_type] if model_type == "dv3" else None
        S = 16 if "patch16" in model_id else 14
        model = PretrainedViTWrapper(
            model_id,
            stride=S,
            add_flash_attn=False,
            device=device,
            chk_path=model_chk,
            conf_path=conf_path,
        )
        model = model.eval()
        if to_half:
            model = model.half()
        return model

    chk_path = f"{model_dir}/{model_chkpoints[model_type]}"
    weights = torch.load(chk_path, weights_only=True, map_location=device)
    if model_type == "nope":
        model = PretrainedViTWrapper(
            MODEL_LIST[1],
            stride=S,
            add_flash_attn=False,
            device=device,
        )
        model.load_state_dict(weights)
    elif model_type == "dvt":
        model = DenoisingViTWrapper(
            chk_path,
            MODEL_LIST[1],
            stride=S,
            add_flash_attn=False,
            device=device,
        )
    elif "alibi_dv2" in model_type:
        slope_type = "learned" if "_l" in model_type else "constant"
        add_cls = False if "nr" in model_type else True
        n_reg_tokens = 0 if "nr" in model_type else 4
        jitter_mag = 0.025 if "_j" in model_type else 0.0

        model = AlibiVitWrapper(
            MODEL_LIST[1],
            stride=S,
            add_flash_attn=False,
            device=device,
            slope_type=slope_type,
            normalize=True,
            wrap=True,
            add_cls=add_cls,
            n_reg_tokens=n_reg_tokens,
            jitter_mag=jitter_mag,
        )
        model.load_state_dict(weights)
    else:
        raise Exception("Invalid model type")

    if remove_final_norm:
        model.model.norm = nn.Identity()

    assert model is not None
    model = model.eval()
    if to_half:
        model = model.half()

    return model


def get_models(
    model_types: tuple[ModelTypes, ...],
    model_dir: str,
    device: str,
    to_half: bool = False,
    conf_path: str | None = None,
) -> dict[ModelTypes, PretrainedViTWrapper]:
    return {k: get_model(k, model_dir, device, to_half, conf_path=conf_path) for k in model_types}


def get_features(
    model: PretrainedViTWrapper,
    pil_img: Image.Image,
    channel_last: bool = False,
    channel_blank: bool = False,
    to_half: bool = False,
    device: str = "cuda:0",
    S: int = 14,
) -> np.ndarray:
    tr = closest_resize(pil_img.height, pil_img.width, model.stride)
    img_tensor = convert_image(pil_img, tr, device_str=device, to_half=to_half)
    with torch.no_grad():
        emb = model.forward_features(img_tensor, make_2D=True)
    emb_np = to_numpy(emb.squeeze(0))
    if channel_blank:
        channels_to_blank = [47, 113, 117, 359]
        emb_np[channels_to_blank, :, :] = 0
    if channel_last:
        emb_np = np.transpose(emb_np, (1, 2, 0))

    return emb_np


def add_custom_font(font_folder: str, font_name: str = "Grotesk") -> None:
    try:
        font_paths = (f"{font_folder}/{font_name}.ttf", f"{font_folder}/{font_name}-Bold.ttf")
        for font_path in font_paths:
            font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_paths[0])

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = prop.get_name()
    except Exception as e:
        print(f"Can't load custom font: {e} ")
