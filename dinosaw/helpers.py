import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from dinosaw.wrappers import PretrainedViTWrapper, MODEL_LIST
from PVW import WrapperRegistry
from dinosaw.utils import to_numpy, closest_resize, convert_image

from typing import Literal, cast, get_args

import matplotlib.pyplot as plt
from matplotlib import font_manager


ModelTypes = Literal[
    "dv2",
    "dv2_b",
    "dv2_cb",
    "dv2_db",
    "dvt",
    "sinusoid_dv2",
    "alibi_dv2",
    "alibi_dv2_h",
    "alibi_dv2_cb",
    "alibi_dv2_cb_s_l",
    "alibi_dv2_cb_nr_l",
    "alibi_dv2_cb_l_j",
    "alibi_dv2_coco",
    "alibi_dv2_coco_e1",
    "alibi_dv3",
    "nope",
    "dv",
    "dv_b",
    "dv3",
    "dv3_b",
    "vit_b",
    "clip_b",
    "eva02_b",
    "sam_b",
    "convnext",
    "vit_b_in",
    "deit",
    "vit_t_in",
    "classical",
    "eupe_s",
]
TimmModels = Literal[
    "dv2",
    "dv2_b",
    "dv2_cb",
    "dv2_db",
    "dv",
    "dv_b",
    "dv3",
    "dv3_b",
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
    "dv2_db": "DINOv2(DB)",
    "dvt": "DVT",
    "sinusoid_dv2": "Sinusoid",
    "alibi_dv2": "ALiBi-Dv2",
    "alibi_dv2_h": "ALiBi(H)-Dv2",
    "alibi_dv2_cb": "ALiBi(CB)-Dv2",
    "alibi_dv2_cb_s_l": "ALiBi(CB-S-L)-Dv2",
    "alibi_dv2_cb_nr_l": "ALiBi(CB-NR-L)-Dv2",
    "alibi_dv2_cb_l_j": "ALiBi(CB-L-J)-Dv2",
    "alibi_dv2_coco": "ALiBi(COCO)-Dv2",
    "alibi_dv2_coco_e1": "ALiBi(COCO)-Dv2-e1",
    "alibi_dv3": "ALiBi-Dv3",
    "nope": "NoPE",
    "dv": "DINO",
    "dv_b": "DINO-B",
    "dv3": "DINOv3",
    "dv3_b": "DINOv3-B",
    "clip_b": "CLIP-B",
    "eva02_b": "EVA02-B",
    "sam_b": "SAM-B",
    "convnext": "",
    "vit_b": "ViT-B",
    "vit_b_in": "ViT-B-INet",
    "deit": "",
    "vit_t_in": "",
    "classical": "Classical",
}
model_chkpoints: dict[ModelTypes, str] = {
    "dv2": "",
    "dv2_b": "",
    "dvt": "dvt.pth",
    "sinusoid_dv2": "e2.pth",
    "alibi_dv2": "alibi_dv2_vits14_reg.pth",
    "alibi_dv2_h": "alibi_homog_dv2_vits14_reg.pth",
    "alibi_dv2_cb": "alibi_cb_dv2_vits14_reg.pth",
    "alibi_dv2_cb_s_l": "alibi_scratch_cb_dv2_vits14_reg_learned_m.pth",
    "alibi_dv2_cb_nr_l": "alibi_cb_dv2_vits14_noreg.pth",
    "alibi_dv2_cb_l_j": "alibi_cb_dv2_vits14_j_ms.pth",
    "alibi_dv2_coco": "alibi_coco_dv2_vits14_reg_ms.pth",
    "alibi_dv2_coco_e1": "alibi_coco_dv2_vits14_reg_ms_e1.pth",
    "alibi_dv3": "alibi_dv3_ms.pth",
    "nope": "nope_coco_dv2_vits14_reg_ms.pth",
    "dv": "",
    "dv_b": "",
    "dv3": "dinov3_vits_patch16_plus_reg4.pth",
    # "dv3": "",
    "dv3_b": "dinov3_vitb_patch16_reg4.pth",
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
    "dv3_b": MODEL_LIST[15],
    "vit_b": MODEL_LIST[6],
    "clip_b": MODEL_LIST[7],
    "eva02_b": MODEL_LIST[8],
    "sam_b": MODEL_LIST[9],
    "convnext": MODEL_LIST[10],
    "vit_b_in": MODEL_LIST[11],
    "deit": MODEL_LIST[12],
    "vit_t_in": MODEL_LIST[13],
    "dv2_cb": MODEL_LIST[1],
    "dv2_db": MODEL_LIST[1],
}


def _get_stride(model_type: ModelTypes, model_id: str) -> int:
    if "patch16" in model_id.lower():
        return 16
    if "dv3" in model_type.lower():
        return 16
    else:
        return 14


model_type_to_registry_name = {
    "dv2": "dinov2_s",
    "dv2_b": "dinov2_b",
    "dv": "dino_s",
    "dv_b": "dino_b",
    "dv3": "dinov3_s",
    "dv3_b": "dinov3_b",
    "vit_b": "mae_b",
    "clip_b": "clip_b",
    "eva02_b": "eva02_b",
    "sam_b": "sam_b",
    "convnext": "convnext",
    "vit_b_in": "in1k_b",
    "deit": "deit_s",
    "vit_t_in": "vit_t_in",
    "dv2_cb": "dinov2_s",
}


def get_model(
    model_type: ModelTypes,
    model_dir: str,
    device: str,
    to_half: bool = False,
    conf_path: str | None = None,
    remove_final_norm: bool = False,
) -> PretrainedViTWrapper:
    if model_type == "classical":
        return None

    registry_name = model_type_to_registry_name.get(model_type, model_type)
    chk_name = model_chkpoints.get(model_type, "")
    checkpoint_path = f"{model_dir}/{chk_name}" if chk_name else None

    build_kwargs = {}
    is_local = "dv3" in model_type or "dinov3" in registry_name
    if is_local:
        base_chk_name = model_chkpoints.get("dv3" if "dv3" in model_type else model_type, "")
        build_kwargs["checkpoint_path"] = f"{model_dir}/{base_chk_name}" if base_chk_name else None
        build_kwargs["model_conf_path"] = conf_path if conf_path else "dinov3"

    if model_type == "dvt":
        build_kwargs["denoiser_path"] = checkpoint_path
        checkpoint_path = None

    model = WrapperRegistry.build(registry_name, device=device, **build_kwargs)

    # Load checkpoint weights if they weren't loaded during backbone creation
    if checkpoint_path and (not is_local or checkpoint_path != build_kwargs.get("checkpoint_path")):
        weights = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(weights)

    if remove_final_norm:
        if hasattr(model.vit, "norm"):
            model.vit.norm = nn.Identity()

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
