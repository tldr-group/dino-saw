import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

from dinosaw.wrappers import PretrainedViTWrapper, MODEL_LIST, WRAPPER_CHECKPOINTS, ModelTypes, MODEL_NAMES
from PVW import WrapperRegistry
from dinosaw.utils import to_numpy, closest_resize, convert_image
from typing import cast
import matplotlib.pyplot as plt
from matplotlib import font_manager
model_names = MODEL_NAMES
model_chkpoints: dict[str, str] = WRAPPER_CHECKPOINTS



def _get_stride(model_type: ModelTypes, model_id: str) -> int:
    if "patch16" in model_id.lower():
        return 16
    if "dv3" in model_type.lower():
        return 16
    else:
        return 14


model_type_to_registry_name = {
    # Old notation mappings
    "dv2": "dinov2_s",
    "dv3": "dinov3_s",
    "alibi_dv2": "alibi_dinov2_s",
    "alibi_dv3": "alibi_dinov3_s",
    "sinusoid_dv2": "sinusoid_dinov2_s",
    "dv2_db": "debiased_dinov2_s",
    # New / standard notation mappings
    "dinov2_s": "dinov2_s",
    "dinov3_s": "dinov3_s",
    "alibi_dinov2_s": "alibi_dinov2_s",
    "alibi_dinov3_s": "alibi_dinov3_s",
    "sinusoid_dinov2_s": "sinusoid_dinov2_s",
    "debiased_dinov2_s": "debiased_dinov2_s",
    "nope": "nope",
    "dvt": "dvt",
    # Others
    "dv2_b": "dinov2_b",
    "dv": "dino_s",
    "dv_b": "dino_b",
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


def _find_chk_path(model_dir: str, chk_name: str) -> str | None:
    if not chk_name:
        return None
    path = f"{model_dir}/{chk_name}"
    if os.path.exists(path):
        return path
    for fallback in [
        f"models/{chk_name}",
        f"models/checkpoints/{chk_name}",
        f"models/checkpoints/backbones/{chk_name}",
        f"models/checkpoints/trained/{chk_name}",
        f"trained_models/{chk_name}",
    ]:
        if os.path.exists(fallback):
            return fallback
    return path


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
    chk_name = model_chkpoints.get(registry_name, "")
    checkpoint_path = _find_chk_path(model_dir, chk_name)

    build_kwargs = {}
    is_local = "dv3" in model_type or "dinov3" in registry_name
    if is_local:
        base_chk_key = "dinov3_b" if "dinov3_b" in registry_name or "dv3_b" in model_type else "dinov3_s"
        base_chk_name = model_chkpoints.get(base_chk_key, "")
        build_kwargs["checkpoint_path"] = _find_chk_path(model_dir, base_chk_name)
        build_kwargs["model_conf_path"] = conf_path if conf_path else "dinov3"

    if model_type == "dvt":
        build_kwargs["denoiser_path"] = checkpoint_path
        checkpoint_path = None

    model = WrapperRegistry.build(registry_name, device=device, **build_kwargs)

    # Load checkpoint weights if they weren't loaded during backbone creation
    if checkpoint_path and (not is_local or checkpoint_path != build_kwargs.get("checkpoint_path")):
        weights = torch.load(checkpoint_path, weights_only=True, map_location=device)
        weights = {k.replace("model.", "vit.", 1) if k.startswith("model.") else k: v for k, v in weights.items()}
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
