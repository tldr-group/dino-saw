import torch
import numpy as np
from PIL import Image

from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper
from dinosaw.comaprisons.denoising_vits import DenoisingViTWrapper
from dinosaw.utils import to_numpy, closest_resize, convert_image

from typing import Literal, get_args

import matplotlib.pyplot as plt
from matplotlib import font_manager


ModelTypes = Literal["dv2", "dv2_b", "dvt", "alibi_dv2", "alibi_dv2_h", "alibi_dv2_cb", "nope"]
model_types: tuple[ModelTypes] = get_args(ModelTypes)
model_names: dict[ModelTypes, str] = {
    "dv2": "DINOv2",
    "dv2_b": "DINOv2-B",
    "dvt": "DVT",
    "alibi_dv2": "ALiBi-Dv2",
    "alibi_dv2_h": "ALiBi(H)-Dv2",
    "alibi_dv2_cb": "ALiBi(CB)-Dv2",
    "nope": "NoPE",
}
model_chkpoints: dict[ModelTypes, str] = {
    "dv2": "",
    "dv2_b": "",
    "dvt": "dvt.pth",
    "alibi_dv2": "alibi_dv2_vits14_reg.pth",
    "alibi_dv2_h": "alibi_homog_dv2_vits14_reg.pth",
    "alibi_dv2_cb": "alibi_cb_dv2_vits14_reg.pth",
    "nope": "nope_dv2_vits14_reg.pth",
}


def get_model(model_type: ModelTypes, model_dir: str, device: str, to_half: bool = False) -> PretrainedViTWrapper:
    S = 14
    model: PretrainedViTWrapper | None = None

    if model_type in ("dv2", "dv2_b"):
        model_id = MODEL_LIST[1] if model_type == "dv2" else MODEL_LIST[3]
        model = PretrainedViTWrapper(
            model_id,
            stride=S,
            add_flash_attn=False,
            device=device,
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
        model = AlibiVitWrapper(
            MODEL_LIST[1],
            stride=S,
            add_flash_attn=False,
            device=device,
            slope_type="constant",
            normalize=True,
            wrap=True,
        )
        model.load_state_dict(weights)
    else:
        raise Exception("Invalid model type")

    assert model is not None
    model = model.eval()
    if to_half:
        model = model.half()

    return model


def get_models(
    model_types: tuple[ModelTypes, ...], model_dir: str, device: str, to_half: bool = False
) -> dict[ModelTypes, PretrainedViTWrapper]:
    return {k: get_model(k, model_dir, device, to_half) for k in model_types}


def get_features(
    model: PretrainedViTWrapper,
    pil_img: Image.Image,
    channel_last: bool = False,
    channel_blank: bool = False,
    to_half: bool = False,
    device: str = "cuda:0",
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
