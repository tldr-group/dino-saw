from typing import Literal, get_args
from PVW import PretrainedViTWrapper, WrapperRegistry, WrapperConfig, BackboneConfig

from dinosaw.wrappers.alibi import replace_pe_with_sincos, register_alibi_model
from dinosaw.wrappers.simple_debias import DebiasedViTWrapper, ChannelBlankedWrapper, TransformAverageWrapper
from dinosaw.wrappers.denoising_vits import DenoisingViTWrapper

# 1. Type definitions
ModelTypes = Literal[
    # New / standard notation
    "dinov2_s",
    "dinov3_s",
    "dinov3_s+",
    "nope_dinov2_s",
    "sinusoid_dinov2_s",
    "sinusoid_dinov2_s_cb",
    "cb_dinov2_s",
    "tr_dinov2_s",
    "debiased_dinov2_s",
    "dvt_dinov2_s",
    "alibi_dinov2_s",
    "alibi_dinov3_s+",
    # Others
    "dinov2_b",
    "dino_s",
    "dino_b",
    "dinov3_b",
    "eupe_s",
    "mae_b",
    "clip_b",
    "eva02_b",
    "sam_b",
    "convnext",
    "in1k_b",
    "deit_s",
    "vit_t_in",
    "classical",
    # Commented out variants
    "alibi_dinov2_s_no_norm_no_wrap_no_cb",
    "alibi_dinov2_s_no_norm_no_wrap_cb",
    "alibi_dinov2_s_no_norm_no_wrap",
    "alibi_dinov2_s_no_norm_wrap",
    "alibi_dinov2_s_no_norm_wrap_ms",
    "alibi_dinov3_s+_no_norm_wrap",
    "alibi_dinov3_s+_no_norm_wrap_ms",
    "alibi_dinov3_s+_norm_wrap_ms",
]

MODEL_LIST: tuple[ModelTypes, ...] = get_args(ModelTypes)


# Checkpoint file mappings
WRAPPER_CHECKPOINTS: dict[ModelTypes, str] = {
    # Base DinoV3 Backbones
    "dinov3_s+": "backbones/dinov3_vits_patch16_plus_reg4.pth",
    "dinov3_b": "backbones/dinov3_vitb_patch16_reg4.pth",
    "dvt_dinov2_s": "backbones/dvt.pth",
    "eupe_s": "backbones/EUPE-ViT-S.pt",
    # Wrapper checkpoints
    # "alibi_dinov2_s": "trained/alibi_dv2_vits14_reg.pth",
    "alibi_dinov2_s": "trained/alibi_coco_dv2_vits14_reg_ms.pth",
    "alibi_dinov3_s+": "trained/alibi_dv3_ms.pth",
    "nope_dinov2_s": "trained/nope_coco_dv2_vits14_reg_ms.pth",
    "sinusoid_dinov2_s": "e2.pth",
    "sinusoid_dinov2_s_cb": "ablations/sinusoid_dv2_cb.pth",
    # Ablations
    "alibi_dinov2_s_no_norm_no_wrap_no_cb": "ablations/alibi_dv2_no_norm_no_wrap_nocb.pth",
    "alibi_dinov2_s_no_norm_no_wrap_cb": "ablations/alibi_dv2_no_norm_no_wrap_cb.pth",
    "alibi_dinov2_s_no_norm_no_wrap": "ablations/alibi_dv2_no_norm_no_wrap_cb.pth",
    "alibi_dinov2_s_no_norm_wrap": "ablations/alibi_dv2_no_norm_wrap_more_cb.pth",
    "alibi_dinov2_s_no_norm_wrap_ms": "ablations/alibi_dv2_no_norm_wrap_more_cb_ms.pth",
    "alibi_dinov3_s+_no_norm_wrap": "ablations/alibi_dv3_plus_no_norm_wrap_cb.pth",
    "alibi_dinov3_s+_no_norm_wrap_ms": "ablations/alibi_dv3_plus_no_norm_wrap_cb_ms.pth",
    "alibi_dinov3_s+_norm_wrap_ms": "ablations/alibi_dv3_plus_norm_wrap_cb_ms.pth",
}

# 2. Name lookup mapping
MODEL_NAMES: dict[ModelTypes, str] = {
    # New / standard notation
    "dinov2_s": "DINOv2",
    "dinov3_s+": "DINOv3",
    "alibi_dinov2_s": "ALiBi-Dv2",
    "alibi_dinov3_s+": "ALiBi-Dv3",
    "nope_dinov2_s": "NoPE",
    "sinusoid_dinov2_s": "Sinusoid",
    "sinusoid_dinov2_s_cb": "Sinusoid(CB)",
    "debiased_dinov2_s": "DINOv2(DB)",
    "dvt_dinov2_s": "DVT",
    "cb_dinov2_s": "DINOv2(CB)",
    # Others
    "dinov2_b": "DINOv2-B",
    "dino_s": "DINO",
    "dino_b": "DINO-B",
    "dinov3_b": "DINOv3-B",
    "mae_b": "ViT-B",
    "clip_b": "CLIP-B",
    "eva02_b": "EVA02-B",
    "sam_b": "SAM-B",
    "convnext": "",
    "in1k_b": "ViT-B-INet",
    "deit_s": "",
    "vit_t_in": "",
    "classical": "Classical",
    # Ablations
    "alibi_dinov2_s_no_norm_no_wrap_no_cb": "ALiBi-Dv2(-norm,-wrap,-CB)",
    "alibi_dinov2_s_no_norm_no_wrap_cb": "ALiBi-Dv2(-norm,-wrap,CB)",
    "alibi_dinov2_s_no_norm_no_wrap": "ALiBi-Dv2(-norm,-wrap)",
    "alibi_dinov2_s_no_norm_wrap": "ALiBi-Dv2(-norm)",
    "alibi_dinov2_s_no_norm_wrap_ms": "ALiBi-Dv2(-norm,+ms)",
    "alibi_dinov3_s+_no_norm_wrap": "ALiBi-Dv3(-norm)",
    "alibi_dinov3_s+_no_norm_wrap_ms": "ALiBi-Dv3(-norm,+ms)",
    "alibi_dinov3_s+_norm_wrap_ms": "ALiBi-Dv3(+norm,+ms)",
}


def get_model(
    model_type: ModelTypes,
    checkpoint_path: str | None,
    device: str,
    eval: bool = True,
    conf_path: str | None = "models",
    **build_kwargs,
) -> PretrainedViTWrapper:
    build_kwargs["checkpoint_path"] = checkpoint_path
    build_kwargs["model_conf_path"] = conf_path
    if model_type == "dvt_dinov2_s":
        build_kwargs["denoiser_path"] = checkpoint_path
        build_kwargs["checkpoint_path"] = None
    is_dinov3 = model_type in ("dinov3_s+", "dinov3_b", "alibi_dinov3_s+")
    if is_dinov3:
        build_kwargs["backbone_type"] = "torch_hub"
        build_kwargs["pretrained"] = False
    # is_alibi = "alibi" in model_type.lower()
    # if is_alibi:
    #     build_kwargs["pretrained"] = False

    wrapper = WrapperRegistry.build(model_type, device=device, **build_kwargs)
    if eval:
        wrapper.eval()
    return wrapper


def get_models(
    model_types: tuple[ModelTypes, ...],
    device: str,
    eval: bool = True,
    checkpoint_dir: str = "models/checkpoints",
    conf_dir: str = "models",
    **build_kwargs,
) -> dict[ModelTypes, PretrainedViTWrapper]:
    models = {}
    for model_type in model_types:
        rel_path = WRAPPER_CHECKPOINTS.get(model_type, None)
        checkpoint_path = f"{checkpoint_dir}/{rel_path}" if rel_path else None
        models[model_type] = get_model(
            model_type, checkpoint_path, device, eval=eval, conf_path=conf_dir, **build_kwargs
        )
    return models


# 3. Base Models
# Register dinov2_s explicitly with timm backend
WrapperRegistry.register(
    "dinov2_s",
    WrapperConfig(backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s")),
)

# Register dinov3_s mapping to dinov3_s+ (with register and SwiGLU)
WrapperRegistry.register(
    "dinov3_s+",
    WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="torch_hub",
            model_arch="dinov3_s+",
            pretrained=False,
            model_conf_path="dinov3",
        )
    ),
)

# 4. NoPE
WrapperRegistry.register(
    "nope_dinov2_s",
    WrapperConfig(backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s", remove_pos_embed=True)),
)

# 5. Sinusoid
WrapperRegistry.register(
    "sinusoid_dinov2_s",
    WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="dinov2_s",
            modifications=[replace_pe_with_sincos],
        )
    ),
)

WrapperRegistry.register(
    "sinusoid_dinov2_s_cb",
    WrapperConfig(
        backbone_cfg=BackboneConfig(
            backbone_type="timm",
            model_arch="dinov2_s",
            modifications=[replace_pe_with_sincos],
        )
    ),
)

# 6. Debiased
WrapperRegistry.register(
    "debiased_dinov2_s",
    WrapperConfig(
        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s"),
        wrapper_class=DebiasedViTWrapper,
    ),
)

# 7. Denoising ViT
WrapperRegistry.register(
    "dvt_dinov2_s",
    WrapperConfig(
        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s"),
        wrapper_class=DenoisingViTWrapper,
    ),
)

WrapperRegistry.register(
    "cb_dinov2_s",
    WrapperConfig(
        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s"),
        wrapper_class=ChannelBlankedWrapper,
        wrapper_kwargs={"channels_to_blank": [47, 117, 359]},
    ),
)

WrapperRegistry.register(
    "tr_dinov2_s",
    WrapperConfig(
        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s"),
        wrapper_class=TransformAverageWrapper,
    ),
)

# 8. ALiBi models
register_alibi_model(
    "alibi_dinov2_s",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
)

register_alibi_model(
    "alibi_dinov2_s_no_wrap",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=False,
)
register_alibi_model(
    "alibi_dinov2_s_no_norm",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=True,
    normalize=False,
)

register_alibi_model(
    "alibi_dinov2_s_no_norm_no_wrap_no_cb",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=True,
    normalize=False,
)

register_alibi_model(
    "alibi_dinov2_s_no_norm_no_wrap_cb",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=True,
    normalize=False,
)

register_alibi_model(
    "alibi_dinov2_s_no_norm_no_wrap",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=False,
    normalize=False,
)

register_alibi_model(
    "alibi_dinov2_s_no_norm_wrap",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=True,
    normalize=False,
)

register_alibi_model(
    "alibi_dinov2_s_no_norm_wrap_ms",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    wrap=True,
    normalize=False,
)


register_alibi_model(
    "alibi_dinov3_s+",
    "dinov3_s+",
    backbone_type="torch_hub",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
)

register_alibi_model(
    "alibi_dinov3_s+_no_norm_wrap",
    "dinov3_s+",
    backbone_type="torch_hub",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    normalize=False,
    wrap=True,
)

register_alibi_model(
    "alibi_dinov3_s+_no_norm_wrap_ms",
    "dinov3_s+",
    backbone_type="torch_hub",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
    normalize=False,
    wrap=True,
)

register_alibi_model(
    "alibi_dinov3_s+_norm_wrap_ms",
    "dinov3_s+",
    backbone_type="torch_hub",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
)

# Skip for now / commented-out variants:
# register_alibi_model(
#     "alibi_dinov2_s_h",
#     "dinov2_s",
#     slope_type="constant",
#     n_reg_tokens=4,
#     add_cls=True,
# )

# register_alibi_model(
#     "alibi_dinov2_s_cb",
#     "dinov2_s",
#     slope_type="constant",
#     n_reg_tokens=4,
#     add_cls=True,
# )

# register_alibi_model(
#     "alibi_dinov2_s_cb_s_l",
#     "dinov2_s",
#     slope_type="learned",
#     n_reg_tokens=4,
#     add_cls=True,
# )

# register_alibi_model(
#     "alibi_dinov2_s_cb_nr_l",
#     "dinov2_s",
#     slope_type="learned",
#     n_reg_tokens=0,
#     add_cls=False,
# )

# register_alibi_model(
#     "alibi_dinov2_s_cb_l_j",
#     "dinov2_s",
#     slope_type="learned",
#     n_reg_tokens=4,
#     add_cls=True,
#     jitter_mag=0.025,
# )

# register_alibi_model(
#     "alibi_dinov2_s_coco",
#     "dinov2_s",
#     slope_type="constant",
#     n_reg_tokens=4,
#     add_cls=True,
# )

# register_alibi_model(
#     "alibi_dinov2_s_coco_e1",
#     "dinov2_s",
#     slope_type="constant",
#     n_reg_tokens=4,
#     add_cls=True,
# )
