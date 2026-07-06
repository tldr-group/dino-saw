from typing import Literal
from PVW import WrapperRegistry, WrapperConfig, BackboneConfig

from dinosaw.wrappers.alibi import replace_pe_with_sincos, register_alibi_model
from dinosaw.wrappers.simple_debias import DebiasedViTWrapper
from dinosaw.wrappers.denoising_vits import DenoisingViTWrapper

# 1. Type definitions
ModelTypes = Literal[
    # New / standard notation
    "dinov2_s",
    "dinov3_s",
    "dinov3_s+",
    "nope_dinov2_s",
    "sinusoid_dinov2_s",
    "debiased_dinov2_s",
    "dvt_dinov2_s",
    "alibi_dinov2_s",
    "alibi_dinov3_s+",
    # Old notation
    "dv2",
    "dv3",
    "sinusoid_dv2",
    "dv2_db",
    "alibi_dv2",
    "alibi_dv3",
    # Others
    "dinov2_b",
    "dino_s",
    "dino_b",
    "dinov3_b",
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
    "alibi_dinov2_s_h",
    "alibi_dinov2_s_cb",
    "alibi_dinov2_s_cb_s_l",
    "alibi_dinov2_s_cb_nr_l",
    "alibi_dinov2_s_cb_l_j",
    "alibi_dinov2_s_coco",
    "alibi_dinov2_s_coco_e1",
]

# 2. Name lookup mapping
MODEL_NAMES = {
    # New / standard notation
    "dinov2_s": "DINOv2",
    "dinov3_s+": "DINOv3",
    "alibi_dinov2_s": "ALiBi-Dv2",
    "alibi_dinov3_s": "ALiBi-Dv3",
    "nope": "NoPE",
    "sinusoid_dinov2_s": "Sinusoid",
    "debiased_dinov2_s": "DINOv2(DB)",
    "dvt": "DVT",
    # Old notation
    "dv2": "DINOv2",
    "dv2_b": "DINOv2-B",
    "dv": "DINO",
    "dv_b": "DINO-B",
    "dv3": "DINOv3",
    "dv3_b": "DINOv3-B",
    "alibi_dv2": "ALiBi-Dv2",
    "alibi_dv3": "ALiBi-Dv3",
    "sinusoid_dv2": "Sinusoid",
    "dv2_db": "DINOv2(DB)",
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
    # Commented out variants
    "alibi_dinov2_s_h": "ALiBi(H)-Dv2",
    "alibi_dinov2_s_cb": "ALiBi(CB)-Dv2",
    "alibi_dinov2_s_cb_s_l": "ALiBi(CB-S-L)-Dv2",
    "alibi_dinov2_s_cb_nr_l": "ALiBi(CB-NR-L)-Dv2",
    "alibi_dinov2_s_cb_l_j": "ALiBi(CB-L-J)-Dv2",
    "alibi_dinov2_s_coco": "ALiBi(COCO)-Dv2",
    "alibi_dinov2_s_coco_e1": "ALiBi(COCO)-Dv2-e1",
}

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
    "nope",
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

# 8. ALiBi models
register_alibi_model(
    "alibi_dinov2_s",
    "dinov2_s",
    slope_type="constant",
    n_reg_tokens=4,
    add_cls=True,
)

register_alibi_model(
    "alibi_dinov3_s+",
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

# Central list of active registered models
MODEL_LIST = [
    "dinov2_s",
    "dinov3_s",
    "nope",
    "sinusoid_dinov2_s",
    "debiased_dinov2_s",
    "dvt",
    "alibi_dinov2_s",
    "alibi_dinov3_s",
]

# Checkpoint file mappings
WRAPPER_CHECKPOINTS = {
    # Base DinoV3 Backbones
    "dinov3_s": "dinov3_vits_patch16_plus_reg4.pth",
    "dinov3_b": "dinov3_vitb_patch16_reg4.pth",
    # Wrapper checkpoints
    "alibi_dinov2_s": "alibi_dv2_vits14_reg.pth",
    "alibi_dinov3_s": "alibi_dv3_ms.pth",
    "nope": "nope_coco_dv2_vits14_reg_ms.pth",
    "dvt": "dvt.pth",
    "sinusoid_dinov2_s": "e2.pth",
    # Commented out checkpoint mappings:
    # "alibi_dinov2_s_h": "alibi_homog_dv2_vits14_reg.pth",
    # "alibi_dinov2_s_cb": "alibi_cb_dv2_vits14_reg.pth",
    # "alibi_dinov2_s_cb_s_l": "alibi_scratch_cb_dv2_vits14_reg_learned_m.pth",
    # "alibi_dinov2_s_cb_nr_l": "alibi_cb_dv2_vits14_noreg.pth",
    # "alibi_dinov2_s_cb_l_j": "alibi_cb_dv2_vits14_j_ms.pth",
    # "alibi_dinov2_s_coco": "alibi_coco_dv2_vits14_reg_ms.pth",
    # "alibi_dinov2_s_coco_e1": "alibi_coco_dv2_vits14_reg_ms_e1.pth",
}
