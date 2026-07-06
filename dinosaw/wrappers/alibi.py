import torch
import torch.nn as nn
from functools import partial

from dinosaw.alibi_logic import (
    DistanceMatrixWrapper,
    AlibiAttention,
    convert_dv3_model,
    build_2d_sincos_pos_embed,
)
from PVW import WrapperRegistry, WrapperConfig, BackboneConfig
from PVW.types import satisfies_protocol, Dv3ViT


def convert_timm_model(
    model: nn.Module,
    slope_type: str = "constant",
    jitter_mag: float = 0.0,
    distance_matrix: DistanceMatrixWrapper | None = None,
) -> nn.Module:
    """Replace attention layers in a timm VisionTransformer with AlibiAttention."""
    for blk in model.blocks:
        old_attn = blk.attn
        new_attn = AlibiAttention(
            distance_matrix=distance_matrix,
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=(old_attn.qkv.bias is not None),
            qk_norm=getattr(old_attn, "qk_norm", False),
            proj_bias=(old_attn.proj.bias is not None),
            attn_drop=getattr(old_attn.attn_drop, "p", 0.0),
            proj_drop=getattr(old_attn.proj_drop, "p", 0.0),
            slope_type=slope_type,
            jitter_mag=jitter_mag,
        )

        # Copy state dict from old attention layer (weights, biases)
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        new_attn.to(device=old_attn.qkv.weight.device, dtype=old_attn.qkv.weight.dtype)
        blk.attn = new_attn

    return model


def add_alibi(
    vit: nn.Module,
    slope_type: str = "constant",
    n_reg_tokens: int = 4,
    metric: str = "euclidean",
    normalize: bool = True,
    wrap: bool = True,
    add_cls: bool = True,
    jitter_mag: float = 0.0,
) -> nn.Module:
    """Backbone transformation to add ALiBi distance bias to attention blocks.
    Registers a pre-forward hook to update the distance matrix dynamically based on input shape.
    """
    dm = DistanceMatrixWrapper(
        n_tokens_h=16,
        n_tokens_w=16,
        n_reg_tokens=n_reg_tokens,
        metric=metric,
        normalize=normalize,
        wrap=wrap,
        add_cls=add_cls,
    )

    param = next(vit.parameters(), None)
    if param is not None:
        dm.to(device=param.device, dtype=param.dtype)

    if hasattr(vit, "rope_embed") or satisfies_protocol(vit, Dv3ViT):
        vit = convert_dv3_model(
            vit,
            slope_type=slope_type,
            jitter_mag=jitter_mag,
            distance_matrix=dm,
        )
    else:
        vit = convert_timm_model(
            vit,
            slope_type=slope_type,
            jitter_mag=jitter_mag,
            distance_matrix=dm,
        )

    vit.distance_matrix = dm

    def update_alibi_hook(module, args):
        if len(args) == 0:
            return
        x = args[0]
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            b, _, h, w = x.shape
            p = module.patch_size
            s = module.proj.stride
            n_patch_h = (h - p[0]) // s[0] + 1
            n_patch_w = (w - p[1]) // s[1] + 1
            dm.update(n_patch_h, n_patch_w)

    vit.patch_embed.register_forward_pre_hook(update_alibi_hook)
    return vit


def replace_pe_with_sincos(vit: nn.Module) -> nn.Module:
    """Backbone transformation to replace the absolute position embedding with 2D sincos positional embedding."""
    old_pos_embed = vit.pos_embed
    if old_pos_embed is not None:
        _, _, embed_dim = old_pos_embed.shape
        stride = vit.patch_embed.proj.stride[0]
        H, W = 224, 224
        new_pos_embed = build_2d_sincos_pos_embed(
            H // stride, W // stride, embed_dim, torch.float32, old_pos_embed.device
        )
        new_pos_embed *= old_pos_embed.std() / new_pos_embed.std()

        if hasattr(vit, "set_input_size"):
            vit.set_input_size((224, 224), (14, 14))

        vit.pos_embed = nn.Parameter(new_pos_embed, requires_grad=False)
    return vit


# Helper to register alibi models
def register_alibi_model(
    name: str,
    model_arch: str,
    backbone_type: str = "timm",
    slope_type: str = "constant",
    n_reg_tokens: int = 4,
    add_cls: bool = True,
    jitter_mag: float = 0.0,
):
    mod = partial(
        add_alibi,
        slope_type=slope_type,
        n_reg_tokens=n_reg_tokens,
        metric="euclidean",
        normalize=True,
        wrap=True,
        add_cls=add_cls,
        jitter_mag=jitter_mag,
    )
    WrapperRegistry.register(
        name,
        WrapperConfig(
            backbone_cfg=BackboneConfig(
                backbone_type=backbone_type,
                model_arch=model_arch,  # type: ignore
                pretrained=False,
                remove_pos_embed=True,
                modifications=[mod],
            )
        ),
    )
