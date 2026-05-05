"""
A simple wrapper around timm vision transformers that allow for adjusting the model stride
and has a helper to get patch features out.

Adpated from https://github.com/Jiawei-Yang/Denoising-ViT/blob/main/dvt/models/vit_wrapper.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from timm import create_model
from timm.data.transforms_factory import create_transform
from timm.data.config import resolve_data_config
from timm.layers.mlp import Mlp
from timm.layers.attention import Attention
from timm.models.vision_transformer import VisionTransformer, Block


from typing import Type, Literal, Optional


import re
from typing import cast
from types import MethodType


MODEL_LIST = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_small_patch16_224.dino",
    "vit_base_patch16_224.mae",
    "vit_base_patch16_clip_224.openai",
    "eva02_base_patch14_224.mim_in22k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "deit3_small_patch16_224.fb_in1k",
    "tiny_vit_5m_224.dist_in22k_ft_in1k",
    "vit_base_patch16_224.dino",
]
DEFAULT_MODEL = MODEL_LIST[1]


class PretrainedViTWrapper(nn.Module):
    def __init__(
        self,
        model_identifier: str = DEFAULT_MODEL,
        stride: int = 14,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: str = "cpu",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert model_identifier in MODEL_LIST, f"Model type {model_identifier} not tested yet."
        self.model_identifier = model_identifier

        self.stride = stride
        patch_size_from_id = re.search(r"patch(\d+)", model_identifier)
        self.patch_size = 14 if patch_size_from_id is None else int(patch_size_from_id.group(1))

        n_reg_tokens_from_id = re.search(r"reg(\d+)", model_identifier)
        self.n_reg_tokens = 0 if n_reg_tokens_from_id is None else int(n_reg_tokens_from_id.group(1))
        self.n_cls_tokens = 1

        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.pretrained = pretrained
        self.model, self.transformation = self.create_model(model_identifier, device, **kwargs)

        # overwrite the stride size
        if stride != self.model.patch_embed.proj.stride[0]:
            self.model.patch_embed.proj.stride = (stride, stride)

            def dynamic_feat_size(self, img_size: tuple[int, int]) -> tuple[int, int]:
                """Get grid (feature) size for given image size taking account of dynamic padding.
                NOTE: must be torchscript compatible so using fixed tuple indexing
                """
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = MethodType(dynamic_feat_size, self.model.patch_embed)

        self.to(device)

    @property
    def n_output_dims(self) -> int:
        assert self.model.pos_embed
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)  # type: ignore

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self, model_identifier: str, device: str, **kwargs
    ) -> tuple[VisionTransformer, transforms.Compose]:

        model = create_model(
            model_identifier,
            pretrained=self.pretrained,  # True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            pretrained_strict=False,
            **kwargs,
        )

        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        # data_config = resolve_model_data_config(model=model)
        data_config = resolve_data_config(model=model)

        model = cast(VisionTransformer, model)
        img_transforms = cast(transforms.Compose, create_transform(**data_config, is_training=False))
        return model, img_transforms

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | list[int] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            intermediates_only=True,
        )

    def forward(self, x: torch.Tensor, attn_mask=None):
        return self.model(x, attn_mask)

    def forward_features(
        self,
        x: torch.Tensor,
        make_2D: bool = False,
        add_reg: bool = False,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p) // s + 1, (w - p) // s + 1

        if add_reg:
            feats = self.model.forward_features(x)
        else:  # ignore CLS + reg tokens
            feats = self.model.forward_features(x)[:, self.n_reg_tokens + self.n_cls_tokens :]

        feats = feats.permute((0, 2, 1))

        if make_2D and not add_reg:
            feats = feats.reshape((b, -1, n_patch_h, n_patch_w))
        return feats


def get_distance_matrix(
    n_tokens_h: int,
    n_tokens_w: int,
    n_reg_tokens: int = 4,
    metric: str = "euclidean",
    normalize: bool = True,
    wrap: bool = False,
    add_cls: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(n_tokens_h, device=device),
            torch.arange(n_tokens_w, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)  # (N, 2)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    diff = diff.abs()

    if wrap:
        diff[..., 0] = torch.minimum(diff[..., 0], n_tokens_h - diff[..., 0])
        diff[..., 1] = torch.minimum(diff[..., 1], n_tokens_w - diff[..., 1])

    if metric == "euclidean":
        D = torch.sqrt((diff**2).sum(-1))
    elif metric == "manhattan":
        D = diff.sum(-1)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    if normalize:
        D /= D.max()
    D *= -1

    n_extra_tokens = int(add_cls) + n_reg_tokens
    # timm prepends [CLS] + [REG]
    D = F.pad(D, (n_extra_tokens, 0, n_extra_tokens, 0), mode="constant")

    return D.to(device=device, dtype=dtype)


AlibiSlopeType = Literal["fixed", "learned", "constant"]


def get_alibi_slope(
    num_heads: int, slope_type: AlibiSlopeType = "constant", device: str = "cpu"
) -> torch.Tensor | nn.Parameter:
    m: torch.Tensor | nn.Parameter
    match slope_type:
        case "fixed":
            xs = (2**8) ** (1 / num_heads)
            m = torch.tensor([1 / xs ** (i + 1) for i in range(num_heads)], device=device)
        case "learned":
            m = torch.rand(num_heads, device=device)
        case "constant":
            m = torch.ones(num_heads, device=device)
        case _:
            raise ValueError(f"Unknown slope type {type}")

    m = m.unsqueeze(-1).unsqueeze(-1)
    if slope_type == "learned":
        m = nn.Parameter(m)
    else:
        m.requires_grad = False
    return m


class DistanceMatrixWrapper(nn.Module):
    def __init__(
        self,
        n_tokens_h: int,
        n_tokens_w: int,
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
    ) -> None:
        super().__init__()

        self.n_tokens_h = n_tokens_h
        self.n_tokens_w = n_tokens_w
        self.n_reg_tokens = n_reg_tokens
        self.metric = metric
        self.normalize = normalize
        self.wrap = wrap
        self.add_cls = add_cls

        self.update(
            n_tokens_h,
            n_tokens_w,
            n_reg_tokens=n_reg_tokens,
            metric=metric,
            normalize=normalize,
            wrap=wrap,
            add_cls=add_cls,
            force_update=True,
        )

    def update(
        self,
        n_tokens_h: int,
        n_tokens_w: int,
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
        force_update: bool = False,
    ) -> None:
        is_stale = False
        for attr, val in (
            ("n_tokens_h", n_tokens_h),
            ("n_tokens_w", n_tokens_w),
            ("n_reg_tokens", n_reg_tokens),
            ("metric", metric),
            ("normalize", normalize),
            ("wrap", wrap),
            ("add_cls", add_cls),
        ):
            if getattr(self, attr) != val:
                is_stale = True

        if not is_stale and not force_update:
            # nop if nothing has changed
            return

        try:
            device = self.matrix.device
            dtype = self.matrix.dtype
        except AttributeError:
            device = "cpu"
            dtype = torch.float32

        distance_matrix = get_distance_matrix(
            n_tokens_h,
            n_tokens_w,
            self.n_reg_tokens,
            wrap=self.wrap,
            metric=self.metric,
            normalize=self.normalize,
            add_cls=self.add_cls,
            device=device,
            dtype=dtype,
        )

        self.n_tokens_h = n_tokens_h
        self.n_tokens_w = n_tokens_w

        self.register_buffer("matrix", distance_matrix, persistent=False)


class AlibiAttention(Attention):
    def __init__(
        self,
        distance_matrix: DistanceMatrixWrapper,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        slope_type: AlibiSlopeType = "constant",
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.distance_matrix = distance_matrix
        self.set_alibi_slope(slope_type=slope_type)

        self.n_tokens_h = 16
        self.n_tokens_w = 16

        self.set_alibi_slope(slope_type)

    def set_alibi_slope(self, slope_type: AlibiSlopeType):
        m = get_alibi_slope(self.num_heads, slope_type=slope_type, device=self.qkv.weight.device)  # type: ignore

        if isinstance(m, nn.Parameter):
            self.register_parameter("m", m)
        elif isinstance(m, torch.Tensor):
            self.register_buffer("m", m, persistent=False)
        else:
            raise Exception(f"Unexpected slope type {type(m)}")

    def forward(self, x: torch.Tensor, attn_mask=None, attn_bias=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        bias = self.m * self.distance_matrix.matrix  # type: ignore
        bias = bias.unsqueeze(0)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=bias
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AlibiBlock(Block):
    def __init__(
        self,
        distance_matrix: DistanceMatrixWrapper,
        slope_type: AlibiSlopeType,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = AlibiAttention(
            distance_matrix,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            slope_type=slope_type,
        )


class AlibiVitWrapper(PretrainedViTWrapper):
    def __init__(
        self,
        model_identifier: str = DEFAULT_MODEL,
        stride: int = 14,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: str = "cpu",
        slope_type: AlibiSlopeType = "constant",
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
        **kwargs,
    ):
        distance_matrix = DistanceMatrixWrapper(
            n_tokens_h=16,
            n_tokens_w=16,
            n_reg_tokens=n_reg_tokens,
            metric=metric,
            normalize=normalize,
            wrap=wrap,
            add_cls=add_cls,
        )

        def block_fn_wrapper(dim: int, num_heads: int, **block_kwargs: dict) -> AlibiBlock:
            return AlibiBlock(
                distance_matrix=distance_matrix,
                slope_type=slope_type,
                dim=dim,
                num_heads=num_heads,
                **block_kwargs,
            )

        super().__init__(
            model_identifier=model_identifier,
            stride=stride,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            device=device,
            block_fn=block_fn_wrapper,
            **kwargs,
        )
        self.n_cls_tokens = 0 if not add_cls else 1
        self.n_reg_tokens = n_reg_tokens

        if self.n_cls_tokens == 0:
            self.model.cls_token = None

        if self.n_reg_tokens == 0:
            self.model.reg_token = None

        self.distance_matrix = distance_matrix
        self.slope_type = slope_type

    def forward_features(self, x: torch.Tensor, make_2D: bool = False, add_reg: bool = False, **kwargs) -> torch.Tensor:
        assert self.model.pos_embed is not None
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p) // s + 1, (w - p) // s + 1

        self.distance_matrix.update(n_patch_h, n_patch_w)

        feats = super().forward_features(x, make_2D, add_reg)
        return feats

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | list[int] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ):
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p) // s + 1, (w - p) // s + 1

        self.distance_matrix.update(n_patch_h, n_patch_w)
        return super().get_intermediate_layers(x, n, reshape, return_prefix_tokens, norm)


if __name__ == "__main__":
    dv2 = AlibiVitWrapper()
    x = torch.zeros((1, 3, 14 * 4, 14 * 4))
    o = dv2.forward_features(x, True, False, abs_pos_enc_drop_prob=0.5)
    print(o.shape)
