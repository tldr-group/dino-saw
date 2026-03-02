"""
A simple wrapper around timm vision transformers that allow for adjusting the model stride
and has a helper to get patch features out.

Adpated from https://github.com/Jiawei-Yang/Denoising-ViT/blob/main/dvt/models/vit_wrapper.py
"""

import torch
from torch import nn
from torchvision import transforms

from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.models.vision_transformer import VisionTransformer, Attention, Block


from dinosaw.models.alibi import AlibiBlock, AlibiSlopeType, DistanceMatrixWrapper

import re
from typing import cast, Callable, Literal
from types import MethodType


FLASH_ATTN_INSTALLED = False
try:
    from flash_attn import flash_attn_qkvpacked_func

    print("flash attention installed")
    FLASH_ATTN_INSTALLED = True
except ImportError:
    pass


FeatureType = Literal["FEATUP", "LOFTUP_FULL", "LOFTUP_COMPRESSED"]
FIT3D_DINOv2_REG_SMALL_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_reg_small_finetuned.pth"

MODEL_LIST = [
    # DINOv2
    "vit_small_patch14_dinov2.lvd142m",
    # DINOv2 + register
    "vit_small_patch14_reg4_dinov2.lvd142m",
    # FIT3D finetuned
    "fit3D_vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    # DINOv3
    # "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vits_patch16_plus_reg4",
    "vit_small_patch16_224.dino",
    "vit_base_patch16_224.mae",
    "vit_base_patch16_clip_224.openai",
    "eva02_base_patch14_224.mim_in22k",
    "samvit_base_patch16.sa1b",
    "convnextv2_tiny.fcmae",
    # "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "deit3_small_patch16_224.fb_in1k",
    "tiny_vit_5m_224.dist_in22k_ft_in1k",
    "vit_base_patch16_224.dino",
]
MODEL_MAP: dict[FeatureType, str] = {
    "FEATUP": MODEL_LIST[2],
    "LOFTUP_FULL": MODEL_LIST[1],
    "LOFTUP_COMPRESSED": MODEL_LIST[1],
}


class Patch:
    @staticmethod
    def add_flash_attn() -> Callable:
        """Replaces normal 'forward()' method of the memory efficient attention layer (block.attn)
        in the Dv2 model with an optional early return with attention. Used if xformers used.

        :return: the new forward method
        :rtype: Callable
        """

        def forward(
            self: Attention,
            x: torch.Tensor,
            attn_mask=None,
            attn_bias=None,
        ) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            x = flash_attn_qkvpacked_func(qkv)  # type: ignore
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward


def add_flash_attention(model: VisionTransformer) -> VisionTransformer:
    if FLASH_ATTN_INSTALLED is False:
        print("Warning: no flash attn available")
        return model
    blk: Block
    for blk in model.blocks:  # type: ignore
        blk.attn.forward = MethodType(Patch.add_flash_attn(), blk.attn)
    return model


class PretrainedViTWrapper(nn.Module):
    def __init__(
        self,
        model_identifier: str = "vit_base_patch14_dinov2.lvd142m",
        stride: int = 14,
        add_flash_attn: bool = True,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: str = "cpu",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert model_identifier in MODEL_LIST, (
            f"Model type {model_identifier} not tested yet."
        )
        self.model_identifier = model_identifier

        self.stride = stride
        patch_size_from_id = re.search(r"patch(\d+)", model_identifier)
        self.patch_size = (
            14 if patch_size_from_id is None else int(patch_size_from_id.group(1))
        )

        n_reg_tokens_from_id = re.search(r"reg(\d+)", model_identifier)
        self.n_reg_tokens = (
            0 if n_reg_tokens_from_id is None else int(n_reg_tokens_from_id.group(1))
        )
        self.n_cls_tokens = 0 if "sam" in model_identifier else 1

        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.pretrained = pretrained
        self.model, self.transformation = self.create_model(
            model_identifier, device, **kwargs
        )

        # overwrite the stride size
        if (
            stride != self.model.patch_embed.proj.stride[0]
            and "convnext" not in model_identifier
        ):
            self.model.patch_embed.proj.stride = (stride, stride)

            def dynamic_feat_size(self, img_size: tuple[int, int]) -> tuple[int, int]:
                """Get grid (feature) size for given image size taking account of dynamic padding.
                NOTE: must be torchscript compatible so using fixed tuple indexing
                """
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = MethodType(
                dynamic_feat_size, self.model.patch_embed
            )

        if add_flash_attn:
            self.model = self.model.half()
            self.model = add_flash_attention(self.model)

        self.to(device)

    @property
    def n_output_dims(self) -> int:
        assert self.model.pos_embed
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self, model_identifier: str, device: str, **kwargs
    ) -> tuple[VisionTransformer, transforms.Compose]:
        is_fit3D = "fit3D" in model_identifier
        if is_fit3D:
            model_identifier = model_identifier[6:]

        is_dv3 = "dinov3" in model_identifier
        if is_dv3:
            path = kwargs["chk_path"]
            conf_path = kwargs["conf_path"]
            assert path is not None
            model = torch.hub.load(
                conf_path, "dinov3_vits16plus", source="local", weights=path
            )
            return model, None

        if "sam" in model_identifier or "conv" in model_identifier:
            model = create_model(
                model_identifier,
                pretrained=self.pretrained,  # True,
                num_classes=0,
                **kwargs,
            )
        else:
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
        img_transforms = cast(
            transforms.Compose, create_transform(**data_config, is_training=False)
        )

        if is_fit3D:
            # load finetuned weights
            state_dict = torch.hub.load_state_dict_from_url(
                FIT3D_DINOv2_REG_SMALL_URL, map_location=device
            )
            model.load_state_dict(state_dict)

        return model, img_transforms

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | list[int] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
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

        if "dinov3" in self.model_identifier:
            feats = self.model.forward_features(x)["x_norm_patchtokens"]
            feats = feats.permute((0, 2, 1))
            if make_2D:
                feats = feats.reshape((b, -1, n_patch_h, n_patch_w))
            return feats

        if add_reg:
            feats = self.model.forward_features(x)
        else:  # ignore CLS + reg tokens
            feats = self.model.forward_features(x)[
                :, self.n_reg_tokens + self.n_cls_tokens :
            ]

        if "sam" in self.model_identifier:
            return feats
        feats = feats.permute((0, 2, 1))

        if make_2D and not add_reg:
            feats = feats.reshape((b, -1, n_patch_h, n_patch_w))
        return feats


class AlibiVitWrapper(PretrainedViTWrapper):
    def __init__(
        self,
        model_identifier: str = "vit_small_patch14_dinov2.lvd142m",
        stride: int = 14,
        add_flash_attn: bool = True,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: str = "cpu",
        slope_type: AlibiSlopeType = "constant",
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
        jitter_mag: float = 0.0,
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

        def block_fn_wrapper(
            dim: int, num_heads: int, **block_kwargs: dict
        ) -> AlibiBlock:
            return AlibiBlock(
                distance_matrix=distance_matrix,
                slope_type=slope_type,
                jitter_mag=jitter_mag,
                dim=dim,
                num_heads=num_heads,
                **block_kwargs,
            )

        super().__init__(
            model_identifier=model_identifier,
            stride=stride,
            add_flash_attn=add_flash_attn,
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
        # self.model.pos_embed.requires_grad = False  # freeze pos embedding
        self.slope_type = slope_type

        # for blk in self.model.blocks:
        #     blk: AlibiBlock
        #     blk.attn.set_alibi_slope(slope_type=self.slope_type)

    def set_alibi_enabled(self, enabled: bool):
        return

    def forward(self, x: torch.Tensor):
        # TODO: set alibi distance matrix size
        return self.model(x)

    def forward_features(
        self, x: torch.Tensor, make_2D: bool = False, add_reg: bool = False, **kwargs
    ) -> torch.Tensor:
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
        return super().get_intermediate_layers(
            x, n, reshape, return_prefix_tokens, norm
        )

    def forward_features_with_pos_enc_dropout(
        self,
        x: torch.Tensor,
        make_2D: bool = False,
        add_reg: bool = False,
        abs_pos_enc_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        assert self.model.pos_embed is not None
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p) // s + 1, (w - p) // s + 1

        self.distance_matrix.update(n_patch_h, n_patch_w)

        do_abs_pos_enc_drop = torch.rand(1).item() <= abs_pos_enc_drop_prob
        cached_pos_embed_data = self.model.pos_embed.data.clone()
        if do_abs_pos_enc_drop and self.training:
            self.model.pos_embed.data.zero_()

        feats = super().forward_features(x, make_2D, add_reg)
        self.model.pos_embed.data.copy_(cached_pos_embed_data)
        return feats


if __name__ == "__main__":
    dv2 = AlibiVitWrapper(
        "fit3D_vit_small_patch14_reg4_dinov2.lvd142m", add_flash_attn=False
    )
    x = torch.zeros((1, 3, 14 * 4, 14 * 4))
    o = dv2.forward_features(x, True, False, abs_pos_enc_drop_prob=0.5)
    print(o.shape)
