"""Taken from https://github.com/Jiawei-Yang/Denoising-ViT"""

from functools import partial

import torch
from PVW import PretrainedViTWrapper
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import nn


class DenoisingViT(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feat_dim: int = 768,
        vit: PretrainedViTWrapper | None = None,
        enable_pe: bool = True,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.vit = vit
        self.denoiser = Block(
            dim=feat_dim,
            num_heads=feat_dim // 64,
            mlp_ratio=4,
            qkv_bias=True,
            qk_norm=False,
            init_values=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            mlp_layer=Mlp,
        )
        if num_blocks > 1:
            self.denoiser = nn.Sequential(
                *[
                    Block(
                        dim=feat_dim,
                        num_heads=feat_dim // 64,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_norm=False,
                        init_values=None,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        act_layer=nn.GELU,
                        mlp_layer=Mlp,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.pos_embed = None
        if enable_pe:
            seq_len = noise_map_height * noise_map_width
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feat_dim) * 0.02)
        if self.vit is not None:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(
        self,
        x,
        return_dict=False,
        return_channel_first=False,
        return_class_token=False,
        norm=True,
    ):
        class_tokens = None
        if self.vit is not None:
            with torch.no_grad():
                # get_intermediate_layers in PVW returns NCHW formatted features if make_2D=True
                # let's call forward_intermediates from the vit wrapper
                # which returns a list of [B, C, H, W] tensors if make_2D=True
                vit_outputs = self.vit.forward_intermediates(
                    x,
                    n=[self.vit.num_blocks - 1],
                    make_2D=True,
                    norm=norm,
                )
                original_feats = vit_outputs[0].permute(0, 2, 3, 1)  # [B, H, W, C]
                x = original_feats
        else:
            original_feats = x.clone()
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        x = self.denoiser(x)
        x = x.reshape(b, h, w, c)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)
        if return_dict:
            return {
                "denoised_feats": x,
                "original_feats": original_feats.detach(),
                "class_tokens": class_tokens.detach() if class_tokens is not None else None,
            }
        if return_class_token:
            assert class_tokens is not None
            return x, class_tokens
        return x

    def forward_(
        self, x: torch.Tensor, permute_channel: bool = True, return_channel_first: bool = True
    ) -> torch.Tensor:
        if permute_channel:
            x = x.permute((0, 2, 3, 1))
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        x = self.denoiser(x)
        x = x.reshape(b, h, w, c)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class DenoisingViTWrapper(PretrainedViTWrapper):
    def __init__(
        self,
        vit,
        device: torch.device | str,
        denoiser_path: str | None = None,
        **kwargs,
    ):
        super().__init__(vit=vit, device=device, **kwargs)
        self.denoiser = get_denoiser(denoiser_path, device=device, to_eval=True)

    def forward_features(
        self,
        x: torch.Tensor,
        make_2D: bool = True,
    ) -> torch.Tensor:
        # Extract features with make_2D=True as denoiser expects 4D NCHW tensor
        feats = super().forward_features(x, make_2D=True)
        denoised = self.denoiser.forward_(feats, permute_channel=True, return_channel_first=True)
        if not make_2D:
            b, c, h, w = denoised.shape
            denoised = denoised.view(b, c, h * w)
        return denoised


def get_denoiser(
    chk_path: str | None, device: str = "cpu", to_eval: bool = False, to_half: bool = False
) -> DenoisingViT:
    model = DenoisingViT(feat_dim=384)
    if chk_path is not None:
        obj = torch.load(chk_path, weights_only=True, map_location=device)
        model.load_state_dict(obj["denoiser"])
    model = model.to(device)
    if to_eval:
        model = model.eval()
    if to_half:
        model = model.half()
    return model



