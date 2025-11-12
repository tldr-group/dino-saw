from timm.models import VisionTransformer
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, Attention

import numpy as np
import torch
from torch import Tensor, randn
import torch.nn as nn
import torch.nn.functional as F

from dinosaw.models.vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_LIST,
)


from typing import Optional, Type

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

def wrapped_distance_matrix(
        N: int,
        metric: str = 'manhattan', 
        device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the distance between each pixel (with wrapped-bounday condition).
    returns a Matrix with shape (H*W, H*W).
    metric: 'euclidean' or 'manhattan'
    """
    H,W = int(np.sqrt(N)), int(np.sqrt(N))
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),#, dtype=torch.float32),
        torch.arange(W, device=device),#, dtype=torch.float32),
        indexing='ij'
    ), dim=-1).reshape(-1, 2)  # (N, 2)

    # Paarweise Differenzen
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)

    # Wrapped Distanzen
    diff = diff.abs()
    diff[..., 0] = torch.minimum(diff[..., 0], H - diff[..., 0])
    diff[..., 1] = torch.minimum(diff[..., 1], W - diff[..., 1])

    if metric == 'euclidean':
        D = torch.sqrt((diff ** 2).sum(-1))
    elif metric == 'manhattan':
        D = diff.sum(-1) * (-1)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    
    # adding padding for CLS und register tokens
    #print(D.shape)
    D = F.pad(D, (0,5,0,5), mode="constant")
    #print(D.shape)
    return D

class AlibiAttention(Attention):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.fused_attn = False
        self.register_buffer("m", get_alibi_slope(self.num_heads))

    def forward(self, x: Tensor, attn_mask: None) -> Tensor:
        # TODO: consider flexAttention here for efficient biased attention w/ custom score function
        # https://pytorch.org/blog/flexattention/

        # TODO: Problems with registers in ALIBI

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            #print(q.shape, k.shape)
            attn = q @ k.transpose(-2, -1)

            #print(attn.shape)

            bias = (self.m * wrapped_distance_matrix(N)).unsqueeze(0) # Alibi bias
            #print(bias.shape)
            attn = attn + bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print(x.shape)
        return x



class AlibiBlock(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm:bool = False,
            scale_mlp_norm:bool=False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias, proj_drop=proj_drop, attn_drop=attn_drop, init_values=init_values, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer)
        self.attn = AlibiAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

if __name__ == "__main__":
    
    vt = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, block_fn=AlibiBlock)#, pos_embed="none")
    print(vt.model)
    #print(vt.n_output_dims())
    
    vt.pos_embed = None
    in_t = randn(1, 3, 224, 224)
    out = vt.forward_features(in_t, make_2D=True)
    print(out.shape)
    # print(vt)