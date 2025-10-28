from timm.models import VisionTransformer
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, Attention

from torch import Tensor, randn
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, Type


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
        super().__init__(dim, num_heads, qkv_bias, proj_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.fused_attn = False

    def forward(self, x: Tensor) -> Tensor:
        # TODO: consider flexAttention here for efficient biased attention w/ custom score function
        # https://pytorch.org/blog/flexattention/
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
            print(q.shape, k.shape)
            attn = q @ k.transpose(-2, -1)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class AlibiBlock(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, proj_bias, proj_drop, attn_drop, init_values, drop_path, act_layer, norm_layer, mlp_layer)
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
    vt = VisionTransformer(block_fn=AlibiBlock, pos_embed="none",)
    in_t = randn(1, 3, 224, 224)
    out = vt.forward_features(in_t)
    print(out.shape)
    print(vt.pos_embed)