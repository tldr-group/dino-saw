import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, Attention


from typing import Type, Literal, Optional


def get_distance_matrix(
    n_tokens_h: int,
    n_tokens_w: int,
    n_reg_tokens: int = 4,
    metric: str = "euclidean",
    normalize: bool = True,
    wrap: bool = False,
    add_cls: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    # TODO: is this (H,W) or (W,H) - and which is right?
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
    torch.cuda.empty_cache()
    return D.to(device)


AlibiSlopeType = Literal["fixed", "learned", "constant"]


def get_alibi_slope(
    num_heads: int, slope_type: AlibiSlopeType = "constant", device: str = "cpu"
) -> torch.Tensor | nn.Parameter:
    m: torch.Tensor | nn.Parameter
    match slope_type:
        case "fixed":
            xs = (2**8) ** (1 / num_heads)
            m = torch.tensor(
                [1 / xs ** (i + 1) for i in range(num_heads)], device=device
            )
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


class AlibiAttention(Attention):
    def __init__(
        self,
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
        self.fused_attn = False

        self.set_alibi_slope(slope_type=slope_type)

        self.is_enabled = True

    def set_alibi_slope(self, slope_type: AlibiSlopeType):
        m = get_alibi_slope(
            self.num_heads, slope_type=slope_type, device=self.qkv.weight.device
        )
        if isinstance(m, torch.Tensor):
            self.register_buffer("m", m)
        else:
            self.register_parameter("m", m)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        bias = attn_mask.unsqueeze(0) if attn_mask is not None else None

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=bias,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn + bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn.to(v.dtype)
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
        **kwargs,
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
    import matplotlib.pyplot as plt
    from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST

    h, w = 8, 30

    dm = get_distance_matrix(
        h,
        w,
        n_reg_tokens=0,
        add_cls=False,
        metric="euclidean",
        wrap=True,
        normalize=False,
        device="cpu",
    )
    dm_np = dm.detach().cpu().numpy()
    print(dm_np.shape)

    plt.imsave("tmp/dm.png", dm_np, cmap="hot")
    plt.imsave("tmp/dm_heatmap.png", dm_np[31].reshape((h, w)), cmap="hot")

    vt = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, block_fn=AlibiBlock)

    in_t = torch.rand(1, 3, h * 14, w * 14)

    out = vt.forward_features(in_t, make_2D=True)
    print(out.shape)
