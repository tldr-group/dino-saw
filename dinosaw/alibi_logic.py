from typing import Literal

import torch
import torch.nn.functional as F
from dinov3.layers.attention import SelfAttention as DV3SelfAttention
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Attention, Block
from torch import nn


def build_2d_sincos_pos_embed(
    h: int,
    w: int,
    embed_dim: int,
    dtype,
    device,
    base_wavelength: int = 10_000,
) -> torch.Tensor:
    """
    Returns:
        (1, h*w, embed_dim)
    """

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    # grid of patch coordinates
    grid_h = torch.arange(h, dtype=dtype, device=device)
    grid_w = torch.arange(w, dtype=dtype, device=device)

    y, x = torch.meshgrid(grid_h, grid_w, indexing="ij")

    y = y.reshape(-1)  # (H*W,)
    x = x.reshape(-1)

    dim_quarter = embed_dim // 4

    omega = torch.arange(dim_quarter, dtype=dtype, device=device)
    omega = 1.0 / (base_wavelength ** (omega / dim_quarter))

    out_y = y[:, None] * omega[None, :]
    out_x = x[:, None] * omega[None, :]

    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

    pos_embed = torch.cat([emb_y, emb_x], dim=1)

    return pos_embed.unsqueeze(0)


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
            torch.arange(n_tokens_h, device=device, dtype=dtype),
            torch.arange(n_tokens_w, device=device, dtype=dtype),
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


class AlibiSelfAttention(DV3SelfAttention):
    """ALiBi-style attention compatible with DINOv3's SelfAttention implementation.

    This wraps the original DV3 `SelfAttention` but injects an ALiBi bias
    computed from a `DistanceMatrixWrapper`.
    """

    def __init__(
        self,
        *args,
        distance_matrix: DistanceMatrixWrapper | None = None,
        slope_type: AlibiSlopeType = "constant",
        jitter_mag: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        m = get_alibi_slope(self.num_heads, slope_type=slope_type, device=self.qkv.weight.device)
        if isinstance(m, torch.nn.Parameter):
            self.register_parameter("m", m)
        else:
            self.register_buffer("m", m, persistent=False)

        self.distance_matrix = distance_matrix
        self.jitter_mag = jitter_mag

    def compute_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.distance_matrix is not None:
            bias = self.m * self.distance_matrix.matrix
            bias = bias.unsqueeze(0)

            attn = attn + bias.to(attn.dtype)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


def convert_dv3_model(
    model,
    slope_type: AlibiSlopeType = "constant",
    jitter_mag: float = 0.0,
    n_tokens_h: int = 16,
    n_tokens_w: int = 16,
    distance_matrix: DistanceMatrixWrapper | None = None,
):
    """Disable RoPE on a DinoV3 model and replace its attention blocks with ALiBi attention.

    Returns the modified model (in-place).
    """
    try:
        model.rope_embed = None
    except Exception:
        model.rope_embed = None

    # use provided distance_matrix if supplied, otherwise create one
    if distance_matrix is None:
        n_reg = getattr(model, "n_storage_tokens", 0)
        dm = DistanceMatrixWrapper(n_tokens_h, n_tokens_w, n_reg_tokens=n_reg, wrap=True, add_cls=True)
    else:
        dm = distance_matrix
        # ensure dm has the requested token grid
        try:
            dm.update(n_tokens_h, n_tokens_w)
        except Exception:
            pass
    model.distance_matrix = dm

    for blk in model.blocks:
        old_attn = blk.attn
        new_attn = AlibiSelfAttention(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=(old_attn.qkv.bias is not None),
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            mask_k_bias=hasattr(old_attn.qkv, "bias_mask"),
            device=old_attn.qkv.weight.device,
            distance_matrix=model.distance_matrix,
            slope_type=slope_type,
            jitter_mag=jitter_mag,
        )

        try:
            new_attn.qkv.load_state_dict(old_attn.qkv.state_dict())
        except Exception:
            pass
        try:
            new_attn.proj.load_state_dict(old_attn.proj.state_dict())
        except Exception:
            pass

        if hasattr(old_attn, "attn_drop"):
            new_attn.attn_drop = old_attn.attn_drop
        if hasattr(old_attn, "proj_drop"):
            new_attn.proj_drop = old_attn.proj_drop

        blk.attn = new_attn

    return model


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
        norm_layer: type[nn.Module] = nn.LayerNorm,
        slope_type: AlibiSlopeType = "constant",
        jitter_mag: float = 0.0,
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
        self.fused_attn = False
        self.set_alibi_slope(slope_type=slope_type)

        self.n_tokens_h = 16
        self.n_tokens_w = 16

        self.jitter_mag = jitter_mag
        self.set_alibi_slope(slope_type)

    def set_alibi_slope(self, slope_type: AlibiSlopeType):
        m = get_alibi_slope(self.num_heads, slope_type=slope_type, device=self.qkv.weight.device)

        if isinstance(m, nn.Parameter):
            # delattr(self, "m")
            self.register_parameter("m", m)
        elif isinstance(m, torch.Tensor):
            self.register_buffer("m", m, persistent=False)
        else:
            raise Exception(f"Unexpected slope type {type(m)}")

    def forward(self, x: torch.Tensor, attn_mask=None, attn_bias=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        bias = self.m * self.distance_matrix.matrix
        bias = bias.unsqueeze(0)

        if self.jitter_mag > 0.0:
            jitter = torch.randn_like(bias) * self.jitter_mag
            bias += jitter

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=bias
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn + bias

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
        distance_matrix: DistanceMatrixWrapper,
        slope_type: AlibiSlopeType,
        jitter_mag: float,
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
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_layer: type[nn.Module] = Mlp,
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
            jitter_mag=jitter_mag,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from dinosaw.wrappers import MODEL_LIST, PretrainedViTWrapper

    h, w = 8, 30

    dm = get_distance_matrix(
        h, w, n_reg_tokens=0, add_cls=False, metric="euclidean", wrap=True, normalize=False, device="cpu"
    )
    dm_np = dm.detach().cpu().numpy()
    print(dm_np.shape)

    plt.imsave("tmp/dm.png", dm_np, cmap="hot")
    plt.imsave("tmp/dm_heatmap.png", dm_np[31].reshape((h, w)), cmap="hot")

    vt = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, block_fn=AlibiBlock)

    in_t = torch.rand(1, 3, h * 14, w * 14)

    out = vt.forward_features(in_t, make_2D=True)
    print(out.shape)
