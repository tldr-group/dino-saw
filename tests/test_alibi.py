import math

import torch
from PVW import WrapperRegistry
from timm import create_model

from dinosaw.alibi_logic import (
    DistanceMatrixWrapper,
    build_2d_sincos_pos_embed,
    get_distance_matrix,
)
from dinosaw.wrappers.alibi import add_alibi


def test_build_2d_sincos_pos_embed():
    h, w = 8, 8
    embed_dim = 64
    pos_embed = build_2d_sincos_pos_embed(h, w, embed_dim, dtype=torch.float32, device="cpu")
    assert pos_embed.shape == (1, h * w, embed_dim)


def test_get_distance_matrix():
    h, w = 4, 4
    dm = get_distance_matrix(h, w, n_reg_tokens=0, metric="euclidean", wrap=False, add_cls=False)
    assert dm.shape == (h * w, h * w)
    for i in range(h * w):
        assert abs(dm[i, i].item()) < 1e-6


def test_distance_matrix_wrapper():
    dmw = DistanceMatrixWrapper(
        n_tokens_h=4,
        n_tokens_w=4,
        n_reg_tokens=2,
        metric="euclidean",
        normalize=True,
        wrap=True,
        add_cls=True,
    )
    # Total tokens = 1 (cls) + 2 (reg) + 16 (spatial) = 19
    assert dmw.matrix.shape == (19, 19)

    # Test update
    dmw.update(n_tokens_h=5, n_tokens_w=5)
    # Total tokens = 1 + 2 + 25 = 28
    assert dmw.matrix.shape == (28, 28)


def test_add_alibi_timm():
    model = create_model("vit_tiny_patch16_224", pretrained=False)
    model = add_alibi(
        model,
        slope_type="constant",
        n_reg_tokens=0,
        add_cls=False,
    )
    assert hasattr(model, "distance_matrix")
    from dinosaw.alibi_logic import AlibiAttention

    assert isinstance(model.blocks[0].attn, AlibiAttention)


def test_distance_matrix_detailed():
    h, w = 2, 5

    # 1. Manhattan, no norm, no wrap
    dm = get_distance_matrix(h, w, n_reg_tokens=0, metric="manhattan", normalize=False, wrap=False, add_cls=False)
    # [0, 0] to [1, 4] -> unwrapped Manhattan distance is |0-1| + |0-4| = 5.
    # Distances are negative in distance matrix.
    assert abs(dm[0, 9].item() - (-5.0)) < 1e-5

    # 2. Manhattan, no norm, wrap
    dm_wrap = get_distance_matrix(h, w, n_reg_tokens=0, metric="manhattan", normalize=False, wrap=True, add_cls=False)
    # Wrapped Manhattan: h-diff = min(1, 2-1) = 1, w-diff = min(4, 5-4) = 1. Distance = 2.
    assert abs(dm_wrap[0, 9].item() - (-2.0)) < 1e-5

    # 3. Manhattan, normalize, no wrap
    dm_norm = get_distance_matrix(h, w, n_reg_tokens=0, metric="manhattan", normalize=True, wrap=False, add_cls=False)
    # Max distance is 5. Normalized distance is 5 / 5 = 1.
    assert abs(dm_norm[0, 9].item() - (-1.0)) < 1e-5

    # 4. Euclidean, no norm, no wrap
    dm_eucl = get_distance_matrix(h, w, n_reg_tokens=0, metric="euclidean", normalize=False, wrap=False, add_cls=False)
    # sqrt((0-1)^2 + (0-4)^2) = sqrt(17) ~= 4.1231
    expected = -math.sqrt(17.0)
    assert abs(dm_eucl[0, 9].item() - expected) < 1e-5


def test_alibi_forward_and_update():
    # Build alibi model
    model = WrapperRegistry.build("alibi_dinov2_s", device="cpu")
    model = model.eval()

    # Pass 1: 224 x 224 input
    x1 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model.forward_features(x1, make_2D=True)
    # Stride is 14. 224 / 14 = 16.
    assert model.vit.distance_matrix.n_tokens_h == 16
    assert model.vit.distance_matrix.n_tokens_w == 16
    # cls (1) + reg (4) + spatial (256) = 261
    assert model.vit.distance_matrix.matrix.shape == (261, 261)

    # Pass 2: subsequent forward pass with different resolution (112 x 112)
    x2 = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out2 = model.forward_features(x2, make_2D=True)
    # 112 / 14 = 8.
    assert model.vit.distance_matrix.n_tokens_h == 8
    assert model.vit.distance_matrix.n_tokens_w == 8
    # cls (1) + reg (4) + spatial (64) = 69
    assert model.vit.distance_matrix.matrix.shape == (69, 69)


def _reference_get_distance_matrix(
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
    """Original out-of-place reference implementation of get_distance_matrix."""
    import torch.nn.functional as F

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(n_tokens_h, device=device, dtype=dtype),
            torch.arange(n_tokens_w, device=device, dtype=dtype),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
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
    D = F.pad(D, (n_extra_tokens, 0, n_extra_tokens, 0), mode="constant")
    return D.to(device=device, dtype=dtype)


def _reference_alibi_attn_forward(attn_module, x: torch.Tensor, ref_dm: torch.Tensor) -> torch.Tensor:
    """Original out-of-place reference implementation of AlibiAttention.forward with (1, H, N, N) bias."""
    import torch.nn.functional as F

    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    m_full = attn_module.m.expand(attn_module.num_heads, 1, 1)
    bias = (m_full * ref_dm).unsqueeze(0)

    if attn_module.fused_attn:
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0, attn_mask=bias
        )
    else:
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        out = attn @ v

    out = out.transpose(1, 2).reshape(B, N, C)
    out = attn_module.proj(out)
    out = attn_module.proj_drop(out)
    return out


def test_inplace_distance_matrix_equivalence():
    for h, w in [(4, 4), (7, 11), (16, 16)]:
        for n_reg in [0, 4]:
            for metric in ["euclidean", "manhattan"]:
                for normalize in [True, False]:
                    for wrap in [True, False]:
                        for add_cls in [True, False]:
                            dm_new = get_distance_matrix(
                                h,
                                w,
                                n_reg_tokens=n_reg,
                                metric=metric,
                                normalize=normalize,
                                wrap=wrap,
                                add_cls=add_cls,
                            )
                            dm_ref = _reference_get_distance_matrix(
                                h,
                                w,
                                n_reg_tokens=n_reg,
                                metric=metric,
                                normalize=normalize,
                                wrap=wrap,
                                add_cls=add_cls,
                            )
                            assert dm_new.shape == dm_ref.shape
                            assert dm_new.stride(0) % 8 == 0
                            assert torch.allclose(dm_new, dm_ref, atol=1e-6, rtol=1e-6)


def test_alibi_forward_equivalence_with_reference():
    devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        for slope_type in ["constant", "fixed", "learned"]:
            for fused_attn in [False, True]:
                torch.manual_seed(42)
                model_new = create_model("vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True)
                model_new.pos_embed = None
                model_new = add_alibi(
                    model_new,
                    slope_type=slope_type,
                    n_reg_tokens=0,
                    add_cls=True,
                ).to(device).eval()

                model_ref = create_model("vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True)
                model_ref.pos_embed = None
                model_ref = add_alibi(
                    model_ref,
                    slope_type=slope_type,
                    n_reg_tokens=0,
                    add_cls=True,
                ).to(device).eval()
                model_ref.load_state_dict(model_new.state_dict(), strict=False)

                for b_new, b_ref in zip(model_new.blocks, model_ref.blocks):
                    b_new.attn.fused_attn = fused_attn
                    b_ref.attn.fused_attn = fused_attn
                    with torch.no_grad():
                        b_ref.attn.m.copy_(b_new.attn.m)

                    orig_attn = b_ref.attn

                    def _make_ref_forward(attn_mod):
                        def _fwd(x, attn_mask=None, attn_bias=None, **kwargs):
                            dm_wrapper = attn_mod.distance_matrix
                            ref_dm = _reference_get_distance_matrix(
                                dm_wrapper.n_tokens_h,
                                dm_wrapper.n_tokens_w,
                                n_reg_tokens=dm_wrapper.n_reg_tokens,
                                metric=dm_wrapper.metric,
                                normalize=dm_wrapper.normalize,
                                wrap=dm_wrapper.wrap,
                                add_cls=dm_wrapper.add_cls,
                                device=x.device,
                                dtype=x.dtype,
                            )
                            return _reference_alibi_attn_forward(attn_mod, x, ref_dm)

                        return _fwd

                    orig_attn.forward = _make_ref_forward(orig_attn)

                for img_size in [224, 160]:
                    x = torch.randn(2, 3, img_size, img_size, device=device)
                    with torch.no_grad():
                        out_new = model_new.forward_features(x)
                        out_ref = model_ref.forward_features(x)
                    assert torch.allclose(out_new, out_ref, atol=1e-5, rtol=1e-5), (
                        f"Mismatch for device={device}, slope_type={slope_type}, fused_attn={fused_attn}, "
                        f"img_size={img_size}: max diff={(out_new - out_ref).abs().max().item()}"
                    )

