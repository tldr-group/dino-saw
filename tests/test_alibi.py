import torch
import pytest
import math
from dinosaw.alibi_logic import (
    build_2d_sincos_pos_embed,
    get_distance_matrix,
    DistanceMatrixWrapper,
)
from dinosaw.wrappers.alibi import convert_timm_model, add_alibi
from timm import create_model
from PVW import WrapperRegistry
from dinosaw.wrappers import DebiasedViTWrapper, DenoisingViTWrapper, PretrainedViTWrapper


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
    dm = get_distance_matrix(
        h, w, n_reg_tokens=0, metric="manhattan", normalize=False, wrap=False, add_cls=False
    )
    # [0, 0] to [1, 4] -> unwrapped Manhattan distance is |0-1| + |0-4| = 5.
    # Distances are negative in distance matrix.
    assert abs(dm[0, 9].item() - (-5.0)) < 1e-5

    # 2. Manhattan, no norm, wrap
    dm_wrap = get_distance_matrix(
        h, w, n_reg_tokens=0, metric="manhattan", normalize=False, wrap=True, add_cls=False
    )
    # Wrapped Manhattan: h-diff = min(1, 2-1) = 1, w-diff = min(4, 5-4) = 1. Distance = 2.
    assert abs(dm_wrap[0, 9].item() - (-2.0)) < 1e-5

    # 3. Manhattan, normalize, no wrap
    dm_norm = get_distance_matrix(
        h, w, n_reg_tokens=0, metric="manhattan", normalize=True, wrap=False, add_cls=False
    )
    # Max distance is 5. Normalized distance is 5 / 5 = 1.
    assert abs(dm_norm[0, 9].item() - (-1.0)) < 1e-5

    # 4. Euclidean, no norm, no wrap
    dm_eucl = get_distance_matrix(
        h, w, n_reg_tokens=0, metric="euclidean", normalize=False, wrap=False, add_cls=False
    )
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



