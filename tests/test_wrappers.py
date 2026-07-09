import os
import torch
import pytest
from dinosaw.helpers import get_model
from PVW import WrapperRegistry, BackboneRegistry, BackboneConfig
from dinosaw.wrappers import (
    DebiasedViTWrapper,
    DenoisingViTWrapper,
    PretrainedViTWrapper,
    ChannelBlankedWrapper,
    TransformAverageWrapper,
)


def get_chk_dir(filename):
    for path in [
        f"models/{filename}",
        f"models/checkpoints/{filename}",
        f"models/checkpoints/backbones/{filename}",
        f"models/checkpoints/trained/{filename}",
        f"trained_models/{filename}",
    ]:
        if os.path.exists(path):
            return os.path.abspath(os.path.dirname(path))
    return None


dvt_dir = get_chk_dir("dvt.pth")
alibi_dv2_dir = get_chk_dir("alibi_dv2_vits14_reg.pth")
nope_dir = get_chk_dir("nope_dv2_vits14_reg.pth")
alibi_dv3_dir = get_chk_dir("alibi_dv3_ms.pth")


@pytest.mark.skipif(dvt_dir is None, reason="dvt.pth checkpoint missing")
def test_dvt_wrapper_loading():
    model = get_model("dvt", dvt_dir, device="cpu")
    assert isinstance(model, DenoisingViTWrapper)

    # Do inference on different sized tensors
    x1 = torch.randn(1, 3, 224, 224)
    out1 = model.forward_features(x1, make_2D=True)
    assert out1.shape == (1, 384, 16, 16)

    x2 = torch.randn(1, 3, 112, 112)
    out2 = model.forward_features(x2, make_2D=True)
    assert out2.shape == (1, 384, 8, 8)


@pytest.mark.skipif(alibi_dv2_dir is None, reason="alibi_dv2_vits14_reg.pth checkpoint missing")
def test_alibi_dv2_wrapper_loading():
    # Test old name alibi_dv2
    model_old = get_model("alibi_dv2", alibi_dv2_dir, device="cpu")
    assert isinstance(model_old, PretrainedViTWrapper)

    # Test new name alibi_dinov2_s
    model_new = get_model("alibi_dinov2_s", alibi_dv2_dir, device="cpu")
    assert isinstance(model_new, PretrainedViTWrapper)

    # Do inference on different sized tensors
    x1 = torch.randn(1, 3, 224, 224)
    out1 = model_new.forward_features(x1, make_2D=True)
    assert out1.shape == (1, 384, 16, 16)

    x2 = torch.randn(1, 3, 112, 112)
    out2 = model_new.forward_features(x2, make_2D=True)
    assert out2.shape == (1, 384, 8, 8)


@pytest.mark.skipif(nope_dir is None, reason="nope_dv2_vits14_reg.pth checkpoint missing")
def test_nope_wrapper_loading():
    model = WrapperRegistry.build("nope", device="cpu")
    model = model.eval()

    # Do inference on different sized tensors
    x1 = torch.randn(1, 3, 224, 224)
    out1 = model.forward_features(x1, make_2D=True)
    assert out1.shape == (1, 384, 16, 16)

    x2 = torch.randn(1, 3, 112, 112)
    out2 = model.forward_features(x2, make_2D=True)
    assert out2.shape == (1, 384, 8, 8)


@pytest.mark.skipif(alibi_dv3_dir is None, reason="alibi_dv3_ms.pth checkpoint missing")
def test_alibi_dv3_wrapper_loading():
    # Test old name alibi_dv3
    # model_old = get_model("alibi_dv3", alibi_dv3_dir, device="cpu", conf_path="models/dinov3")
    # assert isinstance(model_old, PretrainedViTWrapper)

    # Test new name alibi_dinov3_s
    model_new = get_model("alibi_dinov3_s+", alibi_dv3_dir, device="cpu", conf_path="models/dinov3")
    assert isinstance(model_new, PretrainedViTWrapper)

    # Do inference on different sized tensors
    x1 = torch.randn(1, 3, 224, 224)
    out1 = model_new.forward_features(x1, make_2D=True)
    assert out1.shape == (1, 384, 14, 14)

    x2 = torch.randn(1, 3, 112, 112)
    out2 = model_new.forward_features(x2, make_2D=True)
    assert out2.shape == (1, 384, 7, 7)


def test_simple_debias_wrappers():
    # Build a backbone offline with pretrained=False
    cfg = BackboneConfig(
        backbone_type="timm",
        model_arch="dinov2_s",
        pretrained=False,
    )
    vit = BackboneRegistry.build(cfg)
    device = "cpu"

    # 1. Test ChannelBlankedWrapper
    channels_to_blank = [0, 5, 10]
    cb_wrapper = ChannelBlankedWrapper(
        vit=vit,
        device=device,
        channels_to_blank=channels_to_blank,
    )
    cb_wrapper.eval()

    x = torch.randn(1, 3, 224, 224)
    # Test make_2D = True (4D output)
    out_2d = cb_wrapper.forward_features(x, make_2D=True)
    assert out_2d.shape == (1, 384, 16, 16)
    # Check that blanked channels are all zero
    for ch in channels_to_blank:
        assert torch.all(out_2d[:, ch] == 0.0)
    # Check that other channels are not all zero
    assert not torch.all(out_2d[:, 1] == 0.0)

    # Test make_2D = False (3D output)
    out_1d = cb_wrapper.forward_features(x, make_2D=False)
    assert out_1d.shape == (1, 384, 256)
    for ch in channels_to_blank:
        assert torch.all(out_1d[:, ch] == 0.0)
    assert not torch.all(out_1d[:, 1] == 0.0)

    # 2. Test TransformAverageWrapper
    ta_wrapper = TransformAverageWrapper(
        vit=vit,
        device=device,
    )
    ta_wrapper.eval()

    out_ta = ta_wrapper.forward_features(x, make_2D=True)
    assert out_ta.shape == (1, 384, 16, 16)

    # 3. Test DebiasedViTWrapper
    deb_wrapper = DebiasedViTWrapper(
        vit=vit,
        device=device,
    )
    deb_wrapper.eval()

    # Test make_2D = True
    out_deb_2d = deb_wrapper.forward_features(x, make_2D=True, svd_components=4)
    assert out_deb_2d.shape == (1, 384, 16, 16)

    # Test make_2D = False
    out_deb_1d = deb_wrapper.forward_features(x, make_2D=False, svd_components=4)
    assert out_deb_1d.shape == (1, 384, 256)

