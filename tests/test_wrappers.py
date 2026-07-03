import os
import torch
import pytest
from dinosaw.helpers import get_model
from PVW import WrapperRegistry
from dinosaw.wrappers import DebiasedViTWrapper, DenoisingViTWrapper, PretrainedViTWrapper


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
    chk_path = os.path.join(nope_dir, "nope_dv2_vits14_reg.pth")
    model = WrapperRegistry.build("nope", device="cpu")
    state_dict = torch.load(chk_path, weights_only=True, map_location="cpu")
    state_dict = {
        k.replace("model.", "vit.", 1) if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
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
    model_old = get_model("alibi_dv3", alibi_dv3_dir, device="cpu", conf_path="models/dinov3")
    assert isinstance(model_old, PretrainedViTWrapper)

    # Test new name alibi_dinov3_s
    model_new = get_model("alibi_dinov3_s", alibi_dv3_dir, device="cpu", conf_path="models/dinov3")
    assert isinstance(model_new, PretrainedViTWrapper)

    # Do inference on different sized tensors
    x1 = torch.randn(1, 3, 224, 224)
    out1 = model_new.forward_features(x1, make_2D=True)
    assert out1.shape == (1, 384, 14, 14)

    x2 = torch.randn(1, 3, 112, 112)
    out2 = model_new.forward_features(x2, make_2D=True)
    assert out2.shape == (1, 384, 7, 7)
