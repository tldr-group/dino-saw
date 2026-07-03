import os
import tempfile
import torch
import pytest
from PIL import Image
from dinosaw.datasets.joint_embed_dataset import JointEmbeddingDataset
from PVW import WrapperRegistry


@pytest.fixture
def temp_img_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 10, i * 20, i * 30))
            img.save(os.path.join(tmpdir, f"dummy_{i}.jpg"))
        yield tmpdir


def test_joint_embedding_dataset(temp_img_dir):
    device = "cpu"
    model = WrapperRegistry.build("dinov2_s", device=device, pretrained=False)

    dataset = JointEmbeddingDataset(
        embed_model=model,
        base_path=temp_img_dir,
        split="val",
        device=device,
        squeeze_batch_dim_from_image=True,
    )

    assert len(dataset) == 3
    img, target_emb, tr = dataset[0]

    # Image shape should be [3, 224, 224]
    assert img.shape == (3, 224, 224)
    # Target embedding shape should be [embed_dim, H_patch, W_patch] (384, 16, 16)
    assert target_emb.shape == (384, 16, 16)
    assert callable(tr)
