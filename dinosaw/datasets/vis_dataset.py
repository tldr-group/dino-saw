import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from dinosaw.datasets.train_student_dataset import EmbeddingDataset
from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper
from dinosaw.utils import to_numpy, do_2D_pca


def get_arrs_from_batch(
    img: torch.Tensor,
    lr_feats: torch.Tensor,
    pred_homog_feats: torch.Tensor | None,
) -> list[list[np.ndarray]]:
    b, _, _, _ = lr_feats.shape

    arrs: list[list[np.ndarray]] = []
    for i in range(b):
        img_tensor, lr_feat_tensor, pred_homog_tensor = (
            img[i],
            lr_feats[i],
            pred_homog_feats[i],
        )
        img_arr = to_numpy(img_tensor.permute((1, 2, 0)))

        out_2D_arrs: list[np.ndarray] = [img_arr]
        tensors = (
            (lr_feat_tensor, pred_homog_tensor) if isinstance(pred_homog_feats, torch.Tensor) else (lr_feat_tensor)
        )
        for i, d in enumerate(tensors):
            feat_arr = to_numpy(d)
            out_2D = do_2D_pca(feat_arr, 3, post_norm="minmax")
            out_2D_arrs.append(out_2D)
        arrs.append(out_2D_arrs)
    return arrs


def unnorm(x: torch.Tensor) -> torch.Tensor:
    return TF.normalize(x, [-0.485, -0.456, -0.406], [1 / 0.229, 1 / 0.224, 1 / 0.225])


def visualise(
    img: torch.Tensor | Image.Image,
    lr_feats: torch.Tensor,
    pred_homog_feats: torch.Tensor | None,
    out_path: str,
) -> None:
    # b, c, h, w = hr_feats.shape
    n_rows = 3 if isinstance(pred_homog_feats, torch.Tensor) else 2
    arrs = get_arrs_from_batch(
        img,
        lr_feats,
        pred_homog_feats,
    )
    fig, axs = plt.subplots(nrows=n_rows, ncols=len(arrs))
    fig.set_size_inches(32, 4.4)
    for i, arr in enumerate(arrs):
        for j, sub_arr in enumerate(arr):
            if len(arrs) == 1:
                axs[j].imshow(sub_arr)
                axs[j].set_axis_off()
            else:
                axs[j, i].imshow(sub_arr)
                axs[j, i].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    DEVICE = "cuda:1"
    dv2 = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
    dv2 = dv2.eval()

    ds = EmbeddingDataset("Dataset/IN_reduced_224", "val")
    dl = DataLoader(
        ds,
        20,
        True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    next(iter(dl))
    img, lr = next(iter(dl))

    dv2_feats = dv2.forward_features(img.to(DEVICE), make_2D=True)
    print(img.shape, lr.shape, dv2_feats.shape)
    visualise(unnorm(img).to(torch.uint8), lr, dv2_feats, "tmp/batch_vis.png")
