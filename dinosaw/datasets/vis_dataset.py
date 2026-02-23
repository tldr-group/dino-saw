import torch
import torchvision.transforms.functional as TF  # type: ignore
from torch.utils.data import DataLoader
import numpy as np

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from dinosaw.utils import to_numpy, do_2D_pca


def pascal_colormap():
    cmap = torch.zeros(256, 3, dtype=torch.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = torch.tensor([r, g, b], dtype=torch.uint8)
    return cmap


def colorize(pred_mask):  # pred_mask: (H,W) int
    CMAP = pascal_colormap()
    # returns (3,H,W)
    rgb = CMAP[pred_mask]  # (H,W,3)
    return rgb  # H,W,3


def get_arrs_from_batch(
    img: torch.Tensor,
    lr_feats: torch.Tensor,
    pred_homog_feats: torch.Tensor | None,
) -> list[list[np.ndarray]]:
    b, _, _, _ = img.shape

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
            (lr_feat_tensor, pred_homog_tensor)
            if isinstance(pred_homog_feats, torch.Tensor)
            else (lr_feat_tensor)
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
    out_path: str | None = None,
) -> Image.Image:
    n_rows = 3 if isinstance(pred_homog_feats, torch.Tensor) else 3
    img_unnormed = unnorm(img)
    img_rgb = (img_unnormed - img_unnormed.min()) / (
        img_unnormed.max() - img_unnormed.min()
    )
    arrs = get_arrs_from_batch(
        img_rgb,
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
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    if out_path:
        pil_img.save(out_path)
    return pil_img


def get_seg_arrs_from_batch(
    img: torch.Tensor,
    mask: torch.Tensor,
    pred_homog_feats: torch.Tensor | None,
) -> list[list[np.ndarray]]:
    b, _, _, _ = img.shape
    arrs: list[list[np.ndarray]] = []
    for i in range(b):
        img_tensor, mask_tensor, pred_homog_tensor = (
            img[i],
            mask[i],
            pred_homog_feats[i],
        )
        img_arr = to_numpy(img_tensor.permute((1, 2, 0)))

        out_2D_arrs: list[np.ndarray] = [img_arr]
        out_2D_arrs.append(colorize(to_numpy(mask_tensor)))

        if isinstance(pred_homog_tensor, torch.Tensor):
            out_2D = colorize(to_numpy(pred_homog_tensor).argmax(axis=0))
            out_2D_arrs.append(out_2D)
            out_2D_arrs.append(
                do_2D_pca(to_numpy(pred_homog_tensor), 3, post_norm="minmax")
            )

        arrs.append(out_2D_arrs)
    return arrs


def visualise_segmentation(
    img: torch.Tensor | Image.Image,
    mask: torch.Tensor,
    pred_feats: torch.Tensor | None,
    out_path: str | None,
):
    n_rows = 4 if isinstance(pred_feats, torch.Tensor) else 2
    img_unnormed = unnorm(img)
    img_rgb = (img_unnormed - img_unnormed.min()) / (
        img_unnormed.max() - img_unnormed.min()
    )

    arrs = get_seg_arrs_from_batch(
        img_rgb,
        mask,
        pred_feats,
    )

    fig, axs = plt.subplots(nrows=n_rows, ncols=len(arrs))
    fig.set_size_inches(32, 4.4 / 3 * 4)
    for i, arr in enumerate(arrs):
        for j, sub_arr in enumerate(arr):
            # print(f"{sub_arr.shape=}")
            if len(arrs) == 1:
                axs[j].imshow(sub_arr)
                axs[j].set_axis_off()
            else:
                axs[j, i].imshow(sub_arr)
                axs[j, i].set_axis_off()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    if out_path:
        pil_img.save(out_path)
    return pil_img


if __name__ == "__main__":
    from dinosaw.datasets.train_student_dataset import HomogenizedEmbeddingDataset
    from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper

    DEVICE = "cuda:0"
    dv2 = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
    dv2 = dv2.eval()

    ds = HomogenizedEmbeddingDataset(
        "data/IN_reduced_224", "val", store_in_memory=False, norm_feats=True
    )
    dl = DataLoader(
        ds,
        32,
        True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    next(iter(dl))
    img, lr = next(iter(dl))

    dv2_feats = dv2.forward_features(img.to(DEVICE), make_2D=True)
    print(img.shape, lr.shape, dv2_feats.shape)
    visualise(img, lr, dv2_feats, "tmp/batch_vis.png")
