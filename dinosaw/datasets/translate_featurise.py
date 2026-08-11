from functools import cache
from math import ceil

import torch

from dinosaw.utils import closest_crop, do_2D_pca, load_image, to_numpy
from dinosaw.wrappers import MODEL_LIST, PretrainedViTWrapper


def get_shifts(h: int, w: int, step: int, mult: int) -> list[tuple[int, int]]:
    """Get all shifts for image (h,w) and $step i.e for (h,w)=28 & step=14 -> [(0,0), (0,14), ..., (0,28), ...]"""
    shifts: list[tuple[int, int]] = []
    for y in range(0, h, step * mult):
        for x in range(0, w, step * mult):
            shifts.append((y, x))
    return shifts


@torch.no_grad()
def get_batch(img: torch.Tensor, shifts: list[tuple[int, int]]) -> torch.Tensor:
    """Apply list of shifts to img tensor with torch.roll, return (B,C,H,W) tensor where B=N_shifts. NB: wrap BCs"""
    shifted_imgs = []
    for shift in shifts:
        y_shift, x_shift = shift
        shifted_img = torch.roll(img, shifts=(y_shift, x_shift), dims=(2, 3))
        shifted_imgs.append(shifted_img)
    return torch.cat(shifted_imgs, dim=0)


@torch.no_grad()
def get_feats(
    full_img_batch: torch.Tensor, vit_wrapper: PretrainedViTWrapper, max_batch_size: int, device: str = "cuda"
) -> torch.Tensor:
    """Get features of (B,C,H,W) batch of translated images, splitting into minibatches of size max_batch_size for memory"""
    b, _, _, _ = full_img_batch.shape
    n_minibatches = ceil(b / max_batch_size)
    out_feats: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(n_minibatches):
            minibatch = full_img_batch[i * max_batch_size : (i + 1) * max_batch_size]
            minibatch_feats = vit_wrapper.forward_features(minibatch.to(device), make_2D=True).cpu()
            out_feats.append(minibatch_feats)
    return torch.cat(out_feats, dim=0)


def invert_shifts_and_average(
    feats_batch: torch.Tensor, shifts: list[tuple[int, int]], patch_size: int
) -> torch.Tensor:
    """Invert shifts in feature space (image space // patch_size), and average. Return (1,C,N_th,N_tw) tensor."""
    b, _, _, _ = feats_batch.shape
    inverted: list[torch.Tensor] = []
    for i, shift in enumerate(shifts):
        y_shift, x_shift = shift
        inverted_feat = torch.roll(feats_batch[i], shifts=(-y_shift // patch_size, -x_shift // patch_size), dims=(1, 2))
        inverted.append(inverted_feat)
    stacked = torch.stack(inverted, dim=0)
    return torch.sum(stacked, dim=0, keepdim=True, dtype=torch.float64) / b


@cache
@torch.no_grad()
def translate_featurise(
    img: torch.Tensor,
    vit_wrapper: PretrainedViTWrapper,
    step: int = 14,
    mult: int = 1,
    max_batch_size: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    _, _, h, w = img.shape
    img = img.cpu()
    shifts = get_shifts(h, w, step, mult)
    img_batch = get_batch(img, shifts)
    # img_batch = img_batch.to(device)
    feats_batch = get_feats(img_batch, vit_wrapper, max_batch_size, device)
    translated_feats = invert_shifts_and_average(feats_batch, shifts, vit_wrapper.patch_size)
    return translated_feats


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from dinosaw.utils import gen_sample_mask, get_ramp, linear_probe

    torch.cuda.empty_cache()
    DEVICE = "cuda:1"

    tr = closest_crop(518, 518, 14)
    img_tensor, _ = load_image("images/default_image_518.png", tr, True, True, device_str=DEVICE)
    img_tensor = torch.zeros_like(img_tensor).half()

    dv2 = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=True, device=DEVICE)
    translated_feats = translate_featurise(img_tensor, dv2, step=14, mult=1, max_batch_size=16, device=DEVICE)
    translated_feats = translated_feats.to("cpu")

    translated_feats_np = to_numpy(translated_feats)
    c, h, w = translated_feats_np.shape

    reduced = do_2D_pca(translated_feats_np, 3, post_norm="minmax")

    c, n_th, n_tw = translated_feats_np.shape

    ramp = get_ramp("lr", n_th, n_tw)
    sample_mask = gen_sample_mask((n_th, n_tw), step=5, cutoff_frac=0.7)
    pred, r_sq = linear_probe(translated_feats_np, ramp, sample_mask)

    plt.imsave("tmp/translate_avg_pca.png", reduced)
    plt.imsave("tmp/homog_probe.png", pred)
    print(r_sq)

    print(translated_feats.shape)
