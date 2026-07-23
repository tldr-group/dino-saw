import torch
import numpy as np

from PVW import PretrainedViTWrapper

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision import transforms

from PIL import Image
from PIL.ImageColor import getcolor
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Literal


from random import seed as rseed


NormType = Literal["minmax", "std", None]
norm_dict = {
    "minmax": MinMaxScaler(feature_range=(0, 1), clip=True, copy=False),
    "std": StandardScaler(copy=False),
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().squeeze().numpy()
    return arr


def get_features(
    model: PretrainedViTWrapper,
    pil_img: Image.Image,
    channel_last: bool = False,
) -> np.ndarray:
    with torch.no_grad():
        emb = model.forward_features(pil_img, make_2D=True)
    emb_np = to_numpy(emb)
    if channel_last:
        emb_np = np.transpose(emb_np, (1, 2, 0))
    return emb_np


# ========================= PCA STUFF =========================
def flatten(x: torch.Tensor | np.ndarray) -> np.ndarray:
    y: np.ndarray
    if type(x) is torch.Tensor:
        y = to_numpy(x)
    else:
        y = x  # type: ignore
    c, h, w = y.shape
    y = y.reshape((c, h * w))
    y = y.T
    return y


def do_pca(
    arr: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    # arr in shape (n_samples, n_features)
    if n_samples > -1:
        inds = np.arange(arr.shape[0])
        sample_inds = np.random.choice(inds, n_samples)
        train_data = arr[sample_inds]
    else:
        train_data = arr

    if pre_norm is not None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[pre_norm]
        scaler.fit_transform(arr)

    pca = PCA(n_components=n_components)

    pca.fit_transform(train_data)
    projection = pca.transform(arr)

    if post_norm is not None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[post_norm]
        scaler.fit_transform(projection)
    return projection


def do_2D_pca(
    arr_2D: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    c, h, w = arr_2D.shape
    flat = flatten(arr_2D)
    proj = do_pca(flat, n_components, n_samples, pre_norm, post_norm)
    proj_2d = proj.reshape((h, w, n_components))
    return proj_2d


# ========================= INPUT TRANSFORMS =========================


MU, SIGMA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INV_MU = (-MU[0] / SIGMA[0], -MU[1] / SIGMA[1], -MU[2] / SIGMA[2])
INV_SIGMA = (1 / SIGMA[0], 1 / SIGMA[1], 1 / SIGMA[2])

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MU, std=SIGMA),
    ]
)

unnormalize = transforms.Normalize(
    mean=INV_MU,
    std=INV_SIGMA,
)


def closest_crop(h: int, w: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose([transforms.CenterCrop((new_h, new_w)), to_norm_tensor])
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
            ]
        )
    return transform


def closest_resize(h: int, w: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose([transforms.Resize((new_h, new_w)), to_norm_tensor])
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((new_h, new_w)),
            ]
        )
    return transform


def closest_resize_crop(L: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    assert L % patch_size == 0, "L must be divisible by patch_size"
    if to_tensor:
        transform = transforms.Compose(
            [
                transforms.Resize(L),
                transforms.CenterCrop((L, L)),
                to_norm_tensor,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(L),
                transforms.CenterCrop((L, L)),
            ]
        )
    return transform


def get_shortest_side_resize_dims(img_h: int, img_w: int, min_l: int) -> tuple[int, int]:
    if min(img_w, img_h) > min_l:
        sf = min(img_w / min_l, img_h / min_l)
    else:
        sf = max(min_l / img_w, min_l / img_h)
    return (int(max((img_h * sf), min_l)), int(max(img_w * sf, min_l)))


def resize_crop(resize_dims: tuple[int, int], crop_dims: tuple[int, int]) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dims),
            transforms.CenterCrop(crop_dims),
            to_norm_tensor,
        ]
    )
    return transform


def load_image(
    path: str,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
    device_str: str = "cuda:0",
) -> tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor: torch.Tensor = convert_image(image, transform, to_gpu, to_half, batch, device_str=device_str)
    transformed_img = to_img(unnormalize(tensor.squeeze(0)))
    return tensor, transformed_img


def convert_image(
    img: Image.Image | torch.Tensor,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
    device_str: str = "cuda:0",
) -> torch.Tensor:
    tensor: torch.Tensor = transform(img)  # type: ignore
    if to_half:
        tensor = tensor.to(torch.float16)
    if to_gpu:
        tensor = tensor.to(device_str)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor


def plot_losses(train_loss: list[float], val_loss: list[float], out_path: str) -> None:
    epochs = np.arange(len(train_loss))
    plt.semilogy(epochs, train_loss, lw=2, label="train")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def normalize(t_tens):
    return (t_tens - t_tens.min()) / (t_tens.max() - t_tens.min())


def seed_everything(seed: int):
    rseed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% Plotting


def add_custom_font(font_folder: str, font_name: str = "Grotesk") -> None:
    try:
        font_paths = (f"{font_folder}/{font_name}.ttf", f"{font_folder}/{font_name}-Bold.ttf")
        for font_path in font_paths:
            font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_paths[0])

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = prop.get_name()
        plt.rcParams["font.serif"] = prop.get_name()
    except Exception as e:
        print(f"Can't load custom font: {e} ")


COLOURS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
COLORS = [[v / 255.0 for v in getcolor(c, "RGB")] for c in COLOURS]


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_frame_on(False)


def apply_labels_as_overlay(labels: np.ndarray, img: Image.Image, colors: list, alpha: float = 1.0) -> Image.Image:
    labels_unsqueezed = np.expand_dims(labels, -1)

    overlay = label2rgb(labels, colors=colors[1:], kind="overlay", bg_label=0, image_alpha=1, alpha=alpha)
    out = np.where(labels_unsqueezed, overlay * 255, np.array(img)).astype(np.uint8)
    img_with_labels = Image.fromarray(out)
    return img_with_labels
