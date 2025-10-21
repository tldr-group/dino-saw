import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal
from torchvision import transforms
from PIL import Image

NormType = Literal["minmax", "std", None]
norm_dict = {
    "minmax": MinMaxScaler(feature_range=(0, 1), clip=True, copy=False),
    "std": StandardScaler(copy=False),
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if len(arr.shape) == 4:
        arr = arr[0]
    return arr

# ========================= PCA STUFF =========================
def flatten(x: torch.Tensor | np.ndarray) -> np.ndarray:
    y: np.ndarray
    if type(x) == torch.Tensor:
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

    if pre_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[pre_norm]
        scaler.fit_transform(arr)
        # arr = scaler.transform(arr)
        # train_data = scaler.transform(train_data)

    pca = PCA(n_components=n_components)

    train_proj = pca.fit_transform(train_data)
    projection = pca.transform(arr)

    if post_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[post_norm]
        scaler.fit_transform(projection)
        # projection = scaler.transform(projection)
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
) -> tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor: torch.Tensor = convert_image(image, transform, to_gpu, to_half, batch)
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