import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal

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