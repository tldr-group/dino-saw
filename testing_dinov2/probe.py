from typing import Literal
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils import to_numpy



def get_ramp(dir: Literal['lr', 'ud', 'diag', 'radial'], h, w) -> np.ndarray:
    """Generate a ramp mask in the specified direction."""
    if dir == 'lr':
        return np.tile(np.linspace(0, 1, w), (h, 1))
    elif dir == 'ud':
        return np.tile(np.linspace(0, 1, h), (w, 1)).T
    elif dir == 'diag':
        return np.linspace(0, 1, max(h, w))[:h, None] + np.linspace(0, 1, max(h, w))[:w]
    elif dir == 'radial':
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        cy, cx = h // 2, w // 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / r.max()
        return  1 - r_norm
    else:
        raise ValueError("Direction must be 'lr', 'ud', or 'diag'.")


def gen_sample_mask(shape: tuple[int, int], step: int = 4, cutoff_frac: float=1.0) -> np.ndarray:
    """Generate a sampling mask for an array, taking every `step`-th element."""
    cutoff_y, cutoff_x = int(shape[0] * cutoff_frac), int(shape[1] * cutoff_frac)
    mask = np.zeros(shape, dtype=bool)
    mask[0:cutoff_y:step, 0:cutoff_x:step] = True
    return mask


def linear_probe(feats: np.ndarray, target: np.ndarray, sample_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Train a linear regression model on sampled features to predict the target."""
    c, h, w = feats.shape
    X = feats[:, sample_mask].T  # Shape: (num_samples, c)
    y = target[sample_mask]       # Shape: (num_samples,)

    model = LinearRegression()
    model.fit(X, y)

    # Predict on all features
    X_full = feats.reshape(c, -1).T  # Shape: (h*w, c)
    y_pred = model.predict(X_full)    # Shape: (h*w,)

    score = model.score(X_full, target.flatten())  # R^2 score on training data
    return y_pred.reshape(h, w), float(score)



def probe(input_preds: list, remove_channels: list, titles: list, ramp='diag'):
    input_preds_, remove_channels_, titles_ = input_preds.copy(), remove_channels.copy(), titles.copy()
    N_COLS = len(input_preds_)+1
    WIDTH=4
    input_np = []
    for inp, remove in zip(input_preds_, remove_channels_):
        if remove:
            inp[:, [47, 113, 117, 359], :, :] = 0
        input_np.append(to_numpy(inp))
    
    c, h, w = input_np[0].shape
    sample_mask = gen_sample_mask((h, w), step=6, cutoff_frac=0.8)
    ramp = get_ramp(ramp, h=h, w=w)

    input_np.insert(0, ramp)
    titles_.insert(0, "Ramp")
    
    fig, axs = plt.subplots(1, N_COLS, figsize=(WIDTH*N_COLS, WIDTH))
    for arr, ax, title in zip(input_np, axs, titles_):
        if title != 'Ramp':
            print(arr.shape)
            res, score = linear_probe(arr, ramp, sample_mask)
            title += f'\n (R²: {score:.3f})'
        else:
            res = arr
            title += f'\n'
        ax.imshow(res)
        ax.axis('off')
        ax.set_title(title)


        