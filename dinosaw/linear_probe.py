import numpy as np
from typing import Literal, TypedDict


from sklearn.linear_model import LinearRegression


RampTypes = Literal["lr", "ud", "diag", "radial", "raster", "random"]


def get_ramp(ramp_type: RampTypes, h: int, w: int) -> np.ndarray:
    """Generate a ramp mask in the specified direction."""
    if ramp_type == "lr":
        return np.tile(np.linspace(0, 1, w), (h, 1))
    elif ramp_type == "ud":
        return np.tile(np.linspace(0, 1, h), (w, 1)).T
    elif ramp_type == "diag":
        return np.linspace(0, 1, max(h, w))[:h, None] + np.linspace(0, 1, max(h, w))[:w]
    elif ramp_type == "radial":
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cy, cx = h // 2, w // 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / r.max()
        return 1 - r_norm
    elif ramp_type == "random":
        return np.random.rand(h, w)
    elif ramp_type == "raster":
        inds = np.arange(h * w)
        frac_inds = inds / (h * w)

        return frac_inds.reshape((h, w))
    else:
        raise ValueError("Direction must be 'lr', 'ud', or 'diag'.")


def gen_sample_mask(
    shape: tuple[int, int], ramp_type: RampTypes, step: int = 4, cutoff_frac: float = 1.0
) -> np.ndarray:
    """Generate a sampling mask for an array, taking every `step`-th element."""
    mask = np.zeros(shape, dtype=bool)

    if ramp_type == "radial":
        ramp_h, ramp_w = int(shape[0] * cutoff_frac), int(shape[1] * cutoff_frac)
        h, w = shape
        x0, x1 = int(np.ceil((w - ramp_w) / 2)), int(np.floor((w + ramp_w) / 2))
        y0, y1 = int(np.ceil((h - ramp_h) / 2)), int(np.floor((h + ramp_h) / 2))
    else:
        x0, x1 = 0, int(shape[1] * cutoff_frac)
        y0, y1 = 0, int(shape[0] * cutoff_frac)

    mask[y0:y1:step, x0:x1:step] = True
    return mask


def linear_probe_arr(feats: np.ndarray, target: np.ndarray, sample_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Train a linear regression model on sampled features to predict the target."""
    h, w, c = feats.shape
    X = feats[sample_mask, :]  # Shape: (num_samples, c)
    y = target[sample_mask]  # Shape: (num_samples,)

    try:
        model = LinearRegression()
        model.fit(X, y)

        # Predict on all features
        X_full = feats.reshape(-1, c)  # Shape: (h*w, c)
        y_pred = model.predict(X_full)  # Shape: (h*w,)

        score = model.score(X_full, target.flatten())  # R^2 score on training data
    except ValueError:
        y_pred = np.zeros(h * w)
        score = -1.0
    return y_pred.reshape(h, w), float(score)


def linear_probe_by_channel(
    feats: np.ndarray, target: np.ndarray, sample_mask: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    preds = np.zeros_like(feats)
    scores: list[float] = []
    _, _, c = feats.shape
    for i in range(c):
        ch_feat = feats[:, :, i : i + 1]
        pred, score = linear_probe_arr(ch_feat, target, sample_mask)
        preds[:, :, i] = pred
        scores.append(score)
    return preds, scores


class LinearProbeResult(TypedDict):
    stack_r_squared: float
    stack_pred: np.ndarray
    per_channel_scores: list[float] | None
    per_channel_preds: np.ndarray | None


def do_linear_probe(
    feats: np.ndarray, ramp_type: RampTypes, mask_step=4, mask_cutoff_frac=0.8, probe_by_channel: bool = False
) -> LinearProbeResult:
    h, w, c = feats.shape

    feats = np.nan_to_num(feats, nan=0.0)
    sample_mask = gen_sample_mask((h, w), ramp_type, step=mask_step, cutoff_frac=mask_cutoff_frac)
    ramp = get_ramp(ramp_type, h, w)
    stack_pred, stack_r_squared = linear_probe_arr(feats, ramp, sample_mask)

    per_channel_scores: list[float] | None = None
    per_channel_preds: np.ndarray | None = None
    if probe_by_channel:
        per_channel_preds, per_channel_scores = linear_probe_by_channel(feats, ramp, sample_mask)

    return {
        "stack_r_squared": stack_r_squared,
        "stack_pred": stack_pred,
        "per_channel_preds": per_channel_preds,
        "per_channel_scores": per_channel_scores,
    }
