import numpy as np
from typing import Literal, TypedDict


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


RampTypes = Literal["lr", "ud", "diag", "radial", "raster", "lr+ud", "random"]
Regressor = Literal["linear", "ridge"]


def get_ramp(ramp_type: RampTypes, h: int, w: int) -> np.ndarray:
    """Generate a ramp mask in the specified direction."""
    if ramp_type == "lr":
        ramp = np.tile(np.linspace(0, 1, w), (h, 1))
        return np.expand_dims(ramp, axis=-1)
    elif ramp_type == "ud":
        ramp = np.tile(np.linspace(0, 1, h), (w, 1)).T
        return np.expand_dims(ramp, axis=-1)
    elif ramp_type == "diag":
        ramp = np.linspace(0, 1, max(h, w))[:h, None] + np.linspace(0, 1, max(h, w))[:w]
        return np.expand_dims(ramp, axis=-1)
    elif ramp_type == "radial":
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cy, cx = h // 2, w // 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / r.max()
        ramp = 1 - r_norm
        return np.expand_dims(ramp, axis=-1)
    elif ramp_type == "random":
        ramp = np.random.rand(h, w)
        return np.expand_dims(ramp, axis=-1)
    elif ramp_type == "raster":
        inds = np.arange(h * w)
        frac_inds = inds / (h * w)

        return frac_inds.reshape((h, w, 1))
    elif ramp_type == "lr+ud":
        lr_ramp = np.tile(np.linspace(0, 1, w), (h, 1))
        ud_ramp = np.tile(np.linspace(0, 1, h), (w, 1)).T
        return np.stack((lr_ramp, ud_ramp), axis=-1)
    else:
        raise ValueError("Direction must be 'lr', 'ud', or 'diag'.")


def gen_sample_mask(
    shape: tuple[int, int], ramp_type: RampTypes, step: int = 4, cutoff_frac: float = 1.0, random_mask: bool = False
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

    if random_mask:
        n_samples = ((y1 - y0) * (x1 - x0)) // (step * step)
        inds = np.arange((y1 - y0) * (x1 - x0))
        chosen = np.random.choice(inds, size=n_samples, replace=False)
        flat_mask = np.zeros((y1 - y0) * (x1 - x0), dtype=bool)
        flat_mask[chosen] = True
        mask_section = flat_mask.reshape((y1 - y0, x1 - x0))
        mask[y0:y1, x0:x1] = mask_section
    else:
        mask[y0:y1:step, x0:x1:step] = True
    return mask


def linear_probe_arr(
    feats: np.ndarray,
    target: np.ndarray,
    sample_mask: np.ndarray,
    scale_feats: bool = True,
    regressor: Regressor = "ridge",
) -> tuple[np.ndarray, float]:
    """Train a linear regression model on sampled features to predict the target."""
    h, w, c = feats.shape
    X = feats[sample_mask, :]  # Shape: (num_samples, c)
    y = target[sample_mask, :]  # Shape: (num_samples,)

    scaler = None
    if scale_feats:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    model = LinearRegression() if regressor == "linear" else Ridge()

    try:
        model.fit(X, y)

        # Predict on all features
        X_full = feats.reshape(-1, c)  # Shape: (h*w, c)
        if scale_feats:
            assert scaler is not None
            X_full = scaler.transform(X_full)
        y_pred = model.predict(X_full)  # Shape: (h*w,)

        score = model.score(X_full, target.reshape((h * w, -1)))  # R^2 score on training data
    except ValueError:
        y_pred = np.zeros(h * w)
        score = -1.0
    return y_pred.reshape(h, w, -1), float(score)


def linear_probe_by_channel(
    feats: np.ndarray,
    target: np.ndarray,
    sample_mask: np.ndarray,
    scale_feats: bool = True,
    regressor: Regressor = "ridge",
) -> tuple[list[np.ndarray], list[float]]:
    preds: list[np.ndarray] = []
    scores: list[float] = []
    _, _, c = feats.shape
    for i in range(c):
        ch_feat = feats[:, :, i : i + 1]
        pred, score = linear_probe_arr(ch_feat, target, sample_mask, scale_feats=scale_feats, regressor=regressor)
        preds.append(pred)
        scores.append(score)
    return preds, scores


class LinearProbeResult(TypedDict):
    stack_r_squared: float
    stack_pred: np.ndarray
    per_channel_scores: list[float] | None
    per_channel_preds: list[np.ndarray] | None


def do_linear_probe(
    feats: np.ndarray,
    ramp_type: RampTypes,
    mask_step=4,
    mask_cutoff_frac=0.8,
    probe_by_channel: bool = False,
    random_mask: bool = False,
    scale: bool = True,
    regressor: Regressor = "ridge",
) -> LinearProbeResult:
    h, w, c = feats.shape

    feats = np.nan_to_num(feats, nan=0.0)
    sample_mask = gen_sample_mask(
        (h, w), ramp_type, step=mask_step, cutoff_frac=mask_cutoff_frac, random_mask=random_mask
    )
    ramp = get_ramp(ramp_type, h, w)
    stack_pred, stack_r_squared = linear_probe_arr(feats, ramp, sample_mask, scale_feats=scale, regressor=regressor)

    per_channel_scores: list[float] | None = None
    per_channel_preds: list[np.ndarray] | None = None
    if probe_by_channel:
        per_channel_preds, per_channel_scores = linear_probe_by_channel(
            feats, ramp, sample_mask, scale_feats=scale, regressor=regressor
        )

    return {
        "stack_r_squared": stack_r_squared,
        "stack_pred": stack_pred,
        "per_channel_preds": per_channel_preds,
        "per_channel_scores": per_channel_scores,
    }
