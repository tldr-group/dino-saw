import argparse
from dataclasses import dataclass, field
import json
from os import listdir, makedirs
from pathlib import Path
import sys
from typing import Any, TypedDict

import numpy as np
from PIL import Image
from skimage.transform import resize
import torch

from dinosaw.linear_probe import (
    LinearProbeResult,
    RampTypes,
    Regressor,
    do_linear_probe,
)
from dinosaw.utils import get_features
from dinosaw.wrappers import ModelTypes, get_models


"""
python notebooks/paper_figures_pdf/run_linear_probe_make_table.py path/to/config.json


"""


# ==============================================================================
# Configuration & Metadata
# ==============================================================================

class ModelMeta(TypedDict, total=False):
    name: str
    d: str
    pe: str
    supervised: bool
    bold: bool


DEFAULT_MODEL_METADATA: dict[ModelTypes, ModelMeta] = {
    "dino_s": {"name": "DINO", "d": "S", "pe": "Learned", "supervised": False, "bold": False},
    "dino_b": {"name": "DINO", "d": "B", "pe": "Learned", "supervised": False, "bold": False},
    "dinov2_s": {"name": "DINOv2", "d": "S", "pe": "Learned", "supervised": False, "bold": False},
    "dinov2_b": {"name": "DINOv2", "d": "B", "pe": "Learned", "supervised": False, "bold": False},
    "dinov3_s+": {"name": "DINOv3", "d": "S+", "pe": "RoPE", "supervised": False, "bold": False},
    "dinov3_b": {"name": "DINOv3", "d": "B", "pe": "RoPE", "supervised": False, "bold": False},
    "mae_b": {"name": "ViT (MAE)", "d": "B", "pe": "Learned", "supervised": False, "bold": False},
    "sam_b": {"name": "SAM", "d": "256", "pe": "Learned", "supervised": True, "bold": False},
    "in1k_b": {"name": "ViT (IN1k)", "d": "B", "pe": "Learned", "supervised": True, "bold": False},
    "deit_s": {"name": "DEiT", "d": "B", "pe": "Learned", "supervised": True, "bold": False},
    "clip_b": {"name": "CLIP", "d": "B", "pe": "Learned", "supervised": True, "bold": False},
    "eva02_b": {"name": "EVA02", "d": "B", "pe": "RoPE", "supervised": True, "bold": False},
    "cb_dinov2_s": {"name": "DINOv2 (CB)", "d": "S", "pe": "Learned", "supervised": False, "bold": False},
    "dvt_dinov2_s": {"name": "DVT", "d": "S", "pe": "Learned", "supervised": False, "bold": False},
    "alibi_dinov2_s": {"name": "ALiBi-Dv2", "d": "S", "pe": "ALiBI", "supervised": False, "bold": True},
    "alibi_coco_dinov2_s": {"name": "ALiBi-Dv2", "d": "S", "pe": "ALiBI", "supervised": False, "bold": True},
    "alibi_dinov3_s+": {"name": "ALiBi-Dv3", "d": "S+", "pe": "ALiBI", "supervised": False, "bold": True},
    "nope_dinov2_s": {"name": "NoPE", "d": "S", "pe": "None", "supervised": False, "bold": False},
    "sinusoid_dinov2_s": {"name": "Sinusoid", "d": "S", "pe": "Sinusoid", "supervised": False, "bold": False},
    "sinusoid_dinov2_s_cb": {"name": "Sinusoid (CB)", "d": "S", "pe": "Sinusoid", "supervised": False, "bold": False},
}


@dataclass
class DatasetConfig:
    name: str  # Short name used for table header, e.g. 'micro', 'texture', 'noise'
    folder: str  # Path to dataset images folder
    replace_with_random_noise: bool = False
    noise_size: tuple[int, int] = (518, 518)


@dataclass
class ProbeConfig:
    # Execution & environment
    seed: int = 100002
    device: str = "cuda:0"
    checkpoint_dir: str = "models/checkpoints"
    conf_dir: str = "models/dinov3"

    # Models and datasets
    models: list[ModelTypes] = field(
        default_factory=lambda: [
            "dino_s",
            "dino_b",
            "dinov2_s",
            "dinov2_b",
            "dinov3_s+",
            "mae_b",
            "sam_b",
            "in1k_b",
            "deit_s",
            "clip_b",
            "eva02_b",
        ]
    )
    datasets: list[DatasetConfig] = field(
        default_factory=lambda: [
            DatasetConfig(name="micro", folder="notebooks/paper_figures_pdf/data/linear_probe/homog_micros"),
            DatasetConfig(name="texture", folder="notebooks/paper_figures_pdf/data/linear_probe/texture_ds"),
            DatasetConfig(
                name="noise",
                folder="notebooks/paper_figures_pdf/data/linear_probe/homog_micros",
                replace_with_random_noise=True,
            ),
        ]
    )

    # Linear probe parameters
    ramp: RampTypes = "lr+ud"
    mask_cutoff_frac: float = 1.0
    step: int = 6
    random_mask: bool = True
    n_repeats: int = 6
    regressor: Regressor = "ridge"
    probe_by_channel: bool = False
    scale: bool = True

    # Score persistence (detaching generation & rendering)
    run_generation: bool = True
    run_rendering: bool = True
    scores_path: str | None = "notebooks/paper_figures_pdf/saved/linear_probe_scores.json"

    # Table rendering options
    table_output_path: str | None = "notebooks/paper_figures_pdf/saved/linear_probe_table.tex"
    show_std: bool = False
    show_supervised_col: bool = True
    use_multirow: bool = True
    red_threshold: float = 0.4
    caption: str = (
        r"$(x, y)$ linear probe $R^2$ scores for a set of existing models which produce patch-level features. "
        r"Supervised models - including those with an SSL backbone - have lower $R^2$ scores. "
        r"Scores are reported per-dataset: `micro' for micrographs, `texture' for DTD \cite{DTD} texture images "
        r"and `noise' for random noise."
    )
    label: str = "tab:linear_probe_literature"
    model_metadata: dict[str, ModelMeta] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeConfig":
        data = dict(data)
        if "datasets" in data:
            datasets = []
            for d in data["datasets"]:
                if isinstance(d, DatasetConfig):
                    datasets.append(d)
                elif isinstance(d, dict):
                    d_copy = dict(d)
                    if "noise_size" in d_copy:
                        d_copy["noise_size"] = tuple(d_copy["noise_size"])
                    datasets.append(DatasetConfig(**d_copy))
            data["datasets"] = datasets
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "ProbeConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        elif suffix == ".toml":
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.from_dict(data)
        elif suffix in (".yaml", ".yml"):
            import yaml  # type: ignore

            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        elif suffix == ".py":
            import importlib.util

            spec = importlib.util.spec_from_file_location("user_probe_config", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load python config from {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "config"):
                cfg_obj = mod.config
                return cfg_obj if isinstance(cfg_obj, cls) else cls.from_dict(cfg_obj)
            elif hasattr(mod, "CONFIG"):
                cfg_obj = mod.CONFIG
                return cfg_obj if isinstance(cfg_obj, cls) else cls.from_dict(cfg_obj)
            else:
                raise ValueError(f"Python config file {path} must define `config` or `CONFIG`.")
        else:
            raise ValueError(f"Unsupported config file extension '{suffix}'. Use .json, .toml, .yaml, or .py.")


# ==============================================================================
# 1. Score Generation & Disk I/O
# ==============================================================================

class ScoreSummary(TypedDict):
    mean: float
    std: float
    all_scores: list[float]


# Structure: scores[dataset_name][model_name] -> ScoreSummary
ScoresDict = dict[str, dict[str, ScoreSummary]]


def average_results(
    results: list[LinearProbeResult],
) -> tuple[np.ndarray | None, np.ndarray | None, float, float, np.ndarray]:
    """Aggregate linear probe results across images and repeats (from T01_linear_probe.ipynb)."""
    scores_arr = np.array([res["stack_r_squared"] for res in results])
    mean_score = float(np.mean(scores_arr, axis=0))
    std_score = float(np.std(scores_arr, axis=0))

    n_pred_dims = results[0]["stack_pred"].shape[-1]
    mean_pred = np.zeros((34, 34, n_pred_dims))

    for res in results:
        pred = resize(res["stack_pred"], mean_pred.shape, order=1)
        mean_pred += pred / len(results)

    if results[0]["per_channel_scores"] is None:
        return None, None, mean_score, std_score, mean_pred

    channel_scores_arr = np.array([res["per_channel_scores"] for res in results])
    mean_channel_scores = np.mean(channel_scores_arr, axis=0)
    std_channel_scores = np.std(channel_scores_arr, axis=0)

    return mean_channel_scores, std_channel_scores, mean_score, std_score, mean_pred


def generate_scores(cfg: ProbeConfig) -> ScoresDict:
    """Run feature extraction and linear probing for all configured models and datasets."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    enabled_models: tuple[ModelTypes, ...] = tuple(cfg.models)
    models = get_models(
        enabled_models,
        cfg.device,
        True,
        cfg.checkpoint_dir,
        cfg.conf_dir,
    )

    all_scores: ScoresDict = {}

    for ds_cfg in cfg.datasets:
        print(f"\n--- Processing dataset: {ds_cfg.name} ({ds_cfg.folder}) ---")
        # Reset seed per dataset so noise generation / random masks are reproducible
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        image_files = sorted(
            f
            for f in listdir(ds_cfg.folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        n_imgs = len(image_files)

        features: dict[ModelTypes, list[np.ndarray]] = {k: [] for k in models.keys()}
        for img_file in image_files:
            img_path = f"{ds_cfg.folder}/{img_file}"
            img = Image.open(img_path).convert("RGB")
            if ds_cfg.replace_with_random_noise:
                h, w = ds_cfg.noise_size
                noise_arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                img = Image.fromarray(noise_arr).convert("RGB")

            for model_name, model in models.items():
                feats = get_features(model, img, channel_last=True)
                features[model_name].append(feats)

        ramps_to_results: dict[ModelTypes, list[LinearProbeResult]] = {m: [] for m in enabled_models}
        for model_name in enabled_models:
            for i in range(n_imgs):
                feats = features[model_name][i]
                for _ in range(cfg.n_repeats):
                    result = do_linear_probe(
                        feats,
                        cfg.ramp,
                        probe_by_channel=cfg.probe_by_channel,
                        mask_step=cfg.step,
                        mask_cutoff_frac=cfg.mask_cutoff_frac,
                        random_mask=cfg.random_mask,
                        scale=cfg.scale,
                        regressor=cfg.regressor,
                    )
                    ramps_to_results[model_name].append(result)

        all_scores[ds_cfg.name] = {}
        for model_name in enabled_models:
            _, _, mean_score, std_score, _ = average_results(ramps_to_results[model_name])
            raw_scores = [float(r["stack_r_squared"]) for r in ramps_to_results[model_name]]
            all_scores[ds_cfg.name][model_name] = {
                "mean": mean_score,
                "std": std_score,
                "all_scores": raw_scores,
            }
            print(f"{model_name} ({ds_cfg.name}): {mean_score:.2f} ± {std_score:.2f}")

    return all_scores


def save_scores(scores: ScoresDict, path: str | Path) -> None:
    """Dump generated scores to disk as JSON."""
    path = Path(path)
    if path.parent:
        makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {path}")


def load_scores(path: str | Path) -> ScoresDict:
    """Load previously generated scores from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ==============================================================================
# 2. LaTeX Table Rendering
# ==============================================================================

def _format_text(text: str, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold else text


def _format_score_cell(
    mean: float,
    std: float,
    *,
    red_threshold: float = 0.4,
    show_std: bool = False,
    bold: bool = False,
) -> str:
    color = "red!25" if mean > red_threshold else "green!25"
    if show_std:
        val_str = f"{mean:.2f} $\\pm$ {std:.2f}"
    else:
        val_str = f"{mean:.2f}"

    if bold:
        val_str = rf"\textbf{{{val_str}}}"

    return rf"\cellcolor{{{color}}} {val_str}"


def render_latex_table(scores: ScoresDict, cfg: ProbeConfig) -> str:
    """Render a LaTeX table (like example_table_1.tex or example_table_2.tex) from scores."""
    dataset_names = [ds.name for ds in cfg.datasets]
    models = cfg.models

    # Resolve metadata for each model
    resolved_meta: dict[str, ModelMeta] = {}
    for m in models:
        base_meta = dict(
            DEFAULT_MODEL_METADATA.get(
                m,
                {"name": m, "d": "", "pe": "", "supervised": False, "bold": False},
            )
        )
        if m in cfg.model_metadata:
            base_meta.update(cfg.model_metadata[m])
        resolved_meta[m] = base_meta  # type: ignore

    # Pre-calculate multirow spans for consecutive models sharing the same display name
    name_spans: list[int] = [1] * len(models)
    if cfg.use_multirow:
        i = 0
        while i < len(models):
            j = i + 1
            while (
                j < len(models)
                and resolved_meta[models[j]].get("name") == resolved_meta[models[i]].get("name")
                and resolved_meta[models[j]].get("bold", False) == resolved_meta[models[i]].get("bold", False)
            ):
                j += 1
            span = j - i
            if span > 1:
                name_spans[i] = span
                for k in range(i + 1, j):
                    name_spans[k] = 0
            i = j

    # Column alignment & header
    n_meta_cols = 4 if cfg.show_supervised_col else 3
    col_spec = "l " + " ".join(["c"] * (n_meta_cols - 1 + len(dataset_names)))

    header_cols = ["Model", r"\textit{d}", "PE"]
    if cfg.show_supervised_col:
        header_cols.append("Supervised")
    for ds_name in dataset_names:
        header_cols.append(rf"$R^2_{{\text{{{ds_name}}}}}$")
    header_line = "        " + " & ".join(header_cols) + r" \\"

    # Build rows
    row_lines: list[str] = []
    for idx, m in enumerate(models):
        meta = resolved_meta[m]
        is_bold = bool(meta.get("bold", False))
        disp_name = _format_text(meta.get("name", m), is_bold)
        d_str = _format_text(meta.get("d", ""), is_bold)
        pe_str = _format_text(meta.get("pe", ""), is_bold)

        span = name_spans[idx]
        if span > 1:
            model_col = rf"\multirow{{{span}}}{{*}}{{{disp_name}}}"
        elif span == 1:
            model_col = disp_name
        else:
            model_col = ""

        row_cells = [model_col, d_str, pe_str]
        if cfg.show_supervised_col:
            sup_str = r"$\checkmark$" if meta.get("supervised", False) else r"$\times$"
            row_cells.append(sup_str)

        for ds_name in dataset_names:
            ds_scores = scores.get(ds_name, {})
            if m in ds_scores:
                mean_val = ds_scores[m]["mean"]
                std_val = ds_scores[m]["std"]
                cell_str = _format_score_cell(
                    mean_val,
                    std_val,
                    red_threshold=cfg.red_threshold,
                    show_std=cfg.show_std,
                    bold=is_bold,
                )
            else:
                cell_str = "-"
            row_cells.append(cell_str)

        indent = "            " if span == 0 else "        "
        row_lines.append(indent + " & ".join(row_cells) + r" \\")

    lines = [
        r"\begin{table*}",
        r"    \centering",
        rf"    \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        header_line,
        r"    \midrule",
        *row_lines,
        r"    \bottomrule",
        r"    \end{tabular}",
        rf"    \caption{{{cfg.caption}}}",
        rf"    \label{{{cfg.label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run linear probe evaluation and/or render LaTeX results table from a single config file."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration file (.json, .toml, .yaml, or .py).",
    )
    args = parser.parse_args()

    cfg = ProbeConfig.from_file(args.config)

    scores: ScoresDict | None = None

    if cfg.run_generation:
        scores = generate_scores(cfg)
        if cfg.scores_path:
            save_scores(scores, cfg.scores_path)

    if cfg.run_rendering:
        if scores is None:
            if not cfg.scores_path:
                print("Error: `run_generation` is False, but no `scores_path` was provided to load scores from.", file=sys.stderr)
                sys.exit(1)
            scores = load_scores(cfg.scores_path)

        latex_table = render_latex_table(scores, cfg)
        print("\n" + latex_table + "\n")

        if cfg.table_output_path:
            out_path = Path(cfg.table_output_path)
            if out_path.parent:
                makedirs(out_path.parent, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(latex_table + "\n")
            print(f"Saved LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
