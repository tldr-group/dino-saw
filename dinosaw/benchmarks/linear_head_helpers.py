import torch
import torch.nn as nn
from typing import Literal
from dinosaw.models.vit_wrapper import PretrainedViTWrapper, AlibiVitWrapper, MODEL_LIST
from dinosaw.datasets.benchmark_datasets import VOC_Dataset, DatasetADE_NEW
from dinosaw.utils import normalize
from dinosaw.datasets.vis_dataset import colorize
import numpy as np
import matplotlib.pyplot as plt

Model = Literal["Dv2", "NoPE", "ALiBi"]
Benchmark = Literal["VOC07", "VOC12", "ADE20K"]


def get_head(benchmark: Benchmark) -> nn.Sequential:
    match benchmark:
        case "VOC12":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
            )
        case "VOC07":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
            )
        case "ADE20K":
            return nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=150, kernel_size=1),
            )


class BenchmarkModel(torch.nn.Module):
    def __init__(self, model, device: torch.device, size: int, benchmark: str) -> None:
        super().__init__()
        self.size = size
        self.dino = model.eval()

        # freezing dino backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        self.head = get_head(benchmark).to(device)

    def forward(self, x):
        with torch.no_grad():
            lr_feats = self.dino.forward_features(x, make_2D=True)
        lr_pred = self.head(lr_feats)
        hr_pred = torch.nn.functional.interpolate(
            input=lr_pred, size=(self.size, self.size), mode="bilinear"
        )
        return hr_pred


def get_weights(model_type: Model, benchmark: Benchmark):
    match model_type:
        case "Dv2":
            match benchmark:
                case "VOC12":
                    weights = torch.load(
                        "../../trained_linear_models/VOC12_Dv2.pth", map_location="cpu"
                    )
                case "VOC07":
                    weights = torch.load(
                        "../../trained_linear_models/VOC07_Dv2.pth", map_location="cpu"
                    )
                case "ADE20K":
                    weights = torch.load(
                        "../../trained_linear_models/ADE20K_Dv2.pth", map_location="cpu"
                    )
        case "NoPE":
            match benchmark:
                case "VOC12":
                    weights = torch.load(
                        "../../trained_linear_models/VOC12_NoPE.pth", map_location="cpu"
                    )
                case "VOC07":
                    weights = torch.load(
                        "../../trained_linear_models/VOC07_NoPE.pth", map_location="cpu"
                    )
                case "ADE20K":
                    weights = torch.load(
                        "../../trained_linear_models/ADE20K_NoPE.pth",
                        map_location="cpu",
                    )
        case "ALiBi":
            match benchmark:
                case "VOC12":
                    weights = torch.load(
                        "../../trained_linear_models/VOC12_ALiBi.pth",
                        map_location="cpu",
                    )
                case "VOC07":
                    weights = torch.load(
                        "../../trained_linear_models/VOC07_ALiBi.pth",
                        map_location="cpu",
                    )
                case "ADE20K":
                    weights = torch.load(
                        "../../trained_linear_models/ADE20K_ALiBi.pth",
                        map_location="cpu",
                    )

    return weights


def get_val_ds(benchmark: Benchmark):
    match benchmark:
        case "VOC12":
            return VOC_Dataset(
                base_path="../../Datasets/VOC",
                mode="val",
            )
        case "VOC07":
            return VOC_Dataset(
                base_path="../../Datasets/VOC07/VOCdevkit/VOC2007", mode="val"
            )
        case "ADE20K":
            return DatasetADE_NEW(
                base_path="../../Datasets/ADEChallengeData2016",
                mode="val",
                asses_linear_probe=True,
            )


def get_lin_model(model_type: Model, benchmark: Benchmark, device: torch.device | str):
    if model_type == "Dv2":
        model = WrapperRegistry.build("dinov2_s", device=device)
    elif model_type == "NoPE":
        model = WrapperRegistry.build("nope", device=device)
    elif model_type == "ALiBi":
        model = WrapperRegistry.build("alibi_dv2", device=device)
    else:
        raise ValueError(f"Invalid model_type {model_type}")

    lin_model = BenchmarkModel(
        model=model, device=device, size=518, benchmark=benchmark
    )
    lin_model.load_state_dict(get_weights(model_type, benchmark))
    lin_model.eval()
    return lin_model


def hide_axes(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    if hasattr(ax, "set_zticks"):
        ax.set_zticks([])


def plot_samples(
    benchmark: Benchmark, num_samples=5, device: torch.device = "cpu", fs=30
):
    ds = get_val_ds(benchmark)
    Dv2 = get_lin_model("Dv2", benchmark, device)
    NoPE = get_lin_model("NoPE", benchmark, device)
    ALiBi = get_lin_model("ALiBi", benchmark, device)
    sample_indices = [
        int(rand) for rand in np.random.random(num_samples) * (len(ds) - 1)
    ]

    print(sample_indices)

    fig, axes = plt.subplots(
        5,
        ncols=len(sample_indices),
        figsize=(num_samples * 5, len(sample_indices) * 5),
    )

    for idx, sample_index in enumerate(sample_indices):
        axes[0][idx].imshow(normalize(ds[sample_index][0].permute(1, 2, 0)))
        axes[1][idx].imshow(
            colorize(
                Dv2(ds[sample_index][0].unsqueeze(0).to(device))
                .argmax(dim=1)
                .squeeze()
                .cpu()
            )
        )
        axes[2][idx].imshow(
            colorize(
                NoPE(ds[sample_index][0].unsqueeze(0).to(device))
                .argmax(dim=1)
                .squeeze()
                .cpu()
            )
        )
        axes[3][idx].imshow(
            colorize(
                ALiBi(ds[sample_index][0].unsqueeze(0).to(device))
                .argmax(dim=1)
                .squeeze()
                .cpu()
            )
        )
        axes[4][idx].imshow(colorize(ds[sample_index][1]))

    for ax in axes.ravel():
        hide_axes(ax)

    # axes[0][0].set_ylabel("input", fontsize=30, weight=500)
    axes[1][0].set_ylabel("DINOv2", fontsize=fs, weight=500)
    axes[2][0].set_ylabel("NoPE", fontsize=fs, weight=500)
    axes[3][0].set_ylabel("ALiBi", fontsize=fs, weight=700)
    axes[4][0].set_ylabel("ground\ntruth", fontsize=fs, weight=500)

    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.1)
    return fig
