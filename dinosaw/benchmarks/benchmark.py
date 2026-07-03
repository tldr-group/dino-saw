import torch
import numpy as np
from torch import nn, optim
import geobench

from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.v2 as v2

from torchmetrics.classification import MulticlassJaccardIndex

from os import makedirs, environ
from shutil import rmtree
from datetime import datetime

from dinosaw.datasets.vis_dataset import visualise_segmentation
from dinosaw.datasets.benchmark_datasets import (
    VOC_Dataset,
    VOC07_Dataset,
    ADE20KDataset,
    GeoBenchDataset,
    DatasetADE_NEW,
    GF7,
    Satellites,
)
import dinosaw.utils as utils
from dinosaw.wrappers.DPT_head import DPTHead
from dinosaw.alibi_logic import AlibiSlopeType
from dinosaw.wrappers import MODEL_LIST, PretrainedViTWrapper
from dinosaw.utils import seed_everything, closest_resize
import time

from typing import Literal
from dataclasses import dataclass, field

environ["QT_QPA_PLATFORM"] = "offscreen"

Benchmarks = Literal["VOC12", "ADE20K", "m-SA-crop-type", "GF7", "VOC07"]
Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["MSE", "MAE", "cosine", "CE"]
ModelType = Literal["base", "plus_alibi"]


@dataclass
class Config:
    # experiment
    experiment_name: str = "default"

    # model
    model_type: ModelType = "plus_alibi"
    alibi_slope_type: AlibiSlopeType = "constant"
    norm_alibi: bool = True
    wrap_alibi: bool = True
    n_epochs_warmup: int = -1
    freeze_pos_emb: bool = True
    zero_pos_emb: bool = True
    existing_checkpoint: str | None = None
    vit_model_type: str = MODEL_LIST[1]
    stride: int = 14
    chk_path: str | None = None
    dino_chk_path: str | None = None  # Dv3?

    # benchmark specific
    benchmark: Benchmarks | Satellites = "VOC12"

    # training
    n_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    optim: Optims = "AdamW"
    loss_type: Losses = "CE"
    channels_to_blank: list[int] | None = None
    save_per: int = 2


def get_head(benchmark: Benchmarks) -> nn.Sequential | DPTHead:
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
                # nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=150, kernel_size=1),
            )
        case "LandSat":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=134, kernel_size=1),
            )
        case "m-cashew-plant":
            # return nn.Sequential(
            #     nn.SyncBatchNorm(num_features=384),
            #     nn.Dropout2d(p=0.1),
            #     nn.Conv2d(in_channels=384, out_channels=6, kernel_size=1),
            # )

            return DPTHead(n_output_channels=6)
        case "m-SA-crop-type":
            # return nn.Sequential(
            #     nn.SyncBatchNorm(num_features=384),
            #     nn.Dropout2d(p=0.1),
            #     nn.Conv2d(in_channels=384, out_channels=9, kernel_size=1),
            # )
            return DPTHead(n_output_channels=9)
        case "GF7":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=2, kernel_size=1),
            )
        case _:
            raise Exception(f"benchmark '{benchmark}' not supported!")


def get_loss(loss_type: Losses, benchmark: Benchmarks, reduction: str = "mean"):
    match loss_type:
        case "MSE":
            return nn.MSELoss(reduction=reduction)
        case "MAE":
            return nn.L1Loss(reduction=reduction)
        case "cosine":
            cosine_sim = nn.CosineSimilarity(dim=1)

            def cosine_sim_wrapper(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                return (1.0 - cosine_sim(x1, x2)).mean()

            return cosine_sim_wrapper
        case "CE":
            return nn.CrossEntropyLoss(
                reduction=reduction,
                ignore_index=255
                if (benchmark == "VOC12" or benchmark == "VOC07")
                else -1,
            )
        case _:
            raise Exception(f"Unsupported loss {loss_type}")


def get_optim(optim_type: Optims, model: nn.Module, lr: float) -> optim.Optimizer:
    match optim_type:
        case "Adam":
            return torch.optim.Adam(model.parameters(), lr)
        case "AdamW":
            return torch.optim.AdamW(model.parameters(), lr)
        case "SGD":
            return torch.optim.SGD(model.parameters(), lr)
        case _:
            raise Exception(f"Unsupported optimizer {optim_type}")


def get_model(
    model_type: ModelType,
    alibi_slope_type: AlibiSlopeType,
    norm_alibi: bool,
    wrap_alibi: bool,
    n_epochs_warmup: int,
    freeze_abs_pos_emb: bool,
    zero_pos_emb: bool,
    device: torch.device,
    existing_checkpoint: str | None = None,
    vit_model_type: str = MODEL_LIST[1],
    stride: int = 14,
    chk_path: str | None = None,
) -> nn.Module:
    is_dinov3 = "dv3" in vit_model_type or "dinov3" in vit_model_type

    from PVW.types import ARCHS_TO_TIMM_IDS, ARCHS_TO_LOCAL_IDS
    from PVW.factory import BackboneConfig, BackboneRegistry
    from functools import partial
    from dinosaw.wrappers.alibi import add_alibi

    arch_name = None
    for k, v in ARCHS_TO_TIMM_IDS.items():
        if v == vit_model_type:
            arch_name = k
            break
    if arch_name is None:
        for k, v in ARCHS_TO_LOCAL_IDS.items():
            if v == vit_model_type:
                arch_name = k
                break
    if arch_name is None:
        arch_name = vit_model_type

    backbone_type = "torch_hub" if is_dinov3 else "timm"
    cfg = BackboneConfig(
        backbone_type=backbone_type,
        model_arch=arch_name,  # type: ignore
        pretrained=True,
        checkpoint_path=chk_path,
        model_conf_path="dinov3" if is_dinov3 else None,
        stride=(stride, stride),
    )

    if model_type == "plus_alibi":
        mod = partial(
            add_alibi,
            slope_type=alibi_slope_type,
            n_reg_tokens=4,
            metric="euclidean",
            normalize=norm_alibi,
            wrap=wrap_alibi,
            add_cls=True,
            jitter_mag=0.0,
        )
        cfg.modifications.append(mod)

    vit = BackboneRegistry.build(cfg)
    model = PretrainedViTWrapper(vit=vit, device=device)
    model.set_alibi_enabled = lambda enabled: None

    if freeze_abs_pos_emb or zero_pos_emb:
        if is_dinov3:
            pass
        else:
            if hasattr(model.vit, "pos_embed") and model.vit.pos_embed is not None:
                model.vit.pos_embed.requires_grad = False  # freeze pos embedding

    if zero_pos_emb:
        if is_dinov3:
            print("dinov3, zeroing rope_embed")
            if hasattr(model.vit, "rope_embed"):
                model.vit.rope_embed = None
        else:
            if hasattr(model.vit, "pos_embed") and model.vit.pos_embed is not None:
                model.vit.pos_embed.data.zero_()

    if existing_checkpoint is not None:
        print(f"Loading existing checkpoint from {existing_checkpoint}")
        model.load_state_dict(
            torch.load(existing_checkpoint, map_location=device, weights_only=True)
        )

    # model.model.blocks[11].attn.activate_matrix = False
    # model.model.blocks[10].attn.activate_matrix = False
    # model.model.blocks[9].attn.activate_matrix = False
    # model.model.blocks[8].attn.activate_matrix = False

    return model


class BenchmarkModel(nn.Module):
    def __init__(
        self, model: PretrainedViTWrapper, cfg: Config, device: torch.device, size: int
    ) -> None:
        super().__init__()

        self.channels_to_blank = cfg.channels_to_blank
        self.size = size
        self.dino = model.eval()

        # freezing dino backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        self.head = get_head(cfg.benchmark).to(device)

    def forward(self, x):
        if type(self.head) is not DPTHead:
            with torch.no_grad():
                lr_feats = self.dino.forward_features(x, make_2D=True)
                if self.channels_to_blank is not None:
                    lr_feats[:, self.channels_to_blank, :, :] = 0
            lr_pred = self.head(lr_feats)
            hr_pred = nn.functional.interpolate(
                input=lr_pred, size=(self.size, self.size), mode="bilinear"
            )
        else:
            print(type(self.head))
            with torch.no_grad():
                lr_feats = self.dino.get_intermediate_layers(
                    x, [5, 7, 9, 11], return_prefix_tokens=True, reshape=True
                )
                # if self.channels_to_blank is not None:
                #     lr_feats[:, self.channels_to_blank, :, :] = 0
            lr_pred = self.head(lr_feats)
            hr_pred = nn.functional.interpolate(
                input=lr_pred, size=(self.size, self.size), mode="bilinear"
            )
        return hr_pred


def feed_batch_get_loss(
    model: BenchmarkModel,
    optimizer: optim.Optimizer,
    loss_fn,
    metric_fn,
    batch: torch.Tensor,
    training: bool,
    device: str = "cuda",
) -> tuple[float, float]:
    x, y_true = (
        batch
        # if cfg.benchmark != "LandSat"
        # else (
        #     nn.functional.interpolate(batch["image"], (518, 518)),
        #     nn.functional.interpolate(
        #         batch["mask"].unsqueeze(1).float(), (518, 518), mode="nearest-exact"
        #     )
        #     .squeeze()
        #     .long(),
        # )
    )
    # print(f"{x.shape=}, {y_true.shape=}")
    x = x.to(device, non_blocking=True)
    y_true = y_true.to(device, non_blocking=True)
    if training:
        model.head.train()
        optimizer.zero_grad()
    else:
        model.head.eval()
    with torch.set_grad_enabled(training):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        metric = metric_fn(y_pred, y_true)
        if training:
            loss.backward()
            optimizer.step()
    # x = x.to("cpu")
    # y_true = y_true.to("cpu")
    # y_pred = y_pred.to("cpu")
    return loss.detach(), metric.detach()


IMG_L = 518  # * 2
SEED = 1025
N_VIS = 32
seed_everything(SEED)


cfg = Config(
    # experiment_name="TEST_VOC07_alibi_coco_big_norm_fixed_val_loader",
    experiment_name="TEST_VOC07_Dv2_norm_fixed_val_loader",
    benchmark="VOC07",
    # existing_checkpoint="../experiments/20260221_2258_test_learned_slope_with_jitter_last_4_layers_no_alibi_100_224_5e-4_10_518_bs8_5e-5/best_model.pth",
    # existing_checkpoint="/home/pawlo/Arbeit/positional_bias/dino-saw/trained_models/alibi_dv2_cb_l_j_mo.pth",
    # existing_checkpoint="../../trained_models/nope_dv2_vits14_reg.pth",
    # existing_checkpoint="../../trained_models/alibi_const_coco_cb_vits14_reg_slow_ms.pth",
    # existing_checkpoint="../../trained_models/nope_coco_dv2_vits14_reg_ms.pth",
    model_type="base",
    # optim="SGD",
    # wrap_alibi=True,
    # alibi_slope_type="constant",
    batch_size=64,
    n_epochs=50,
    lr=1e-3,
    save_per=1,
)
print(cfg)

EXPR_PATH = (
    f"experiments/{datetime.now().strftime('%Y%m%d_%H%M')}_{cfg.experiment_name}"
)
try:
    rmtree(EXPR_PATH)
except FileNotFoundError:
    pass
makedirs(EXPR_PATH, exist_ok=True)

writer = SummaryWriter(EXPR_PATH)
writer.add_text("desc", cfg.experiment_name)


DEVICE = "cuda:0"
CACHE = False
tr = closest_resize(IMG_L, IMG_L, 14)

match cfg.benchmark:
    case "VOC12":
        train_ds = VOC_Dataset(
            base_path="../../Datasets/VOC",
            mode="train",
        )
        val_ds = VOC_Dataset(
            base_path="../../Datasets/VOC",
            mode="val",
        )
    case "VOC07":
        train_ds = VOC_Dataset(
            base_path="../../Datasets/VOC07/VOCdevkit/VOC2007", mode="train"
        )
        val_ds = VOC_Dataset(
            base_path="../../Datasets/VOC07/VOCdevkit/VOC2007", mode="train"
        )
    case "ADE20K":
        # train_ds = ADE20KDataset(
        #     base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/ADE20K",
        #     mode="train",
        # )
        train_ds = DatasetADE_NEW(
            base_path="../../Datasets/ADEChallengeData2016",
            mode="train",
            max_sample=1000,
        )
        # val_ds = ADE20KDataset(
        #     base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/ADE20K",
        #     mode="val",
        # )
        val_ds = DatasetADE_NEW(
            base_path="../../Datasets/ADEChallengeData2016", mode="val"
        )
    case "m-cashew-plant":
        train_ds = GeoBenchDataset("m-cashew-plant", mode="train", size=IMG_L)
        val_ds = GeoBenchDataset("m-cashew-plant", mode="valid", size=IMG_L)
    case "m-SA-crop-type":
        train_ds = GeoBenchDataset("m-SA-crop-type", mode="train", size=IMG_L)
        val_ds = GeoBenchDataset("m-SA-crop-type", mode="valid", size=IMG_L)
    case "GF7":
        train_ds = GF7(
            "../../Datasets/GF-7 Building (3Bands)", mode="train", size=IMG_L
        )
        val_ds = GF7("../../Datasets/GF-7 Building (3Bands)", mode="val", size=IMG_L)

print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")

train_dl = DataLoader(
    train_ds,
    cfg.batch_size,
    True,
    drop_last=True,
    # num_workers=3,
    # pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=4,
)
val_dl = DataLoader(
    val_ds,
    cfg.batch_size,
    False,
    drop_last=True,
    # num_workers=3,
    # pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=4,
)

# model = AlibiVitWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
# model.set_alibi_enabled(False)

if cfg.model_type == "base":
    model = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
    if cfg.existing_checkpoint is not None:
        model.load_state_dict(
            torch.load(cfg.existing_checkpoint, map_location=DEVICE, weights_only=True)
        )
else:
    model = get_model(
        cfg.model_type,
        cfg.alibi_slope_type,
        cfg.norm_alibi,
        cfg.wrap_alibi,
        cfg.n_epochs_warmup,
        cfg.freeze_pos_emb,
        cfg.zero_pos_emb,
        DEVICE,
        cfg.existing_checkpoint,
        cfg.vit_model_type,
        cfg.stride,
        cfg.dino_chk_path,
    )


bench_model = BenchmarkModel(model, cfg, device=DEVICE, size=IMG_L)

optimizer = get_optim(cfg.optim, bench_model, cfg.lr)
loss_fn = get_loss(cfg.loss_type, cfg.benchmark)

train_losses: list[float] = []
val_losses: list[float] = []
best_val_loss = 1e6

match cfg.benchmark:
    case "VOC12":
        mean_iou = MulticlassJaccardIndex(
            num_classes=21,
            ignore_index=255,
            average="macro",
        ).to(DEVICE)
    case "VOC07":
        mean_iou = MulticlassJaccardIndex(
            num_classes=21, average="macro", ignore_index=255
        ).to(DEVICE)
    case "ADE20K":
        mean_iou = MulticlassJaccardIndex(
            num_classes=150,
            ignore_index=-1,
            average="macro",
        ).to(DEVICE)
    case "LandSat":
        mean_iou = MulticlassJaccardIndex(
            num_classes=134,
            average="macro",
        ).to(DEVICE)
    case "m-cashew-plant":
        mean_iou = MulticlassJaccardIndex(
            num_classes=6, average="macro", ignore_index=-1
        ).to(DEVICE)
    case "m-SA-crop-type":
        mean_iou = MulticlassJaccardIndex(
            num_classes=9, average="macro", ignore_index=-1
        ).to(DEVICE)
    case "GF7":
        mean_iou = MulticlassJaccardIndex(num_classes=2, average="macro").to(DEVICE)


# TODO:
# - pca vis for sqircle & dog (2 different res) for base dv2, trained and cleaned
# - consider test dl for zero pos enc
# - consider complilation
# - consider LR scheduling

for epoch in range(cfg.n_epochs):
    train_loss_sum, train_miou_sum = 0.0, 0.0
    batch: torch.Tensor
    N_batches = len(train_dl)

    for i, batch in enumerate(train_dl):
        loss, miou = feed_batch_get_loss(
            bench_model,
            optimizer,
            loss_fn,
            mean_iou,
            batch,
            training=True,
            device=DEVICE,
        )
        train_loss_sum += loss
        train_miou_sum += miou
        if i % 50 == 0:
            print(f"Train batch {i}/{N_batches}")

    train_loss = train_loss_sum / len(train_dl)
    train_miou = train_miou_sum / len(train_dl)

    val_loss_sum, val_miou_sum = 0.0, 0.0
    batch: torch.Tensor
    # for batch in [next(iter(val_dl))][:1]:
    for batch in val_dl:
        loss, miou = feed_batch_get_loss(
            bench_model,
            optimizer,
            loss_fn,
            mean_iou,
            batch,
            training=False,
            device=DEVICE,
        )
        val_loss_sum += loss
        val_miou_sum += miou

    val_loss = val_loss_sum / len(val_dl)
    val_miou = val_miou_sum / len(val_dl)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(bench_model.state_dict(), f"{EXPR_PATH}/best_model.pth")

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("miou/train", train_miou, epoch)
    writer.add_scalar("miou/val", val_miou, epoch)

    print(
        f"Epoch {epoch:04d}/{cfg.n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    if epoch % cfg.save_per == 0:
        x, y_true = (
            batch
            # if cfg.benchmark != "LandSat"
            # else (
            #     nn.functional.interpolate(batch["image"], (518, 518)),
            #     nn.functional.interpolate(
            #         batch["mask"].unsqueeze(1).float(), (518, 518)
            #     )
            #     .squeeze()
            #     .long(),
            # )
        )
        x = x[:N_VIS].to(DEVICE)
        y_true = y_true[:N_VIS].to(DEVICE)
        with torch.no_grad():
            y_pred = bench_model(x)

        vis_img = visualise_segmentation(x, y_true, y_pred, "tmp/val_vis.png")
        vis_img_arr = np.array(vis_img).transpose(2, 0, 1)
        writer.add_image("vis/val_batch", vis_img_arr, epoch)

        x = x.to("cpu")
        y_true = y_true.to("cpu")
        y_pred = y_pred.to("cpu")
