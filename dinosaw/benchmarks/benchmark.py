import torch
import numpy as np
from torch import nn, optim

from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from torchmetrics.classification import MulticlassJaccardIndex

from os import makedirs, environ
from shutil import rmtree
from datetime import datetime

from dinosaw.datasets.vis_dataset import visualise_segmentation
from dinosaw.datasets.benchmark_datasets import VOC_Dataset, ADE20KDataset
from dinosaw.models.alibi import AlibiSlopeType
from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper
from dinosaw.utils import seed_everything, closest_resize
import time

from typing import Literal
from dataclasses import dataclass

environ["QT_QPA_PLATFORM"] = "offscreen"

Benchmarks = Literal["VOC12", "ADE20K"]
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
    benchmark: Benchmarks = "VOC12"

    # training
    n_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    optim: Optims = "AdamW"
    loss_type: Losses = "CE"

    save_per: int = 2


def get_head(benchmark: Benchmarks) -> nn.Sequential:
    match benchmark:
        case "VOC12":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
            )
        case "ADE20K":
            return nn.Sequential(
                nn.SyncBatchNorm(num_features=384),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels=384, out_channels=151, kernel_size=1),
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
                reduction=reduction, ignore_index=255 if benchmark == "VOC12" else 0
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
    match model_type:
        case "base":
            model = PretrainedViTWrapper(
                vit_model_type,
                stride=stride,
                add_flash_attn=False,
                device=device,
                checkpoint_path=chk_path,
            )
        case "plus_alibi":
            model = AlibiVitWrapper(
                vit_model_type,
                stride=stride,
                add_flash_attn=False,
                device=device,
                slope_type=alibi_slope_type,
                normalize=norm_alibi,
                wrap=wrap_alibi,
                checkpoint_path=chk_path,
            )
        case _:
            raise Exception(f"Unsupported model type {model_type}")

    if freeze_abs_pos_emb or zero_pos_emb:
        # assert model.model.pos_embed is not None
        if "dv3" in vit_model_type or "dinov3" in vit_model_type:
            pass
        else:
            model.model.pos_embed.requires_grad = False  # freeze pos embedding

    if zero_pos_emb:
        # assert model.model.pos_embed is not None
        if "dv3" in vit_model_type or "dinov3" in vit_model_type:
            print("dinov3, zeroing rope_embed")
            model.model.rope_embed = None
        else:
            model.model.pos_embed.data.zero_()

    if model_type == "plus_alibi" and n_epochs_warmup <= 0:
        model.set_alibi_enabled(True)
    elif model_type == "plus_alibi" and n_epochs_warmup > 0:
        model.set_alibi_enabled(False)

    if existing_checkpoint is not None:
        print(f"Loading existing checkpoint from {existing_checkpoint}")
        model.load_state_dict(
            torch.load(existing_checkpoint, map_location=device, weights_only=True)
        )

    return model


class BenchmarkModel(nn.Module):
    def __init__(
        self, model: PretrainedViTWrapper, cfg: Config, device: torch.device
    ) -> None:
        super().__init__()

        self.dino = model.eval()

        # freezing dino backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        self.head = get_head(cfg.benchmark).to(device)

    def forward(self, x):
        lr_feats = self.dino.forward_features(x, make_2D=True)
        lr_pred = self.head(lr_feats)
        hr_pred = nn.functional.interpolate(
            input=lr_pred, size=(518, 518), mode="bilinear"
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
    x, y_true = batch
    x = x.to(device, non_blocking=True)
    y_true = y_true.to(device, non_blocking=True)
    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
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


IMG_L = 518
EXPR_PATH = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M')}"
SEED = 1025
N_VIS = 32
seed_everything(SEED)


cfg = Config(
    experiment_name="dinov2_ADE20K",
    benchmark="ADE20K",
    existing_checkpoint="",
    model_type="base",
    batch_size=32,
    lr=1e-3,
    save_per=1,
)
print(cfg)

try:
    rmtree(EXPR_PATH)
except FileNotFoundError:
    pass
makedirs(EXPR_PATH, exist_ok=True)

writer = SummaryWriter(EXPR_PATH)
writer.add_text("desc", cfg.experiment_name)


DEVICE = "cuda:1"
CACHE = False
tr = closest_resize(IMG_L, IMG_L, 14)

match cfg.benchmark:
    case "VOC12":
        train_ds = VOC_Dataset(
            base_path="/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
            mode="train",
        )
        val_ds = VOC_Dataset(
            base_path="/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
            mode="val",
        )
    case "ADE20K":
        train_ds = ADE20KDataset(
            base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/ADE20K",
            mode="train",
        )
        val_ds = ADE20KDataset(
            base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/ADE20K",
            mode="val",
        )

print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")

train_dl = DataLoader(
    train_ds,
    cfg.batch_size,
    True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_dl = DataLoader(
    val_ds,
    cfg.batch_size,
    True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


# model = AlibiVitWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
# model.set_alibi_enabled(False)

if cfg.model_type == "base":
    model = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
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


bench_model = BenchmarkModel(model, cfg, device=DEVICE)

optimizer = get_optim(cfg.optim, bench_model, cfg.lr)
loss_fn = get_loss(cfg.loss_type, cfg.benchmark)

train_losses: list[float] = []
val_losses: list[float] = []
best_val_loss = 1e6

mean_iou = MulticlassJaccardIndex(
    num_classes=21 if cfg.benchmark == "VOC12" else 151,
    ignore_index=255 if cfg.benchmark == "VOC12" else 0,
    average="macro",
).to(DEVICE)

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

    train_loss = train_loss_sum.item() / len(train_dl)
    train_miou = train_miou_sum.item() / len(train_dl)

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
        x, y_true = batch
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
