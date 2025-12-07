import torch
from torch import nn, optim
import numpy as np

from torch.utils.data import DataLoader

from os import makedirs
from shutil import rmtree
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

from dinosaw.datasets.vis_dataset import visualise
from dinosaw.datasets.train_student_dataset import HomogenizedEmbeddingDataset
from dinosaw.models.alibi import AlibiSlopeType
from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper
from dinosaw.utils import seed_everything

from dataclasses import dataclass
from typing import Literal

Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["MSE", "MAE"]
ModelType = Literal["base", "plus_alibi"]


@dataclass
class Config:
    experiment_name: str = "default_experiment"

    model_type: ModelType = "base"
    alibi_slope_type: AlibiSlopeType = "constant"
    norm_alibi: bool = True
    wrap_alibi: bool = True
    freeze_abs_pos_emb: bool = True
    zero_pos_emb: bool = False

    n_epochs_warmup: int = 1
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    optim: Optims = "AdamW"
    loss_type: Losses = "MSE"
    init_pos_enc_dropout: float = 0.0

    save_per: int = 2


# TODO: make the wrapeprs take in various params; cache slope type etc on AlibiVitWrapper
def get_model(
    model_type: ModelType,
    alibi_slope_type: AlibiSlopeType,
    norm_alibi: bool,
    wrap_alibi: bool,
    n_epochs_warmup: int,
    freeze_abs_pos_emb: bool,
    zero_pos_emb: bool,
    device: torch.device,
) -> nn.Module:
    match model_type:
        case "base":
            model = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device=device)
        case "plus_alibi":
            model = AlibiVitWrapper(
                MODEL_LIST[1],
                add_flash_attn=False,
                device=device,
                slope_type=alibi_slope_type,
                normalize=norm_alibi,
                wrap=wrap_alibi,
            )
        case _:
            raise Exception(f"Unsupported model type {model_type}")

    if freeze_abs_pos_emb or zero_pos_emb:
        assert model.model.pos_embed is not None
        model.model.pos_embed.requires_grad = False  # freeze pos embedding

    if zero_pos_emb:
        assert model.model.pos_embed is not None
        model.model.pos_embed.data.zero_()

    if model_type == "plus_alibi" and n_epochs_warmup <= 0:
        model.set_alibi_enabled(True)
    elif model_type == "plus_alibi" and n_epochs_warmup > 0:
        model.set_alibi_enabled(False)

    return model


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


def get_loss(loss_type: Losses, reduction: str = "mean"):
    match loss_type:
        case "MSE":
            return nn.MSELoss(reduction=reduction)
        case "MAE":
            return nn.L1Loss(reduction=reduction)
        case _:
            raise Exception(f"Unsupported loss {loss_type}")


def feed_batch_get_loss(
    model: PretrainedViTWrapper,
    optimizer: optim.Optimizer,
    loss_fn,
    batch: torch.Tensor,
    pos_enc_dropout: float,
    training: bool,
    device: str = "cuda",
) -> float:
    x, y_true = batch
    x = x.to(device)
    y_true = y_true.to(device)
    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    with torch.set_grad_enabled(training):
        y_pred = model.forward_features(x, make_2D=True, abs_pos_enc_drop_prob=pos_enc_dropout)
        loss = loss_fn(y_pred, y_true)
        if training:
            loss.backward()
            optimizer.step()
    x = x.to("cpu")
    y_true = y_true.to("cpu")
    return loss.item()


EXPR_PATH = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M')}"
SEED = 1025
N_VIS = 32
seed_everything(SEED)

# cfg = Config()
cfg = Config(
    experiment_name="alibi_zero_pos_emb_slower",
    model_type="plus_alibi",
    zero_pos_emb=True,
    n_epochs=100,
    batch_size=128,
    n_epochs_warmup=-1,
    init_pos_enc_dropout=1.0,
    lr=1e-4,
)

try:
    rmtree(EXPR_PATH)
except FileNotFoundError:
    pass
makedirs(EXPR_PATH, exist_ok=True)

writer = SummaryWriter(EXPR_PATH)
writer.add_hparams(cfg.__dict__, {})
writer.add_text("desc", cfg.experiment_name)

DEVICE = "cuda:1"
train_ds = HomogenizedEmbeddingDataset("data/IN_reduced_224", "train", store_in_memory=True)
val_ds = HomogenizedEmbeddingDataset("data/IN_reduced_224", "val", store_in_memory=True)

print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")

train_dl = DataLoader(train_ds, cfg.batch_size, True, drop_last=True)
val_dl = DataLoader(val_ds, cfg.batch_size, True, drop_last=True)


# model = AlibiVitWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
# model.set_alibi_enabled(False)
model = get_model(
    cfg.model_type,
    cfg.alibi_slope_type,
    cfg.norm_alibi,
    cfg.wrap_alibi,
    cfg.n_epochs_warmup,
    cfg.freeze_abs_pos_emb,
    cfg.zero_pos_emb,
    DEVICE,
)

optimizer = get_optim(cfg.optim, model, cfg.lr)
loss_fn = get_loss(cfg.loss_type)

train_losses: list[float] = []
val_losses: list[float] = []
best_val_loss = 1e6

dropout_prob = cfg.init_pos_enc_dropout

# TODO:
# - pca vis for sqircle & dog (2 different res) for base dv2, trained and cleaned
# - consider test dl for zero pos enc
# - consider complilation
# - consider LR scheduling

for epoch in range(cfg.n_epochs):
    if epoch == cfg.n_epochs_warmup:
        print("Enabling Alibi")
        model.set_alibi_enabled(True)

    if epoch > 2 * cfg.n_epochs_warmup:
        # schedule dropout
        n_dropout_epochs_total = cfg.n_epochs - max(2 * cfg.n_epochs_warmup, 0)
        n_dropout_epochs_current = epoch - max(2 * cfg.n_epochs_warmup, 0)

        frac = n_dropout_epochs_current / n_dropout_epochs_total
        dropout_prob = cfg.init_pos_enc_dropout + (1 - cfg.init_pos_enc_dropout) * frac

    train_loss = 0.0
    batch: torch.Tensor
    for batch in train_dl:
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, pos_enc_dropout=dropout_prob, training=True, device=DEVICE
        )
        train_loss += loss
    train_loss /= len(train_dl)
    val_loss = 0.0
    batch: torch.Tensor
    # for batch in [next(iter(val_dl))][:1]:
    for batch in val_dl:
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, pos_enc_dropout=dropout_prob, training=False, device=DEVICE
        )
        val_loss += loss
    val_loss /= len(val_dl)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{EXPR_PATH}/best_model.pth")

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("loss/dropout_prob", dropout_prob, epoch)

    print(f"Epoch {epoch:04d}/{cfg.n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if epoch % cfg.save_per == 0:
        x, y_true = batch
        x = x[:N_VIS].to(DEVICE)
        y_true = y_true[:N_VIS].to(DEVICE)
        y_pred = model.forward_features(x, make_2D=True)

        vis_img = visualise(x, y_true, y_pred, "tmp/val_vis.png")
        vis_img_arr = np.array(vis_img).transpose(2, 0, 1)
        writer.add_image("vis/val_batch", vis_img_arr, epoch)
