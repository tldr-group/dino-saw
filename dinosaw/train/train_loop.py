import torch
import json
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from math import ceil
from PIL import Image

from os import makedirs
from shutil import rmtree
from datetime import datetime

from PVW import PretrainedViTWrapper

from dinosaw.datasets.vis_dataset import visualise
from dinosaw.utils import seed_everything
from dinosaw.train.utils import (
    DatasetType,
    Config,
    get_ds,
    get_linear_probe_images,
    get_model,
    get_optim,
    get_loss,
    evaluate_linear_probe,
    serialize_hparams,
)

import logging

logging.getLogger("PVW.utils").setLevel(logging.WARNING)


def feed_batch_get_loss(
    model: PretrainedViTWrapper,
    optimizer: optim.Optimizer,
    loss_fn,
    batch: torch.Tensor,
    training: bool,
    device: str = "cuda",
    dataset_type: DatasetType = "direct",
) -> float:

    if dataset_type == "joint":
        x, y_true, tr = batch
    else:
        x, y_true = batch

    x = x.to(device)
    y_true = y_true.to(device)
    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    with torch.set_grad_enabled(training):
        y_pred = model.forward_features(x, make_2D=True)
        if dataset_type == "joint":
            y_pred = torch.stack([tr_(i) for tr_, i in zip(tr, y_pred)])

        loss = loss_fn(y_pred, y_true)
        if training:
            loss.backward()
            optimizer.step()
    x = x.to("cpu")
    y_true = y_true.to("cpu")
    y_pred = y_pred.to("cpu")
    return loss.item()


SEED = 1025
N_VIS = 32
seed_everything(SEED)

IMG_L = 224
CACHE = False
cfg = Config(
    experiment_name="alibi_dv3_no_norm_wrap_cb",
    ds_type="otf_coco",
    ds_path="../JAFAR/data/COCOStuff/dataset/images",
    img_l=IMG_L,
    model_type="plus_alibi",
    vit_model_type="dinov3_s+",
    stride=14,
    zero_pos_emb=True,
    alibi_slope_type="constant",
    norm_alibi=False,
    wrap_alibi=True,
    jitter_mag=0.0,
    n_epochs=0.5,
    batch_size=256,
    channels_to_blank=[1, 74, 123, 149, 267, 302, 361],
    channel_dup=False,
    do_random_roll=False,
    loss_type="cosine",
    lr=1e-4,
    pretrained=True,
    add_cls_token=True,
    n_reg_tokens=4,
    save_per=1,
    conf_path="models/dinov3",
    existing_checkpoint="models/checkpoints/backbones/dinov3_vits_patch16_plus_reg4.pth",
    lp_interval=0.05,
    lp_homog_micros_path="notebooks/paper_figures/data/linear_probe/homog_micros",
)
# Multiscale training config
# IMG_L = 518
# CACHE = False
# cfg = Config(
#     experiment_name="test_lin_probe",
#     ds_type="otf_coco",
#     ds_path="../JAFAR/data/COCOStuff/dataset/images",
#     img_l=IMG_L,
#     model_type="plus_alibi",
#     vit_model_type="dinov2_s",
#     stride=14,
#     zero_pos_emb=True,
#     alibi_slope_type="constant",
#     norm_alibi=False,
#     wrap_alibi=True,
#     jitter_mag=0.00,
#     n_epochs=0.5,
#     batch_size=32,
#     pretrained=True,
#     channels_to_blank=[47, 55, 89, 113, 117, 228, 359],
#     do_random_roll=False,
#     loss_type="cosine",
#     lr=1e-5,
#     add_cls_token=True,
#     n_reg_tokens=4,
#     # conf_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
#     existing_checkpoint="experiments/ablations/20260708_1420_alibi_dv2_no_norm_wrap_more_cb/best_model.pth",
#     save_per=1,
#     lp_interval=0.05,
#     lp_homog_micros_path="notebooks/paper_figures/data/linear_probe/homog_micros",
# )
print(cfg)

EXPR_PATH = f"experiments/ablations/{datetime.now().strftime('%Y%m%d_%H%M')}_{cfg.experiment_name}"
try:
    rmtree(EXPR_PATH)
except FileNotFoundError:
    pass
makedirs(EXPR_PATH, exist_ok=True)

writer = SummaryWriter(EXPR_PATH)
hparams = serialize_hparams(cfg.__dict__)
writer.add_hparams(hparams, {"loss/train": 0.0, "loss/val": 0.0})
writer.add_text("desc", cfg.experiment_name)

with open(f"{EXPR_PATH}/config.json", "w+") as f:
    json.dump(cfg.__dict__, f, indent=2)


DEVICE: str = "cuda:1"


train_ds, val_ds = get_ds(cfg, DEVICE, cache=CACHE)

print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")


def my_collate(batch):
    # batch is a list of tuples/dicts from __getitem__
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    funcs = [item[2] for item in batch]

    # Manually stack the data, but keep functions as a raw list
    return torch.stack(x), torch.stack(y), funcs


train_dl = DataLoader(train_ds, cfg.batch_size, True, drop_last=True)
val_dl = DataLoader(val_ds, cfg.batch_size, True, drop_last=True)


model = get_model(
    cfg.model_type,
    cfg.alibi_slope_type,
    cfg.norm_alibi,
    cfg.wrap_alibi,
    cfg.zero_pos_emb,
    DEVICE,
    cfg.existing_checkpoint,
    cfg.vit_model_type,
    cfg.stride,
    cfg.conf_path,
    cfg.pretrained,
    cfg.add_cls_token,
    cfg.n_reg_tokens,
    cfg.jitter_mag,
)

optimizer = get_optim(cfg.optim, model, cfg.lr)
loss_fn = get_loss(cfg.loss_type)

train_losses: list[float] = []
val_losses: list[float] = []
best_val_loss = 1e6


lp_batch_interval = -1
linear_probe_images: list[Image.Image] | None = None
if cfg.lp_interval > 0 and cfg.lp_homog_micros_path is not None:
    linear_probe_images = get_linear_probe_images(cfg.lp_homog_micros_path)
    lp_batch_interval = max(1, int(cfg.lp_interval * len(train_dl)))
    print(
        f"Loaded {len(linear_probe_images)} images for linear probing from {cfg.lp_homog_micros_path} with interval {lp_batch_interval} batches"
    )

max_epochs = ceil(cfg.n_epochs)
for epoch in range(max_epochs):
    train_loss = 0.0
    batch: torch.Tensor

    if epoch == max_epochs - 1 and not (cfg.n_epochs == int(cfg.n_epochs)):
        fraction = cfg.n_epochs - epoch
        epoch_batches = int(fraction * len(train_dl))
    else:
        epoch_batches = len(train_dl)

    for i, batch in enumerate(train_dl):
        if i >= epoch_batches:
            break
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, training=True, device=DEVICE, dataset_type=cfg.ds_type
        )

        train_loss += loss
        if i % 50 == 0:
            print(f"Train batch {i}/{epoch_batches} | Loss: {loss:.4f}")

        if linear_probe_images and i % lp_batch_interval == 0:
            lp_results = evaluate_linear_probe(model, linear_probe_images, DEVICE)
            global_step = epoch * epoch_batches + i
            for ramp, r2 in lp_results.items():
                writer.add_scalar(f"linear_probe/R2_{ramp}", r2, global_step)
            print(f"Train batch {i} | {', '.join([f'{ramp}: {r2:.4f}' for ramp, r2 in lp_results.items()])}")

    train_loss /= epoch_batches

    val_loss = 0.0
    for j, batch in enumerate(val_dl):
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, training=False, device=DEVICE, dataset_type=cfg.ds_type
        )
        val_loss += loss
        if j % 50 == 0:
            print(f"Train batch {j}/{len(val_dl)} | Loss: {loss:.4f}")

    val_loss /= len(val_dl)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.vit.state_dict(), f"{EXPR_PATH}/best_model.pth")

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)

    print(f"Epoch {epoch:04d}/{cfg.n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if epoch % cfg.save_per == 0:
        if cfg.ds_type == "joint":
            x, y_true, tr = batch
        else:
            x, y_true = batch
        x = x[:N_VIS].to(DEVICE)
        y_true = y_true[:N_VIS].to(DEVICE)
        with torch.no_grad():
            y_pred = model.forward_features(x, make_2D=True)

        if cfg.ds_type == "joint":
            x = torch.stack([tr_(i) for tr_, i in zip(tr[:N_VIS], x)])

        vis_img = visualise(x, y_true, y_pred, "tmp/val_vis.png")
        vis_img_arr = np.array(vis_img).transpose(2, 0, 1)
        writer.add_image("vis/val_batch", vis_img_arr, epoch)

        x = x.to("cpu")
        y_true = y_true.to("cpu")
        y_pred = y_pred.to("cpu")
