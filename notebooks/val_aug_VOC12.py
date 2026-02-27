import torch
import torch.nn as nn
from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST, AlibiVitWrapper
from dinosaw.datasets.benchmark_datasets import VOC_Dataset
from typing import Literal
from random import choice
from functools import partial
import matplotlib.pyplot as plt
from dinosaw.utils import seed_everything

Transforms = Literal["roll", "rot90", "flip-ud"]

SEED = 1025
seed_everything(SEED)

BATCH_SIZE = 64
MODEL = "coco"
DEVICE = "cuda:0"

from torchmetrics.classification import MulticlassJaccardIndex

mean_iou = MulticlassJaccardIndex(
    num_classes=21,
    ignore_index=255,
    average="macro",
).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=255)


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(x, dims=(dim,))


def rot(x: torch.Tensor, angle: int) -> torch.Tensor:
    k = angle // 90
    return torch.rot90(x, k=k, dims=(-2, -1))


def shift(x: torch.Tensor, s: int, dir: tuple[int, int]) -> torch.Tensor:
    return torch.roll(x, (dir[0] * s, dir[1] * s), dims=(-2, -1))


def nop(x: torch.Tensor) -> torch.Tensor:
    return x


class AugmentModel(nn.Module):
    def __init__(
        self,
        model: PretrainedViTWrapper,
        device: torch.device,
        size: int,
        tr_type: Transforms,
    ) -> None:
        super().__init__()

        self.channels_to_blank = None
        self.size = size
        self.dino = model.eval()

        # freezing dino backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.SyncBatchNorm(num_features=384),
            # nn.LayerNorm([384, 37, 37]),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
        ).to(device)

        self.img_tr, self.inv_embed_tr = self.get_transform(tr_type)

    def forward(self, x):
        x = self.img_tr(x)
        lr_feats = self.dino.forward_features(x, make_2D=True)
        if self.channels_to_blank is not None:
            lr_feats[:, self.channels_to_blank, :, :] = 0
        lr_feats = self.inv_embed_tr(lr_feats)
        lr_pred = self.head(lr_feats)
        hr_pred = nn.functional.interpolate(
            input=lr_pred, size=(self.size, self.size), mode="bilinear"
        )
        return hr_pred

    def get_transform(self, tr_type: Transforms):
        # tr_types = ("flip", "rot", "shift", "none", "none")
        # tr_type = choice(tr_types)

        if tr_type == "flip-ud":
            # which = choice(("h", "v"))
            # if which == "h":
            #     return partial(flip, dim=-1), partial(flip, dim=-1)
            # else:
            return partial(flip, dim=-2), partial(flip, dim=-2)
        elif tr_type == "flip-lr":
            return partial(flip, dim=-1), partial(flip, dim=-1)
        elif tr_type == "rot90":
            angle = 90
            return partial(rot, angle=angle), partial(rot, angle=-angle)
        elif tr_type == "rot180":
            angle = 180
            return partial(rot, angle=angle), partial(rot, angle=-angle)
        elif tr_type == "rot270":
            angle = 270
            return partial(rot, angle=angle), partial(rot, angle=-angle)
        elif tr_type == "roll":
            dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            dir = choice(dirs)
            s = choice([i for i in range(1, 16)])
            return partial(shift, s=14 * s, dir=dir), partial(
                shift, s=s, dir=(-dir[0], -dir[1])
            )
        elif tr_type == "none":
            return nop, nop
        else:
            raise ValueError(f"Unknown transform type: {tr_type}")


def feed_batch_get_loss(
    model: AugmentModel,
    loss_fn,
    metric_fn,
    batch: torch.Tensor,
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
    model.eval()
    with torch.set_grad_enabled(False):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        metric = metric_fn(y_pred, y_true)
    # x = x.to("cpu")
    # y_true = y_true.to("cpu")
    # y_pred = y_pred.to("cpu")
    return loss.detach().item(), metric.detach().item()


val_ds = VOC_Dataset(
    base_path="../Datasets/VOC",
    mode="val",
)
val_dl = torch.utils.data.DataLoader(
    val_ds,
    BATCH_SIZE,
    False,
    drop_last=True,
    # num_workers=3,
    # pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=4,
)


import pandas as pd

df = pd.DataFrame(
    columns=[
        "model",
        "transform",
        "batch_size",
        "batch_idx",
        "batch_mIoU",
        "batch_loss",
    ]
)
for TR in ["none", "flip-ud", "rot90", "roll", "flip-lr", "rot180", "rot270"]:
    if MODEL == "Dv2":
        model = PretrainedViTWrapper(
            model_identifier=MODEL_LIST[1],
            stride=14,
            add_flash_attn=False,
            device=DEVICE,
        )
    if MODEL == "coco":
        model = AlibiVitWrapper(
            model_identifier=MODEL_LIST[1], add_flash_attn=False, device=DEVICE
        )

    model_aug = AugmentModel(model=model, device=DEVICE, size=518, tr_type=TR)
    model_aug.load_state_dict(
        torch.load("../trained_models/lin_model_WS_VOC_coco.pth", map_location=DEVICE)
    )

    val_losses: list[float] = []
    val_mious: list[float] = []
    best_val_loss = 1e6
    for idx, batch in enumerate(val_dl):
        loss, miou = feed_batch_get_loss(
            model_aug,
            loss_fn,
            mean_iou,
            batch,
            device=DEVICE,
        )

        df = pd.concat(
            [
                pd.DataFrame(
                    [[MODEL, TR, BATCH_SIZE, idx, miou, loss]], columns=df.columns
                ),
                df,
            ],
            ignore_index=True,
        )

df.to_csv(f"{MODEL}_valid_aug.csv")
