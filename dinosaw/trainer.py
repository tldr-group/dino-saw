from lightning import Trainer
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import torch
from typing import Callable, Optional
import torch.nn as nn
import math
from dinosaw.models import PEModel
from dinosaw.datasets import (
    GenericDatasetStudent,
)
import numpy as np
from dataclasses import dataclass, field
from dinosaw.models.alibi import AlibiSlopeType
from typing import Literal

Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["cosine_embedding", "mse", "cosine_similarity"]
DropSchedule = Literal["linear", "cosine", "step"]


class PositionalDropoutScheduler(Callback):
    def __init__(
        self,
        target_attr: str = "model.pos_drop",  # dot-path to your dropout (e.g., "model.pos_drop" or just "pos_drop")
        p_start: float = 0.0,
        p_end: float = 0.1,
        schedule: str | Callable[[int, int, float, float], float] = "linear",
        max_steps: Optional[int] = None,
        clamp: tuple[float, float] = (0.0, 1.0),
        log_name: str = "pos_drop_p",
    ):
        self.target_attr = target_attr
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.schedule = schedule
        self.max_steps = max_steps
        self.clamp = clamp
        self.log_name = log_name
        self._drop_module: Optional[nn.Module] = None
        self.vit = None

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: Optional[str] = None,
    ):
        # Resolve the dropout module
        self._drop_module = self._resolve_target(pl_module, self.target_attr)
        if self._drop_module is None or not hasattr(self._drop_module, "p"):
            raise RuntimeError(
                f"Could not find a Dropout with attribute 'p' at {self._drop_module} "
            )

        # Determine total steps once we know the trainer
        if self.max_steps is None:
            # PL provides an estimate of total optimization steps
            self.max_steps = trainer.estimated_stepping_batches or trainer.max_steps

        # Initialize p
        self._drop_module.p = float(self.p_start)

    def _find_vit(self, pl_module):
        # Passe das an deine Struktur an: z. B. pl_module.vit.model
        vit = getattr(getattr(pl_module, "vit", None), "model", None)
        if vit is None or not hasattr(vit, "pos_embed"):
            for m in pl_module.modules():
                if hasattr(m, "pos_embed"):
                    vit = m
                    break
        if vit is None or not hasattr(vit, "pos_embed"):
            raise RuntimeError(
                "Could not find VisionTransformer with a 'pos_embed' parameter."
            )
        return vit

    def on_fit_start(self, trainer, pl_module):
        self.vit = self._find_vit(pl_module)

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int
    ):
        step = trainer.global_step  # 0-based
        new_p = self._compute_p(step, self.max_steps)
        if new_p == 1:
            with torch.no_grad():
                self.vit.pos_embed = torch.nn.Parameter(
                    torch.zeros_like(self.vit.pos_embed)
                )
        # self._drop_module.p = new_p
        self.vit.pos_drop = torch.nn.modules.Dropout(p=new_p)
        # Optional logging
        pl_module.log(self.log_name, new_p, on_step=True, prog_bar=True, logger=True)

    # --- helpers ---
    def _resolve_target(self, root: nn.Module, path: str) -> Optional[nn.Module]:
        # Try dot-path first
        cur = root
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, nn.Module):
                return cur
        except AttributeError:
            pass
        # Fallback: search by suffix name
        for name, m in root.named_modules():
            if name.endswith(path) and isinstance(m, nn.Module):
                return m
        return None


class BatchDropout(Callback):
    def __init__(
        self,
        p_start: int = 0,
        p_end: int = 1,
        schedule="linear",
        max_steps: int = None,
        clamp: tuple[float, float] = (0.0, 1.0),
        log_name: str = "pos_drop_p",
        warmup_steps: int = 0,
    ):
        self.p_start = p_start
        self.p_end = p_end
        self.schedule = schedule
        self.max_steps = max_steps
        self.clamp = clamp
        self.log_name = log_name
        self.warmup_steps = warmup_steps

        self.base_pos_embed = None

    def setup(self, trainer, pl_module, stage):
        if self.max_steps is None:
            self.max_steps = trainer.estimated_stepping_batches or trainer.max_steps
        return super().setup(trainer, pl_module, stage)

    def on_fit_start(self, trainer, pl_module):
        self.base_pos_embed = pl_module.vit.model.pos_embed

    def _compute_p(self, step: int, max_steps: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return 0.0
        else:
            step = step - self.warmup_steps

        # Normalize step to [0, 1]
        denom = max(1, (max_steps or 1) - 1)
        t = min(1.0, step / denom)

        if callable(self.schedule):
            p = self.schedule(step, max_steps, self.p_start, self.p_end)
        elif self.schedule == "linear":
            p = self.p_start + (self.p_end - self.p_start) * t
        elif self.schedule == "cosine":
            p = self.p_end - (self.p_end - self.p_start) * 0.5 * (
                1.0 + math.cos(math.pi * t)
            )
        elif self.schedule == "step":
            total_bins = 4
            bins = int(math.floor(t * total_bins))  # 0..10
            p = self.p_start + 0.25 * bins

            # Respect p_end as a cap (if p_end > p_start it's an upper cap; if p_end < p_start it's a lower cap)
            if self.p_end >= self.p_start:
                p = min(p, self.p_end)
            else:
                p = max(p, self.p_end)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        lo, hi = self.clamp
        return float(max(lo, min(hi, p)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        current_p = self._compute_p(step=step, max_steps=self.max_steps)
        if np.random.random(1) < current_p:
            pl_module.vit.model.pos_embed = torch.nn.Parameter(
                torch.zeros_like(pl_module.vit.model.pos_embed), requires_grad=False
            )
        else:
            pl_module.vit.model.pos_embed = pl_module.base_pos_embed

        pl_module.log(
            self.log_name, current_p, on_step=True, prog_bar=True, logger=True
        )
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)


@dataclass
class Config:
    # logging related
    experiment_name: str = "default_experiment"
    log_every_n_steps: int = 2
    val_check_interval: float = 0.1

    # Alibi related
    add_alibi: bool = False
    alibi_slope_type: AlibiSlopeType = "constant"
    norm_alibi: bool = True
    wrap_alibi: bool = True

    # abs embed realted
    freeze_abs_pos_emb: bool = True
    zero_pos_emb: bool = False
    use_batch_drop: bool = False
    batch_drop_steps: int = 5_000
    batch_drop_schedule: DropSchedule = "linear"

    # training hparams
    base_path: str = "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_224_normal_Dv2"
    max_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    optim: Optims = "AdamW"
    loss_type: Losses = "mse"
    accelerator: str = "gpu"
    which_device: list[int] = field(default_factory=lambda: [0])
    resize_size: int = None

    # altering the model
    add_block: bool = False
    unfreeze_norms: bool = False  # if True freezes the rest of the model
    unfreeze_pattern: list[str] = None  # if not None freezes the rest of the model

    # continue training
    continue_training: bool = False
    ckpt_path: str = None
    train_hw: int = 16


def main(cfg: Config):
    train_loader = torch.utils.data.DataLoader(
        GenericDatasetStudent(
            base_path=cfg.base_path,
            split="train",
            resize_size=cfg.resize_size,
        ),
        batch_size=cfg.batch_size,
        num_workers=48,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        GenericDatasetStudent(
            base_path=cfg.base_path,
            split="val",
            resize_size=cfg.resize_size,
        ),
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
    )

    # name = "continue_long_run_AdamW_batch_128_lr_1e-4->1e-6_mse_direct_batch_drop_m=1"
    name = cfg.experiment_name

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="trained_models_in_steps_new/",
        filename=name + "-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    endcheckpoint_callback = ModelCheckpoint(
        dirpath="trained_models_in_steps_new/",
        filename=name + "-{epoch:02d}-{val_loss:.2f}_last_epoch",
    )

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("lightning_logs", name=name)

    if cfg.use_batch_drop:
        callbacks = [
            checkpoint_callback,
            BatchDropout(
                p_start=0,
                p_end=1,
                schedule=cfg.batch_drop_schedule,
                max_steps=cfg.batch_drop_steps,
            ),
            endcheckpoint_callback,
        ]
    else:
        callbacks = [
            checkpoint_callback,
            endcheckpoint_callback,
        ]

    trainer = Trainer(
        devices=cfg.which_device,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        accelerator=cfg.accelerator,
        val_check_interval=cfg.val_check_interval,
        logger=logger,
    )  # , strategy="ddp" #AbsPEFader(start=1, end=0.0, total_steps=7_000, freeze_pos_embed=True),

    if cfg.continue_training:
        assert cfg.ckpt_path is not None

    trainer.fit(
        model=PEModel(
            loss_func=cfg.loss_type,
            optimizer=cfg.optim,
            lr=cfg.lr,
            remove_pos_embed=cfg.zero_pos_emb,
            freeze_abs_pos_embed=cfg.freeze_abs_pos_emb,
            use_alibi=cfg.add_alibi,
            slope_type=cfg.alibi_slope_type,
            normalize=cfg.norm_alibi,
            wrap=cfg.norm_alibi,
            add_block=cfg.add_block,
            unfreeze_norms=cfg.unfreeze_norms,
            unfreeze_pattern=cfg.unfreeze_pattern,
            train_hw=cfg.train_hw,
        ),
        ckpt_path=cfg.ckpt_path if cfg.continue_training else None,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    cfg = Config(
        experiment_name="normal_518_Dv2_feats_alibi_plus10epochs",
        base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/Dataset/IN_reduced_518_normal_Dv2",
        batch_size=32,
        max_epochs=104,
        lr=1e-4,
        add_alibi=True,
        zero_pos_emb=True,
        which_device=[0],
        loss_type="cosine_similarity",
        resize_size=518,
        continue_training=True,
        ckpt_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps_new/normal_Dv2_feats_alibi-epoch=93-val_loss=0.05.ckpt",
        # unfreeze_norms=True,
        # unfreeze_pattern=["attn"],
        train_hw=16,
    )
    main(cfg)
