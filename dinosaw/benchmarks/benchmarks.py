import torch
from lightning import Trainer
from dinosaw.benchmarks.benchmark_models import BenchmarkModel

from pytorch_lightning.loggers import TensorBoardLogger

from dinosaw.benchmarks.VOC12 import VOC_Dataset

from dataclasses import dataclass

from typing import Literal

Benchmark = Literal["VOC12"]
Losses = Literal["CE"]
Metrics = Literal["mIoU"]
Optimizer = Literal["Adam", "AdamW", "SGD"]
UpsamplingMethod = Literal["nearest", "linear", "bilinear", "trilinear"]


@dataclass
class Config:
    # model
    name: str = "default_name"

    # data
    batch_size: int = 16
    base_path: str = (
        "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/VOC"
    )
    img_size: int = 518
    set_255_0: bool = True
    dtype: torch.dtype = torch.float32
    load_in_memory: bool = False

    # trainer
    devices: list[int] | int = [0]
    max_steps: int = 40_000
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    precision: str = "16-mixed"

    # model
    # ckpt_path is also used in dataset if load_in_memor = True
    checkpoint_path: str = "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_good_models/try_reproduce_AdamW_batch_128_lr_1e-4_mse_direct_batch_drop_m=1-epoch=99-val_loss=0.12.ckpt"
    benchmark: Benchmark = "VOC12"
    metrics: list[Metrics] = ["mIoU"]
    loss_func: Losses = "CE"
    lr: float = 1e-4
    # only needed if alibimodel was trained with higher resolution
    train_hw: int | None


def main(cfg: Config):
    train_loader = torch.utils.data.DataLoader(
        VOC_Dataset(
            base_path=cfg.base_path,
            mode="train",
            img_size=cfg.img_size,
            set_255_0=cfg.set_255_0,
            dtype=cfg.dtype,
            load_in_memory=cfg.load_in_memory,
            checkpoint_path=cfg.checkpoint_path,
        ),
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        VOC_Dataset(
            base_path=cfg.base_path,
            mode="val",
            img_size=cfg.img_size,
            set_255_0=cfg.set_255_0,
            dtype=cfg.dtype,
            load_in_memory=cfg.load_in_memory,
            checkpoint_path=cfg.checkpoint_path,
        ),
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
    )

    logger = TensorBoardLogger("lightning_logs", name=cfg.name)

    trainer = Trainer(
        devices=cfg.devices,
        max_steps=cfg.max_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        precision=cfg.val_check_interval,
        logger=logger,
    )

    trainer.fit(
        model=BenchmarkModel(
            checkpoint_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps_new/518_resized_cosine_batch_32_lr=1e-4-epoch=07-val_loss=0.09.ckpt",
            benchmark="VOC12",
            metrics=["mIoU"],
            lr=1e-4,
            loaded_feats=False,
            train_hw=37,
        ),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    cfg = Config(name="test_config")

    main(cfg)
