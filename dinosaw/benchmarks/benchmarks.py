import torch
from lightning import Trainer
from dinosaw.benchmarks.benchmark_models import BenchmarkModel

from pytorch_lightning.loggers import TensorBoardLogger

from dinosaw.benchmarks.VOC12 import VOC_Dataset
from dinosaw.benchmarks.ADE20K import ADE20KDataset
import dinosaw.benchmarks.landsat as landsat

from dataclasses import dataclass, field

from typing import Literal

Benchmark = Literal["VOC12", "ADE20K", "landsat"]
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
    set_255_0: bool = False
    dtype: torch.dtype = torch.float32
    load_in_memory: bool = False

    # trainer
    devices: list[int] | int = field(default_factory=lambda: [0])
    max_steps: int = 40_000
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    precision: str = "16-mixed"

    # model
    # ckpt_path is also used in dataset if load_in_memor = True
    checkpoint_path: str = "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_good_models/try_reproduce_AdamW_batch_128_lr_1e-4_mse_direct_batch_drop_m=1-epoch=99-val_loss=0.12.ckpt"
    benchmark: Benchmark = "VOC12"
    metrics: list[Metrics] = field(default_factory=lambda: ["mIoU"])
    loss_func: Losses = "CE"
    lr: float = 1e-4
    optim: Optimizer = "AdamW"
    upsampling_method: UpsamplingMethod = "bilinear"
    upsampling_size: tuple[int, int] = (img_size, img_size)

    # only needed if alibimodel was trained with higher resolution
    train_hw: int | None = None


def main(cfg: Config):
    train_loader = torch.utils.data.DataLoader(
        ADE20KDataset(
            base_path=cfg.base_path,
            mode="train",
            img_size=cfg.img_size,
            # set_255_0=cfg.set_255_0,
            dtype=cfg.dtype,
            # load_in_memory=cfg.load_in_memory,
            # checkpoint_path=cfg.checkpoint_path,
        ),
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        ADE20KDataset(
            base_path=cfg.base_path,
            mode="val",
            img_size=cfg.img_size,
            # set_255_0=cfg.set_255_0,
            dtype=cfg.dtype,
            # load_in_memory=cfg.load_in_memory,
            # checkpoint_path=cfg.checkpoint_path,
        ),
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
    )

    if cfg.benchmark == "landsat":
        train_loader, val_loader = landsat.get_data_loaders(
            cfg.img_size, cfg.batch_size
        )

    logger = TensorBoardLogger("lightning_logs", name=cfg.name)

    trainer = Trainer(
        devices=cfg.devices,
        max_steps=cfg.max_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        precision=cfg.precision,
        logger=logger,
        strategy="ddp",
        accumulate_grad_batches=4,
    )

    model = BenchmarkModel(
        checkpoint_path=cfg.checkpoint_path,
        benchmark=cfg.benchmark,
        metrics=cfg.metrics,
        lr=cfg.lr,
        loaded_feats=cfg.load_in_memory,
        train_hw=cfg.train_hw,
        optim=cfg.optim,
        loss_func=cfg.loss_func,
        upsampling_method=cfg.upsampling_method,
        upsampling_size=cfg.upsampling_size,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cfg = Config(
        name="test_Dv2_landsat_lr53-3_batch64_distributed",
        benchmark="landsat",
        base_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/ADE20K",
        checkpoint_path="/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps_new/normal_518_Dv2_feats_alibi_plus2epochs-epoch=94-val_loss=0.05_last_epoch.ckpt",  # "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/dinosaw/trained_models_in_steps_new/nothing-epoch=03-val_loss=1.70_last_epoch_copy.ckpt",
        train_hw=37,
        batch_size=64,
        lr=5e-3,
        precision="16-mixed",
        devices=2,
    )

    main(cfg)
