import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
from torchmetrics.functional.segmentation import mean_iou
from pytorch_lightning.loggers import TensorBoardLogger

from dinosaw.benchmarks.VOC12 import VOC_Dataset
from dinosaw.models import PEModel
from dinosaw.utils import do_2D_pca, normalize

from typing import Literal

Benchmark = Literal["VOC12"]
Losses = Literal["CE"]
Metrics = Literal["mIoU"]
Optimizer = Literal["Adam", "AdamW", "SGD"]


def get_head(benchmark: Benchmark):
    match benchmark:
        case "VOC12":
            head = torch.nn.Sequential(
                torch.nn.SyncBatchNorm(num_features=384),
                torch.nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
            )
        case _:
            raise Exception(f"No benchmark '{benchmark}' found")
    return head


def pascal_colormap():
    cmap = torch.zeros(256, 3, dtype=torch.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = torch.tensor([r, g, b], dtype=torch.uint8)
    return cmap


def colorize(pred_mask):  # pred_mask: (H,W) int
    CMAP = pascal_colormap()
    # returns (3,H,W)
    rgb = CMAP[pred_mask]  # (H,W,3)
    return rgb.permute(2, 0, 1)  # 3,H,W


class BenchmarkModel(LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        benchmark: Benchmark,
        loss_func: Losses = "CE",
        metrics: list[Metrics] | None = None,
        optim: Optimizer = "AdamW",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_func = loss_func
        self.dino = PEModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, map_location="cuda:0"
        )
        self.head = get_head(benchmark)
        self.metrics = metrics
        self.optim = optim
        self.lr = lr

    def configure_model(self):
        # freeze backbone
        for p in self.dino.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        img, target = batch

        pred = self.forward(img)
        loss = self.calc_loss(pred, target)
        self.log("train_loss", loss)
        self.calc_metrics(pred, target, mode="train")

        return loss

    def validation_step(self, batch, batch_index):
        img, target = batch

        pred = self.forward(img)
        loss = self.calc_loss(pred, target)
        self.log("val_loss", loss)
        self.calc_metrics(pred, target, mode="val")

        if batch_index == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_image(
                "intermediate_output",
                self.gen_vis_grid(img[:4], pred[:4], target[:4]),
                self.current_epoch,
            )

        return loss

    def forward(self, x):
        lr_feats = self.dino(x)
        hr_feats = F.interpolate(input=lr_feats, size=(518, 518), mode="bilinear")
        pred = self.head(hr_feats)
        return pred

    def configure_optimizers(self):
        match self.optim:
            case "AdamW":
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            case "Adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            case "SGD":
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            case _:
                raise Exception(f"Unsupported Optimizer {self.optim}")
        return optimizer

    def calc_loss(self, pred, target):
        match self.loss_func:
            case "CE":
                loss = F.cross_entropy(pred, target.float())
            case _:
                raise Exception(f"Unkown lossfunction '{self.loss_func}'")
        return loss

    def calc_metrics(self, pred, target, mode: str):
        if self.metrics is not None:
            for metric in self.metrics:
                match metric:
                    case "mIoU":
                        miou = mean_iou(pred.argmax(dim=1), target.argmax(dim=1))
                        self.log(f"{mode}_mIoU", miou.mean())
                    case _:
                        raise Exception(f"Unknow metric '{metric}'")

    def gen_vis_grid(self, input, output, target):
        res = []
        for input, output, target in zip(input, output, target):
            orig = normalize(input.to("cpu").squeeze())
            output = resize(
                torch.tensor(
                    do_2D_pca(
                        output.to("cpu").squeeze(), n_components=3, post_norm="minmax"
                    )
                )
                .transpose(0, 2)
                .transpose(1, 2),
                input.shape[-2:-1],
            )
            pred = output.cpu().argmax(dim=0)
            target = target.cpu().argmax(dim=0)
            res.append(torch.stack([pred, target]))
        res = torch.concat(res, dim=0).float().unsqueeze(1)
        # print(res.shape)
        return make_grid(res, 2, normalize=True, value_range=(0, 20))


def main():
    batch_size = 4
    name = "added_batch_norm"

    logger = TensorBoardLogger("lightning_logs", name=name)

    train_loader = torch.utils.data.DataLoader(
        VOC_Dataset(
            base_path="/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
            mode="train",
            img_size=518,
            set_255_0=True,
        ),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

    val_loader = torch.utils.data.DataLoader(
        VOC_Dataset(
            base_path="/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
            mode="val",
            img_size=518,
            set_255_0=True,
        ),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )

    trainer = Trainer(
        max_steps=40_000,
        log_every_n_steps=5,
        val_check_interval=1.0,
        precision="16-mixed",
        logger=logger,
    )
    trainer.fit(
        model=BenchmarkModel(
            checkpoint_path="/home/pawlo/Arbeit/positional_bias/dino-saw/testing_dinov2/trained_models/cosine_sim_batch_128_lr%3D1e-4-epoch%3D99-val_loss%3D0.05_last_epoch.ckpt",
            benchmark="VOC12",
            metrics=["mIoU"],
            lr=1e-4,
        ),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
