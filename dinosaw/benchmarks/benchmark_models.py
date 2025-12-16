import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
from torchmetrics.functional.segmentation import mean_iou
from dinosaw.models import PEModel
from dinosaw.utils import do_2D_pca, normalize
from dinosaw.benchmarks.bench_utils import colorize

from typing import Literal


Benchmark = Literal["VOC12"]
Losses = Literal["CE"]
Metrics = Literal["mIoU"]
Optimizer = Literal["Adam", "AdamW", "SGD"]
UpsamplingMethod = Literal["nearest", "linear", "bilinear", "trilinear"]


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


class BenchmarkModel(LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        benchmark: Benchmark,
        loss_func: Losses = "CE",
        metrics: list[Metrics] | None = None,
        optim: Optimizer = "AdamW",
        lr: float = 1e-4,
        upsampling_method: UpsamplingMethod = "bilinear",
        upsampling_size: tuple[int, int] = (518, 518),
        # if data_loader yields feature maps not images
        loaded_feats: bool = False,
        # only used when AlibiDino trained on larger images is used
        train_hw: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_func = loss_func
        self.dino = PEModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, map_location="cuda:0", train_hw=train_hw
        )
        self.head = get_head(benchmark)
        self.metrics = metrics
        self.optim = optim
        self.lr = lr
        self.upsampling_method = upsampling_method
        self.upsampling_size = upsampling_size
        self.loaded_feats = loaded_feats

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
        if not self.loaded_feats:
            lr_feats = self.dino(x)
            hr_feats = F.interpolate(
                input=lr_feats, size=self.upsampling_size, mode=self.upsampling_method
            )
            pred = self.head(hr_feats)
        else:
            hr_feats = F.interpolate(
                input=x, size=self.upsampling_size, mode=self.upsampling_method
            )
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
            # print(f"{input.shape=};{output.shape=};{target.shape=}")
            if not self.loaded_feats:
                orig = normalize(input.to("cpu").squeeze())
            else:
                orig = F.interpolate(
                    torch.tensor(
                        do_2D_pca(
                            input.to("cpu").squeeze(),
                            n_components=3,
                            post_norm="minmax",
                        )
                    )
                    .transpose(0, 2)
                    .transpose(1, 2)
                    .unsqueeze(0),  # (C,H,W) -> (1,C,H,W) (for interpolate)
                    self.upsampling_size,
                    mode=self.upsampling_method,
                ).squeeze()  # (1,C,H,W) -> (C,H,W)
            pred = colorize(output.cpu().argmax(dim=0))
            target = colorize(target.cpu().argmax(dim=0))
            output = (
                torch.tensor(
                    do_2D_pca(
                        output.to("cpu").squeeze(), n_components=3, post_norm="minmax"
                    )
                )
                .transpose(0, 2)
                .transpose(1, 2)
            )
            # print(f"{orig.shape=};{input.shape=};{output.shape=};{target.shape=}")
            res.append(torch.stack([orig, output, pred, target]))
        res = torch.concat(res, dim=0)
        # print(res.shape)
        return make_grid(res, 4)
