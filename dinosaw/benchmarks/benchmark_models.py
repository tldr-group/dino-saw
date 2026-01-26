import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional import jaccard_index
from torchmetrics import JaccardIndex

from dinosaw.models import PEModel
from dinosaw.utils import do_2D_pca, normalize
from dinosaw.benchmarks.bench_utils import colorize

from typing import Literal

from VOC12 import to_VOC_label

Benchmark = Literal["VOC12", "ADE20K", "landsat"]
Losses = Literal["CE"]
Metrics = Literal["mIoU"]
Optimizer = Literal["Adam", "AdamW", "SGD"]
UpsamplingMethod = Literal["nearest", "linear", "bilinear", "trilinear"]


def get_head(benchmark: Benchmark):
    match benchmark:
        case "VOC12":
            head = torch.nn.Sequential(
                torch.nn.SyncBatchNorm(num_features=384),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(in_channels=384, out_channels=21, kernel_size=1),
            )

        case "ADE20K":
            head = torch.nn.Sequential(
                torch.nn.SyncBatchNorm(num_features=384),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(in_channels=384, out_channels=151, kernel_size=1),
            )
        case "landsat":
            head = torch.nn.Sequential(
                torch.nn.SyncBatchNorm(num_features=384),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(in_channels=384, out_channels=134, kernel_size=1),
            )
        case _:
            raise Exception(f"No benchmark '{benchmark}' found")
    return head


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_dim=384, num_classes=21, out_dim=518):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        # print(f"{x.shape=}")
        lin_in = x.flatten(start_dim=2).transpose(1, 2)
        # print(f"{lin_in.shape=}")
        out = self.linear(lin_in)
        # print(f"{out.shape=}")
        reshaped_out = out.transpose(1, 2).reshape(
            (x.shape[0], 21, self.out_dim, self.out_dim)
        )
        # print(f"{reshaped_out.shape=}")
        return reshaped_out


def get_linear_head():
    # TODO: add benchmark case switch
    head = LinearClassifier()
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
        # self.dino = PEModel.load_from_checkpoint(
        #     checkpoint_path=checkpoint_path, train_hw=train_hw
        # )
        # self.dino = PEModel()
        # self.dino.configure_model()
        from dinosaw.models.vit_wrapper import PretrainedViTWrapper, MODEL_LIST

        self.dino = PretrainedViTWrapper(
            MODEL_LIST[1], add_flash_attn=False, device=self.device
        ).half()
        self.benchmark = benchmark
        self.head = get_head(benchmark)  # get_linear_head()
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
        self.dino.eval()

    def training_step(self, batch, batch_idx):
        if self.benchmark == "landsat":
            img = batch["image"]
            target = batch["mask"]
        else:
            img, target = batch

        target = target

        pred = self.forward(img)
        loss = self.calc_loss(pred, target)
        self.log("train_loss", loss, sync_dist=True)
        self.calc_metrics(pred, target, mode="train")

        return loss

    def validation_step(self, batch, batch_index):
        if self.benchmark == "landsat":
            img = batch["image"]
            target = batch["mask"]
        else:
            img, target = batch
        target = target

        pred = self.forward(img)
        loss = self.calc_loss(pred, target)
        self.log("val_loss", loss, sync_dist=True)
        self.calc_metrics(pred, target, mode="val")

        if batch_index == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_image(
                "intermediate_output",
                self.gen_vis_grid(img[:4], pred[:4], target[:4]),
                self.current_epoch,
            )

        return loss

    def old_forward(self, x):
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

    def forward(self, x):
        x = x.half()
        if self.loaded_feats:
            lr_feats = x
        else:
            lr_feats = self.dino.forward_features(x, make_2D=True)

        # lr_feats[:, [47, 113, 117, 359], :, :] = 0
        lr_pred = self.head(lr_feats.float()).half()
        res = []
        for feat in lr_pred:
            res.append(
                F.interpolate(
                    input=feat.unsqueeze(0),
                    size=self.upsampling_size,
                    mode=self.upsampling_method,
                ).half()
            )
        hr_pred = torch.concat(res, dim=0)
        return hr_pred

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
                loss = F.cross_entropy(
                    pred,
                    target.long(),
                    ignore_index=255 if self.benchmark == "VOC12" else 0,
                )
            case _:
                raise Exception(f"Unkown lossfunction '{self.loss_func}'")
        return loss

    def calc_metrics(self, pred, target, mode: str):
        if self.metrics is not None:
            for metric in self.metrics:
                match metric:
                    case "mIoU":
                        match self.benchmark:
                            case "VOC12":
                                num_classes = 21
                            case "ADE20K":
                                num_classes = 151
                            case "landsat":
                                num_classes = 134
                        # miou = (
                        #     pred.argmax(dim=1),
                        #     target.argmax(dim=1),
                        #     input_format="index",
                        # )
                        miou = jaccard_index(
                            pred.argmax(dim=1),
                            target,
                            task="multiclass",
                            average="macro",
                            ignore_index=255 if self.benchmark == "VOC12" else 0,
                            num_classes=num_classes,
                        )
                        self.log(f"{mode}_mIoU", miou.mean(), sync_dist=True)
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
            target = colorize(target.cpu())
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
