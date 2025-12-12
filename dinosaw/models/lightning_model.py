import lightning as L
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision.utils import make_grid
import torchvision.transforms.functional as v2
from dinosaw.models.vit_wrapper import (
    PretrainedViTWrapper,
    AlibiVitWrapper,
    MODEL_LIST,
)
from .alibi import AlibiSlopeType
from dinosaw.utils import do_2D_pca, normalize
from copy import deepcopy

from typing import Literal

Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["cosine_embedding", "mse"]


def unfreeze_alibi_and_norms(
    model, unfreeze_layernorms=True, unfreeze_other_patterns=None
):
    """
    Unfreeze alibi parameters and optionally LayerNorm params and other user-specified patterns.
    unfreeze_other_patterns: list of substrings in parameter names to unfreeze (e.g. ['head', 'proj'])
    """
    if unfreeze_other_patterns is None:
        unfreeze_other_patterns = []

    for name, p in model.named_parameters():
        if unfreeze_layernorms and (
            ".norm" in name
            or "ln" in name
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
            or "ls1" in name.lower()
            or "ls2" in name.lower()
        ):
            p.requires_grad = True
        else:
            for pat in unfreeze_other_patterns:
                if pat in name:
                    p.requires_grad = True
                    break


class PEModel(L.LightningModule):
    # TODO:
    # - caching distance matrix
    # - Add option to use DINOv3

    def __init__(
        self,
        use_alibi: bool = False,
        remove_pos_embed: bool = False,
        freeze_abs_pos_embed: bool = True,
        loss_func: Losses = "mse",
        optimizer: Optims = "AdamW",
        lr: float = 1e-4,
        slope_type: AlibiSlopeType = "fixed",
        normalize: bool = True,
        wrap: bool = True,
        add_block: bool = False,
        unfreeze_norms: bool = False,
        unfreeze_pattern: list[str] = None,
        train_hw: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_alibi = use_alibi
        self.remove_pos_embed = remove_pos_embed
        self.freeze_abs_pos_embed = freeze_abs_pos_embed
        self.loss_func = loss_func
        self.optim = optimizer
        self.lr = lr
        self.slope_type = slope_type
        self.normalize = normalize
        self.wrap = wrap
        self.add_block = add_block
        self.unfreeze_norms = unfreeze_norms
        self.pattern = unfreeze_pattern
        self.train_hw = train_hw

        self.last_validation_batch = None

    def configure_model(self):
        if self.use_alibi:
            self.vit = AlibiVitWrapper(
                MODEL_LIST[1],
                add_flash_attn=False,
                device=self.device,
                slope_type=self.slope_type,
                normalize=self.normalize,
                wrap=self.wrap,
            )
            if self.train_hw is not None:
                self.vit.set_distance_matrices(self.train_hw, self.train_hw)
        else:
            self.vit = PretrainedViTWrapper(
                MODEL_LIST[1], add_flash_attn=False, device=self.device
            ).train()

        if self.freeze_abs_pos_embed:
            self.vit.model.pos_embed.requires_grad = False

        if self.remove_pos_embed:
            self.vit.model.pos_embed.data.zero_()

        self.base_pos_embed = self.vit.model.pos_embed

        if self.freeze_abs_pos_embed:
            self.base_pos_embed.requires_grad = False

        if self.add_block:
            last_block = self.vit.model.blocks[-1]
            new_block = deepcopy(last_block)
            self.vit.model.blocks.append(new_block)

        if self.pattern is not None or self.unfreeze_norms:
            for p in self.vit.parameters():
                p.requires_grad = False
            unfreeze_alibi_and_norms(
                model=self.vit,
                unfreeze_layernorms=self.unfreeze_norms,
                unfreeze_other_patterns=self.pattern,
            )

    def training_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)

        loss = self.calc_loss(output, target)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        # setting validation pos_embedding to zeros
        self.vit.model.pos_embed = torch.nn.Parameter(
            torch.zeros_like(self.vit.model.pos_embed), requires_grad=False
        )
        return super().on_validation_epoch_start()

    def validation_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)

        loss = self.calc_loss(output, target)

        self.log("val_loss", loss, sync_dist=True)
        self.last_validation_batch = (input, output, target)
        return loss

    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        input, output, target = self.last_validation_batch
        self.vit.model.pos_embed = self.base_pos_embed

        tensorboard.add_image(
            "intermediate_output",
            self.gen_vis_grid(input[:2], output[:2], target[:2]),
            self.current_epoch,
        )

    def forward(self, input):
        return self.vit.forward_features(input, make_2D=True)  # , attn_mask=attn_mask)

    def gen_vis_grid(self, input, output, target):
        res = []
        for input, output, target in zip(input, output, target):
            orig = normalize(input.to("cpu").squeeze())
            pred = v2.resize(
                torch.tensor(
                    do_2D_pca(
                        output.to("cpu").squeeze(), n_components=3, post_norm="minmax"
                    )
                )
                .transpose(0, 2)
                .transpose(1, 2),
                input.shape[-2:-1],
            )
            target = v2.resize(
                torch.tensor(
                    do_2D_pca(
                        target.to("cpu").squeeze(), n_components=3, post_norm="minmax"
                    )
                )
                .transpose(0, 2)
                .transpose(1, 2),
                input.shape[-2:-1],
            )
            res.append(torch.stack([orig, pred, target]))
        res = torch.concat(res, dim=0)
        return make_grid(res, 3)

    def visualize(self, img, prediction):
        fig, axes = plt.subplots(1, 2)
        x = 0
        for ax, img in zip(
            axes.ravel(), [img.to("cpu").squeeze(), prediction.to("cpu").squeeze()]
        ):
            if x == 1:
                img = do_2D_pca(img, n_components=3, post_norm="minmax")
                ax.imshow(img)
            else:
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img.transpose(0, 2).transpose(0, 1).float())
            x += 1

        fig.savefig("test.png")
        plt.close()

    def calc_loss(self, output, target):
        """
        calculates loss function for the model
        output (B,C,H,W)
        target (B,1,C,H,W)
        """

        match self.loss_func:
            case "cosine_embedding":
                loss = nn.functional.cosine_embedding_loss(
                    output.flatten(1),
                    target.squeeze().flatten(1),
                    torch.ones((output.shape[0])).to("cuda"),
                )
            case "mse":
                loss = nn.functional.mse_loss(output, target.squeeze())
            case "cosine_similarity":
                return (
                    1.0
                    - nn.functional.cosine_similarity(output, target.squeeze(), dim=1)
                ).mean()
            case _:
                raise Exception(f"Unsupported lossfunction {self.loss_func}")
        return loss

    def configure_optimizers(self):
        match self.optim:
            case "AdamW":
                optimizer = optim.AdamW(self.parameters(), lr=self.lr)
            case "Adam":
                optimizer = optim.Adam(self.parameters(), lr=self.lr)
            case "SGD":
                optimizer = optim.SGD(self.parameters(), lr=self.lr)
            case _:
                raise Exception(f"Unsupported Optimizer {self.optim}")
        return optimizer
