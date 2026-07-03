import torch
from torch import nn, optim
import numpy as np

from torch.utils.data import DataLoader, Dataset

from os import makedirs
from shutil import rmtree
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

from dinosaw.datasets.vis_dataset import visualise
from dinosaw.datasets.train_student_dataset import HomogenizedEmbeddingDataset
from dinosaw.datasets.joint_embed_dataset import JointEmbeddingDataset, OTFEmbeddingDataset
from dinosaw.alibi_logic import AlibiSlopeType
from dinosaw.wrappers import MODEL_LIST, PretrainedViTWrapper
from dinosaw.wrappers.alibi import add_alibi, replace_pe_with_sincos
from PVW.factory import BackboneConfig, BackboneRegistry
from dinosaw.utils import seed_everything, closest_resize, closest_resize_crop

from dataclasses import dataclass, field
from typing import Literal

Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["MSE", "MAE", "cosine", "CE"]
ModelType = Literal["base", "plus_alibi", "nope", "plus_sincos"]
DatasetType = Literal["joint", "direct", "otf_coco"]


@dataclass
class Config:
    experiment_name: str = "default_experiment"

    ds_path: str = "data/IN_reduced_base_224"
    ds_type: DatasetType = "direct"
    img_l: int = 224

    model_type: ModelType = "base"
    dino_chk_path: str | None = None
    vit_model_type: str = MODEL_LIST[1]
    stride: int = 14
    pretrained: bool = True

    add_cls_token: bool = True
    n_reg_tokens: int = 4

    alibi_slope_type: AlibiSlopeType = "constant"
    norm_alibi: bool = True
    wrap_alibi: bool = True
    freeze_pos_emb: bool = True
    zero_pos_emb: bool = False
    jitter_mag: float = 0.0

    channels_to_blank: list[int] = field(default_factory=lambda: [])
    channel_dup: bool = False
    do_random_roll: bool = False

    n_epochs_warmup: int = 1
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    optim: Optims = "AdamW"
    loss_type: Losses = "MSE"

    save_per: int = 2

    existing_checkpoint: str | None = None


# TODO: make the wrapeprs take in various params; cache slope type etc on AlibiVitWrapper
def get_model(
    model_type: ModelType,
    alibi_slope_type: AlibiSlopeType,
    norm_alibi: bool,
    wrap_alibi: bool,
    n_epochs_warmup: int,
    freeze_abs_pos_emb: bool,
    zero_pos_emb: bool,
    device: str | torch.device,
    existing_checkpoint: str | None = None,
    vit_model_type: str = MODEL_LIST[1],
    stride: int = 14,
    chk_path: str | None = None,
    pretrained: bool = True,
    add_cls_token: bool = True,
    n_reg_tokens: int = 4,
    jitter_mag: float = 0.0,
) -> PretrainedViTWrapper:

    is_dinov3 = "dv3" in vit_model_type or "dinov3" in vit_model_type
    print(vit_model_type)

    from PVW.types import ARCHS_TO_TIMM_IDS, ARCHS_TO_LOCAL_IDS
    arch_name = None
    for k, v in ARCHS_TO_TIMM_IDS.items():
        if v == vit_model_type:
            arch_name = k
            break
    if arch_name is None:
        for k, v in ARCHS_TO_LOCAL_IDS.items():
            if v == vit_model_type:
                arch_name = k
                break
    if arch_name is None:
        arch_name = vit_model_type

    backbone_type = "torch_hub" if is_dinov3 else "timm"
    cfg = BackboneConfig(
        backbone_type=backbone_type,
        model_arch=arch_name,  # type: ignore
        pretrained=pretrained,
        checkpoint_path=chk_path,
        model_conf_path="dinov3" if is_dinov3 else None,
        stride=(stride, stride),
    )

    if model_type == "plus_alibi":
        mod = partial(
            add_alibi,
            slope_type=alibi_slope_type,
            n_reg_tokens=n_reg_tokens,
            metric="euclidean",
            normalize=norm_alibi,
            wrap=wrap_alibi,
            add_cls=add_cls_token,
            jitter_mag=jitter_mag,
        )
        cfg.modifications.append(mod)
    elif model_type == "plus_sincos":
        cfg.modifications.append(replace_pe_with_sincos)

    vit = BackboneRegistry.build(cfg)
    model = PretrainedViTWrapper(vit=vit, device=device)
    model.set_alibi_enabled = lambda enabled: None

    if freeze_abs_pos_emb or zero_pos_emb or model_type == "nope":
        if is_dinov3:
            pass
        else:
            if hasattr(model.vit, "pos_embed") and model.vit.pos_embed is not None:
                model.vit.pos_embed.requires_grad = False  # freeze pos embedding

    if zero_pos_emb or model_type == "nope":
        if is_dinov3:
            print("dinov3, zeroing rope_embed")
            if hasattr(model.vit, "rope_embed"):
                model.vit.rope_embed = None
        else:
            if hasattr(model.vit, "pos_embed") and model.vit.pos_embed is not None:
                model.vit.pos_embed.data.zero_()

    if existing_checkpoint is not None:
        print(f"Loading existing checkpoint from {existing_checkpoint}")
        model.load_state_dict(torch.load(existing_checkpoint, map_location=device, weights_only=True))

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
        case "cosine":
            cosine_sim = nn.CosineSimilarity(dim=1)

            def cosine_sim_wrapper(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                return (1.0 - cosine_sim(x1, x2)).mean()

            return cosine_sim_wrapper
        case _:
            raise Exception(f"Unsupported loss {loss_type}")


def get_ds(cfg: Config, device: str) -> tuple[Dataset, Dataset]:

    train_ds: Dataset
    val_ds: Dataset
    if cfg.ds_type == "direct":
        tr = closest_resize(cfg.img_l, cfg.img_l, 14)

        train_ds = HomogenizedEmbeddingDataset(
            cfg.ds_path,
            "train",
            transform=tr,
            store_in_memory=CACHE,
            channels_to_blank=cfg.channels_to_blank,
            channel_dup=cfg.channel_dup,
            do_random_roll=cfg.do_random_roll,
        )
        val_ds = HomogenizedEmbeddingDataset(
            cfg.ds_path,
            "val",
            transform=tr,
            store_in_memory=CACHE,
            channels_to_blank=cfg.channels_to_blank,
            channel_dup=cfg.channel_dup,
            do_random_roll=cfg.do_random_roll,
        )
        return (train_ds, val_ds)
    elif cfg.ds_type == "otf_coco":
        tr = closest_resize_crop(cfg.img_l, 14)
        embed_model: PretrainedViTWrapper = get_model(
            "base",
            "constant",
            False,
            False,
            -1,
            False,
            False,
            device,
            stride=cfg.stride,
            vit_model_type=cfg.vit_model_type,
            chk_path=cfg.dino_chk_path,
        )
        embed_model.eval()
        embed_model = embed_model.to(device)
        embed_model = torch.compile(embed_model)
        train_ds = OTFEmbeddingDataset(
            embed_model,
            f"{cfg.ds_path}/train2017",
            "train",
            transform=tr,
            dtype=torch.float32,
            device=device,
            fname_file_path=f"{cfg.ds_path}/train2017.txt",
            norm_feats=False,
            channels_to_blank=cfg.channels_to_blank,
            channel_dup=cfg.channel_dup,
            _do_random_roll=cfg.do_random_roll,
        )
        val_ds = OTFEmbeddingDataset(
            embed_model,
            f"{cfg.ds_path}/val2017",
            "val",
            transform=tr,
            dtype=torch.float32,
            device=device,
            fname_file_path=f"{cfg.ds_path}/val2017.txt",
            norm_feats=False,
            channels_to_blank=cfg.channels_to_blank,
            channel_dup=cfg.channel_dup,
            _do_random_roll=cfg.do_random_roll,
        )
        return (train_ds, val_ds)
    else:
        tr = closest_resize_crop(cfg.img_l, 14)
        embed_model: PretrainedViTWrapper = get_model("base", "constant", False, False, -1, False, False, device)
        embed_model.eval()
        embed_model = embed_model.to(device)
        embed_model = torch.compile(embed_model)
        train_ds = JointEmbeddingDataset(
            embed_model,
            f"{cfg.ds_path}/train2017",
            "train",
            transform=tr,
            dtype=torch.float32,
            device=device,
            fname_file_path=f"{cfg.ds_path}/train2017.txt",
            norm_feats=False,
        )
        val_ds = JointEmbeddingDataset(
            embed_model,
            f"{cfg.ds_path}/val2017",
            "val",
            transform=tr,
            dtype=torch.float32,
            device=device,
            fname_file_path=f"{cfg.ds_path}/val2017.txt",
            norm_feats=False,
        )
        return (train_ds, val_ds)


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
    experiment_name="dv2_raster_fixed",
    ds_type="otf_coco",
    ds_path="../JAFAR/data/COCOStuff/dataset/images",
    img_l=IMG_L,
    model_type="plus_sincos",
    vit_model_type=MODEL_LIST[1],
    stride=14,
    zero_pos_emb=False,
    freeze_pos_emb=True,
    alibi_slope_type="constant",
    norm_alibi=True,
    wrap_alibi=True,
    jitter_mag=0.0,
    n_epochs=15,
    batch_size=256,
    # channels_to_blank=[47, 113, 117, 359],
    channel_dup=False,
    do_random_roll=False,
    loss_type="cosine",
    n_epochs_warmup=-1,
    lr=1e-4,
    pretrained=True,
    add_cls_token=True,
    n_reg_tokens=4,
    save_per=1,
    # dino_chk_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
    # existing_checkpoint="experiments/20260127_1554/best_model.pth",
)
# Multiscale training config
# IMG_L = 518
# CACHE = False
# cfg = Config(
#     experiment_name="dv3_coco_quarter_ms",
#     ds_type="otf_coco",
#     ds_path="../JAFAR/data/COCOStuff/dataset/images",
#     img_l=IMG_L,
#     model_type="plus_alibi",
#     vit_model_type=MODEL_LIST[4],
#     stride=16,
#     zero_pos_emb=True,
#     freeze_pos_emb=True,
#     alibi_slope_type="constant",
#     norm_alibi=True,
#     wrap_alibi=True,
#     jitter_mag=0.00,
#     n_epochs=1,
#     batch_size=32,
#     pretrained=True,
#     # channels_to_blank=[47, 113, 117, 359],
#     do_random_roll=False,
#     loss_type="cosine",
#     n_epochs_warmup=-1,
#     lr=1e-5,
#     add_cls_token=True,
#     n_reg_tokens=4,
#     dino_chk_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
#     existing_checkpoint="experiments/current/20260506_2004_dv3_coco/e3.pth",
#     save_per=1,
# )
print(cfg)

EXPR_PATH = f"experiments/current/{datetime.now().strftime('%Y%m%d_%H%M')}_{cfg.experiment_name}"
try:
    rmtree(EXPR_PATH)
except FileNotFoundError:
    pass
makedirs(EXPR_PATH, exist_ok=True)

writer = SummaryWriter(EXPR_PATH)
# writer.add_hparams(cfg.__dict__, {})
writer.add_text("desc", cfg.experiment_name)

DEVICE: str = "cuda:1"


train_ds, val_ds = get_ds(cfg, DEVICE)

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


# model = AlibiVitWrapper(MODEL_LIST[1], add_flash_attn=False, device=DEVICE)
# model.set_alibi_enabled(False)
model = get_model(
    cfg.model_type,
    cfg.alibi_slope_type,
    cfg.norm_alibi,
    cfg.wrap_alibi,
    cfg.n_epochs_warmup,
    cfg.freeze_pos_emb,
    cfg.zero_pos_emb,
    DEVICE,
    cfg.existing_checkpoint,
    cfg.vit_model_type,
    cfg.stride,
    cfg.dino_chk_path,
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


for epoch in range(cfg.n_epochs):
    if epoch == cfg.n_epochs_warmup:
        print("Enabling Alibi")
        model.set_alibi_enabled(True)

    train_loss = 0.0
    batch: torch.Tensor
    N_batches = len(train_dl)
    for i, batch in enumerate(train_dl):
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, training=True, device=DEVICE, dataset_type=cfg.ds_type
        )

        train_loss += loss
        if i % 50 == 0:
            print(f"Train batch {i}/{N_batches} | Loss: {loss:.4f}")

        # if i == N_batches // 4:
        #     break

    train_loss /= len(train_dl)
    val_loss = 0.0

    for j, batch in enumerate(val_dl):
        loss = feed_batch_get_loss(
            model, optimizer, loss_fn, batch, training=False, device=DEVICE, dataset_type=cfg.ds_type
        )
        val_loss += loss
        if j % 50 == 0:
            print(f"Train batch {j}/{N_batches} | Loss: {loss:.4f}")

    val_loss /= len(val_dl)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{EXPR_PATH}/best_model.pth")

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
