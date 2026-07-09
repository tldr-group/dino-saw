import torch
from torch import nn, optim
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from os import listdir

from dinosaw.datasets.train_student_dataset import HomogenizedEmbeddingDataset
from dinosaw.datasets.joint_embed_dataset import JointEmbeddingDataset, OTFEmbeddingDataset
from dinosaw.alibi_logic import AlibiSlopeType
from dinosaw.linear_probe import do_linear_probe
from dinosaw.wrappers import MODEL_LIST, PretrainedViTWrapper
from dinosaw.wrappers.alibi import add_alibi, replace_pe_with_sincos
from PVW.modifications import replace_pos_embed
from PVW.factory import BackboneConfig, BackboneRegistry
from PVW.types import ARCHS_TO_TIMM_IDS, ARCHS_TO_LOCAL_IDS
from dinosaw.utils import closest_resize, closest_resize_crop, to_numpy

from functools import partial
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
    conf_path: str | None = None
    vit_model_type: str = MODEL_LIST[1]
    stride: int = 14
    pretrained: bool = True

    add_cls_token: bool = True
    n_reg_tokens: int = 4

    alibi_slope_type: AlibiSlopeType = "constant"
    norm_alibi: bool = True
    wrap_alibi: bool = True
    zero_pos_emb: bool = False
    jitter_mag: float = 0.0

    channels_to_blank: list[int] = field(default_factory=lambda: [])
    channel_dup: bool = False
    do_random_roll: bool = False

    n_epochs: int | float = 10
    batch_size: int = 256
    lr: float = 1e-4
    optim: Optims = "AdamW"
    loss_type: Losses = "MSE"

    save_per: int = 2

    existing_checkpoint: str | None = None
    backbone_checkpoint: str | None = None

    lp_interval: float = -1
    lp_homog_micros_path: str = "data/linear_probe/homog_micros"


# TODO: make the wrapeprs take in various params; cache slope type etc on AlibiVitWrapper
def get_model(
    model_type: ModelType,
    alibi_slope_type: AlibiSlopeType,
    norm_alibi: bool,
    wrap_alibi: bool,
    zero_pos_emb: bool,
    device: str | torch.device,
    existing_checkpoint: str | None = None,
    backbone_checkpoint: str | None = None,
    vit_model_type: str = MODEL_LIST[1],
    stride: int = 14,
    conf_path: str | None = None,
    pretrained: bool = True,
    add_cls_token: bool = True,
    n_reg_tokens: int = 4,
    jitter_mag: float = 0.0,
) -> PretrainedViTWrapper:

    is_dinov3 = "dv3" in vit_model_type or "dinov3" in vit_model_type

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
    pretrained = False if is_dinov3 else pretrained
    # if existing supplied, use that, otherwise use backbone
    checkpoint_to_load = existing_checkpoint if existing_checkpoint else backbone_checkpoint
    print(f"get_model {vit_model_type} checkpoint: {checkpoint_to_load}")

    cfg = BackboneConfig(
        backbone_type=backbone_type,
        model_arch=arch_name,  # type: ignore
        pretrained=pretrained,
        checkpoint_path=checkpoint_to_load,
        model_conf_path=conf_path,
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

    if zero_pos_emb:
        cfg.modifications.append(partial(replace_pos_embed, new_pos_embed=None, requires_grad=False))

    vit = BackboneRegistry.build(cfg)
    model = PretrainedViTWrapper(vit=vit, device=device)

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


def get_ds(cfg: Config, device: str, cache: bool = False) -> tuple[Dataset, Dataset]:

    train_ds: Dataset
    val_ds: Dataset
    if cfg.ds_type == "direct":
        tr = closest_resize(cfg.img_l, cfg.img_l, 14)

        train_ds = HomogenizedEmbeddingDataset(
            cfg.ds_path,
            "train",
            transform=tr,
            store_in_memory=cache,
            channels_to_blank=cfg.channels_to_blank,
            channel_dup=cfg.channel_dup,
            do_random_roll=cfg.do_random_roll,
        )
        val_ds = HomogenizedEmbeddingDataset(
            cfg.ds_path,
            "val",
            transform=tr,
            store_in_memory=cache,
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
            False,
            device,
            pretrained=True,
            existing_checkpoint=None,
            backbone_checkpoint=cfg.backbone_checkpoint,
            stride=cfg.stride,
            vit_model_type=cfg.vit_model_type,
            conf_path=cfg.conf_path,
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


def serialize_hparams(cfg_dict):
    serialized = {}
    for k, v in cfg_dict.items():
        if isinstance(v, (int, float, bool, str)):
            serialized[k] = v
        elif v is None:
            serialized[k] = "None"
        elif isinstance(v, (list, tuple)):
            serialized[k] = ",".join(map(str, v))
        else:
            serialized[k] = str(v)
    return serialized


def get_linear_probe_images(path: str) -> list[Image.Image]:
    lp_images = []
    img_files = sorted(listdir(path))
    for f in img_files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            img_path = f"{path}/{f}"
            try:
                img = Image.open(img_path).convert("RGB")
                lp_images.append(img)
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
    return lp_images


def evaluate_linear_probe(model: PretrainedViTWrapper, probe_images: list[Image.Image], device):
    model.eval()

    # 1. Extract features for all images
    features = []
    with torch.no_grad():
        for img in probe_images:
            emb = model.forward_features(img, make_2D=True)
            emb_np = to_numpy(emb.squeeze(0))
            # transpose to channel-last (h, w, c) as expected by do_linear_probe
            emb_np = np.transpose(emb_np, (1, 2, 0))
            features.append(emb_np)

    # 2. Perform linear probing for each ramp type: 'lr', 'ud', 'diag', 'radial'
    ramp_types = ["lr", "ud", "diag", "radial"]
    results_r2 = {}

    MASK_CUTOFF_FRAC = 1.0
    STEP = 6
    RANDOM_MASK = True

    for ramp in ramp_types:
        scores = []
        for feats in features:
            res = do_linear_probe(
                feats,
                ramp,
                probe_by_channel=False,
                mask_step=STEP,
                mask_cutoff_frac=MASK_CUTOFF_FRAC,
                random_mask=RANDOM_MASK,
                regressor="ridge",
            )
            scores.append(res["stack_r_squared"])
        results_r2[ramp] = float(np.mean(scores)) if scores else 0.0

    return results_r2
