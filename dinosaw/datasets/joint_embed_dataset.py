import torch
from torchvision.transforms import Compose  # type: ignore
from torch.utils.data import Dataset

from random import randint, choice

from glob import glob

from dinosaw.utils import closest_resize_crop, load_image
from dinosaw.wrappers import PretrainedViTWrapper


from functools import partial
from typing import Literal, Callable, TypeAlias

Tr: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

tr = closest_resize_crop(224, 14)


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(x, dims=(dim,))


def rot(x: torch.Tensor, angle: int) -> torch.Tensor:
    k = angle // 90
    return torch.rot90(x, k=k, dims=(-2, -1))


def shift(x: torch.Tensor, s: int, dir: tuple[int, int]) -> torch.Tensor:
    return torch.roll(x, (dir[0] * s, dir[1] * s), dims=(-2, -1))


def nop(x: torch.Tensor) -> torch.Tensor:
    return x


class OTFEmbeddingDataset(Dataset):
    def __init__(
        self,
        embed_model: PretrainedViTWrapper,
        base_path: str,
        split: Literal["train", "val"],
        transform: Compose = tr,
        dtype=torch.float32,
        device: str = "cuda",
        fname_file_path: str | None = None,
        norm_feats: bool = False,
        squeeze_batch_dim_from_image: bool = True,
        squeeze_batch_dim_from_embed: bool = True,
        channels_to_blank: list[int] = [],
        channel_dup: bool = False,
        _do_random_roll: bool = False,
    ):
        self.embed_model = embed_model
        self.img_paths = self.get_img_paths(base_path, fname_file_path)
        self.transform = transform
        self.dtype = dtype
        self.device = device

        self.norm_feats = norm_feats
        self.squeeze_batch_dim_from_embed = squeeze_batch_dim_from_embed
        self.squeeze_batch_dim_from_image = squeeze_batch_dim_from_image

        self.channels_to_blank = channels_to_blank
        self.channel_dup = channel_dup

    def get_img_paths(self, base_path: str, fname_file_path: str | None) -> list[str]:
        if fname_file_path is not None:
            with open(fname_file_path, "r") as f:
                img_fnames = [line.strip().split(";")[0] for line in f.readlines()]
            img_paths = [f"{base_path}/{fname}" for fname in img_fnames]
        else:
            img_paths = sorted(glob(f"{base_path}/*.jpg"))
        return img_paths

    def operate_on_channels(
        self, embed: torch.Tensor, channels_to_change: list[int], channel_dup: bool
    ) -> torch.Tensor:
        if len(channels_to_change) == 0:
            return embed

        for ch in channels_to_change:
            if channel_dup:  # duplicate from previous channel
                embed[ch, ...] = embed[ch - 1, ...]
            else:  # if blanking set to 0
                embed[ch, ...] = 0.0

        return embed

    def load_image(
        self,
        path: str,
        transform: Compose,
        squeeze_batch_dim_from_image: bool = True,
    ) -> torch.Tensor:
        img, _ = load_image(path, transform, to_gpu=True, to_half=False, device_str=self.device)
        if squeeze_batch_dim_from_image:
            img = img.squeeze(0)
        return img

    def __len__(self) -> int:
        return len(self.img_paths)

    @torch.no_grad()
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.img_paths[index]
        img = self.load_image(path, self.transform, self.squeeze_batch_dim_from_image)

        vit_emb = self.embed_model.forward_features(img.unsqueeze(0), True)
        target_emb = self.operate_on_channels(vit_emb.squeeze(0), self.channels_to_blank, False)
        return img.to(self.dtype), target_emb.to(self.dtype)


class JointEmbeddingDataset(OTFEmbeddingDataset):
    def get_transform(self) -> tuple[Tr, Tr]:
        tr_types = ("flip", "rot", "shift", "none", "none")
        tr_type = choice(tr_types)

        if tr_type == "flip":
            which = choice(("h", "v"))
            if which == "h":
                return partial(flip, dim=-1), partial(flip, dim=-1)
            else:
                return partial(flip, dim=-2), partial(flip, dim=-2)
        elif tr_type == "rot":
            angle = choice((90, 180, 270))
            return partial(rot, angle=angle), partial(rot, angle=angle)
        elif tr_type == "shift":
            dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            dir = choice(dirs)
            s = choice([i for i in range(1, 8)])
            return partial(shift, s=14 * s, dir=dir), partial(shift, s=s, dir=dir)
        elif tr_type == "none":
            return nop, nop
        else:
            raise ValueError(f"Unknown transform type: {tr_type}")

    def __len__(self) -> int:
        return len(self.img_paths)

    @torch.no_grad()
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Tr]:
        path = self.img_paths[index]
        img = self.load_image(path, self.transform, self.squeeze_batch_dim_from_image)

        img_tr, embed_tr = self.get_transform()

        fwd_img = img_tr(img)
        # enforce Tr(ALiBi(I)) = ViT(Tr(I))
        vit_emb = self.embed_model.forward_features(fwd_img.unsqueeze(0), True)
        target_emb = self.operate_on_channels(vit_emb.squeeze(0), self.channels_to_blank, False)
        return img.to(self.dtype), target_emb.to(self.dtype), embed_tr
