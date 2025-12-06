import torch
from torchvision.transforms import Compose  # type: ignore
from torch.utils.data import Dataset

from glob import glob

from dinosaw.utils import load_image, closest_crop

from typing import Literal

tr = closest_crop(224, 224, 14)


class HomogenizedEmbeddingDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        split: Literal["train", "val"],
        transform: Compose = tr,
        dtype=torch.float32,
        store_in_memory: bool = True,
        norm_feats: bool = False,
        squeeze_batch_dim_from_image: bool = True,
        squeeze_batch_dim_from_embed: bool = True,
    ):
        self.img_paths = sorted(glob(f"{base_path}/{split}/imgs/*.png"))
        self.target_paths = sorted(glob(f"{base_path}/{split}/embeddings/*.pt"))

        self.transform = transform
        self.dtype = dtype
        self.norm_feats = norm_feats
        self.squeeze_batch_dim_from_embed = squeeze_batch_dim_from_embed
        self.squeeze_batch_dim_from_image = squeeze_batch_dim_from_image

        self.store_in_memory = store_in_memory

        if store_in_memory:
            self.imgs = [self.load_image(p, self.transform, squeeze_batch_dim_from_image) for p in self.img_paths]
            self.targets = [self.load_embed(p, squeeze_batch_dim_from_embed, norm_feats) for p in self.target_paths]

        self._initial_checks()

    def _initial_checks(self) -> None:
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )

        indices = [0, 100, 200, 555]
        for idx in indices:
            img_filename = self.img_paths[idx].split("/")[-1].replace(".png", "")
            target_filename = self.target_paths[idx].split("/")[-1].replace(".pt", "")
            if img_filename != target_filename:
                raise ValueError(f"Filename mismatch at index {idx}: {img_filename}.png vs {target_filename}.pt")

    def load_embed(
        self, path: str, squeeze_batch_dim_from_embed: bool = True, norm_feats: bool = False
    ) -> torch.Tensor:
        embed = torch.load(path, weights_only=True)
        if norm_feats:
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)
        if squeeze_batch_dim_from_embed:
            embed = embed.squeeze(0)
        return embed

    def load_image(self, path: str, transform: Compose, squeeze_batch_dim_from_image: bool = True) -> torch.Tensor:
        img, _ = load_image(path, transform, to_gpu=False, to_half=False, device_str="cpu")
        if squeeze_batch_dim_from_image:
            img = img.squeeze(0)
        return img

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.store_in_memory:
            img = self.imgs[index]
            target = self.targets[index]
        else:
            img = self.load_image(self.img_paths[index], self.transform, self.squeeze_batch_dim_from_image)
            target = self.load_embed(self.target_paths[index], self.squeeze_batch_dim_from_embed, self.norm_feats)
        return img.to(self.dtype), target.to(self.dtype)


if __name__ == "__main__":
    ds = HomogenizedEmbeddingDataset("Dataset/IN_reduced_224", "train", store_in_memory=False)
    print(len(ds))
    img, target = ds[0]
    print(img.shape, target.shape)
