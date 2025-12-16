import os
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dinosaw.models import PEModel
from dinosaw.utils import load_image, resize_crop, normalize

from typing import Literal

Mode = Literal["train", "val"]


def get_paths(base_path: str, mode: Mode):
    match mode:
        case "train":
            txt_file = "/train.txt"
        case "val":
            txt_file = "/val.txt"
        case _:
            raise Exception(f"Unkown mode {_}")

    img_paths, target_paths = [], []
    for line in open(base_path + txt_file, "r").readlines():
        img_paths.append(base_path + "/JPEGImages/" + line.strip() + ".jpg")
        target_paths.append(base_path + "/SegmentationClass/" + line.strip() + ".png")

    return img_paths, target_paths  # [0:2]


def to_VOC_label(mask: torch.Tensor):
    num_classes = 21
    label = torch.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=torch.long)

    for class_ in range(num_classes):  # 21 == num classes
        label[class_, :, :] = mask == class_

    return label


def load_voc_target(path: str, img_size: int, set_255_0: bool = False):
    img = Image.open(path)
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.CenterCrop((img_size, img_size)),
        ]
    )
    img_transformed = torch.tensor(np.array(transform(img)))

    if set_255_0:
        img_transformed[img_transformed == 255] = 0

    target = to_VOC_label(img_transformed)

    return target


class VOC_Dataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        set_255_0: bool = False,
        dtype: torch.dtype = torch.float32,
        load_in_memory: bool = False,
        checkpoint_path: str | None = None,
    ):
        self.set_255_0 = set_255_0
        self.img_size = img_size
        self.mode = mode
        self.dtype = dtype
        self.load_in_memory = load_in_memory

        self.img_paths, self.target_paths = get_paths(
            base_path=base_path, mode=self.mode
        )
        if self.load_in_memory:
            self.model = PEModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path
            ).half()
            self.imgs, self.targets = self.load_feats_and_targets()
        super().__init__()

    def load_feats_and_targets(self):
        imgs, targets = [], []
        for img_path, target_path in zip(self.img_paths, self.target_paths):
            with torch.inference_mode():
                imgs.append(
                    self.model(
                        load_image(
                            img_path,
                            transform=resize_crop(
                                (self.img_size, self.img_size),
                                (self.img_size, self.img_size),
                            ),
                            device_str="cuda:1",
                        )[0]
                    )
                    .cpu()
                    .squeeze()
                )
            targets.append(
                load_voc_target(
                    target_path,
                    img_size=self.img_size,
                    set_255_0=self.set_255_0,
                )
            )
        return imgs, targets

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        if self.load_in_memory:
            img, target = self.imgs[index], self.targets[index]
        else:
            img = load_image(
                self.img_paths[index],
                transform=resize_crop(
                    (self.img_size, self.img_size), (self.img_size, self.img_size)
                ),
                device_str="cpu",
            )[0].squeeze()

            target = load_voc_target(
                self.target_paths[index],
                img_size=self.img_size,
                set_255_0=self.set_255_0,
            )

        return img.to(self.dtype), target.to(self.dtype)


def main():
    ds = VOC_Dataset(
        "/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
        mode="train",
        set_255_0=True,
    )


if __name__ == "__main__":
    main()
