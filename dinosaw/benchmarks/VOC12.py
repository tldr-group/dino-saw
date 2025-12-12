import os
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
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

    return img_paths, target_paths


def load_voc_target(path: str, img_size):
    img = Image.open(
        "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/VOC/SegmentationClass/2007_000032.png"
    )
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
        ]
    )
    img_transformed = transform(img)
    return img_transformed.to(torch.float16)


class VOC_Dataset(Dataset):
    def __init__(self, base_path: str, mode: Mode, img_size: int = 518):
        self.img_size = img_size
        self.mode = mode

        self.img_paths, self.target_paths = get_paths(
            base_path=base_path, mode=self.mode
        )

        super().__init__()

    def __getitem__(self, index):
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
        )[0]

        target = load_voc_target(self.target_paths[index], img_size=self.img_size)

        print(target.shape)
        plt.imshow(target.squeeze().float().cpu())
        plt.show()
        return img, target


def main():
    ds = VOC_Dataset(
        "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset_/VOC",
        mode="train",
    )

    ds[0]


if __name__ == "__main__":
    main()
