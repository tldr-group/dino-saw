import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms as T
from PIL import Image
import torchvision.transforms as v2
import numpy as np
from dinosaw.utils import load_image, resize_crop

from typing import Literal

Mode = Literal["train", "val"]


def get_paths(base_path: str, mode: Mode):
    # img_paths, target_paths = [], []

    match mode:
        case "train":
            img_paths = np.sort(
                glob.glob(f"{base_path}/ADEChallengeData2016/images/training/*.jpg")
            )
            target_paths = np.sort(
                glob.glob(
                    f"{base_path}/ADEChallengeData2016/annotations/training/*.png"
                )
            )
        case "val":
            img_paths = np.sort(
                glob.glob(f"{base_path}/ADEChallengeData2016/images/validation/*.jpg")
            )
            target_paths = np.sort(
                glob.glob(
                    f"{base_path}/ADEChallengeData2016/annotations/validation/*.png"
                )
            )

    return img_paths, target_paths


def to_ADE20K_label(mask: torch.Tensor):
    num_classes = 150
    label = torch.zeros((num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long)

    for class_ in range(num_classes):
        label[class_, :, :] = mask == class_
    return label


def load_ade20k_target(path: str, img_size: int):
    target = Image.open(path)
    transform = v2.Compose(
        [v2.Resize((img_size, img_size)), v2.CenterCrop((img_size, img_size))]
    )
    target_transformed = pil_to_tensor(transform(target)).squeeze()
    #print(target_transformed.max(), target_transformed.min())
    # label = to_ADE20K_label(target_transformed)
    return target_transformed.long()


class ADE20KDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.img_size = img_size

        self.img_paths, self.target_paths = get_paths(base_path=base_path, mode=mode)
        if len(self.img_paths) == 0:
            raise Exception(f"Dataset has 0 entrys")

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
            device_str="cpu",
        )[0].squeeze()

        target = load_ade20k_target(
            path=self.target_paths[index], img_size=self.img_size
        )

        return img.to(self.dtype), target
