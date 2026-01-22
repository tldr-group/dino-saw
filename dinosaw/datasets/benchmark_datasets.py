import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import numpy as np
from dinosaw.utils import load_image, resize_crop
import glob


from typing import Literal

Mode = Literal["train", "val"]


class VOC_Dataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        set_255_0: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_path = base_path
        self.mode = mode
        self.img_size = img_size
        self.set_255_0 = set_255_0
        self.dtype = dtype

        self.img_paths, self.target_paths = self.get_paths()

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        # loading images
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
            device_str="cpu",
        )[0].squeeze()

        target = self.load_voc_target(
            self.target_paths[index],
            img_size=self.img_size,
            set_255_0=self.set_255_0,
        )

        return img.to(self.dtype), target  # .to(self.dtype)

    def get_paths(self):
        match self.mode:
            case "train":
                txt_file = "/train.txt"
            case "val":
                txt_file = "/val.txt"
            case _:
                raise Exception(f"Unkown mode {_}")

        img_paths, target_paths = [], []
        for line in open(self.base_path + txt_file, "r").readlines():
            img_paths.append(self.base_path + "/JPEGImages/" + line.strip() + ".jpg")
            target_paths.append(
                self.base_path + "/SegmentationClass/" + line.strip() + ".png"
            )

        return img_paths, target_paths  # [0:2]

    def load_voc_target(self, path: str, img_size: int, set_255_0: bool = False):
        img = Image.open(path)
        transform = T.Compose(
            [
                v2.Resize((img_size, img_size)),
                v2.CenterCrop((img_size, img_size)),
            ]
        )
        img_transformed = torch.tensor(np.array(transform(img)))

        if set_255_0:
            img_transformed[img_transformed == 255] = 0

        # target = to_VOC_label(img_transformed)

        target = img_transformed

        return target.long()

    def to_VOC_label(self, mask: torch.Tensor):
        num_classes = 21
        if len(mask.shape) == 2:
            label = torch.zeros(
                (num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[class_, :, :] = mask == class_
        elif len(mask.shape) == 3:
            label = torch.zeros(
                (mask.shape[0], num_classes, mask.shape[-2], mask.shape[-1]),
                dtype=torch.long,
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[:, class_, :, :] = mask == class_
        return label


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

        target = self.load_ade20k_target(
            path=self.target_paths[index], img_size=self.img_size
        )

        return img.to(self.dtype), target

    def get_paths(self, base_path: str, mode: Mode):
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
                    glob.glob(
                        f"{base_path}/ADEChallengeData2016/images/validation/*.jpg"
                    )
                )
                target_paths = np.sort(
                    glob.glob(
                        f"{base_path}/ADEChallengeData2016/annotations/validation/*.png"
                    )
                )

        return img_paths, target_paths

    def to_ADE20K_label(self, mask: torch.Tensor):
        num_classes = 150
        label = torch.zeros(
            (num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long
        )

        for class_ in range(num_classes):
            label[class_, :, :] = mask == class_
        return label

    def load_ade20k_target(self, path: str, img_size: int):
        target = Image.open(path)
        transform = v2.Compose(
            [v2.Resize((img_size, img_size)), v2.CenterCrop((img_size, img_size))]
        )
        target_transformed = pil_to_tensor(transform(target)).squeeze()
        # print(target_transformed.max(), target_transformed.min())
        # label = to_ADE20K_label(target_transformed)
        return target_transformed.long()


if __name__ == "__main__":
    ds = VOC_Dataset(
        "/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
        mode="train",
        set_255_0=True,
    )
