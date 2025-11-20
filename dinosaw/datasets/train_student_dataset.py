import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from PIL import Image
import io
from dinosaw.utils import convert_image, to_norm_tensor, load_image, resize_crop
import glob
import numpy as np

from dinosaw.utils import closest_crop, convert_image, to_norm_tensor

tr = closest_crop(224, 224, 14)


class DatasetTrainStudent(torch.utils.data.Dataset):
    def __init__(self):
        self.dtype = torch.float32
        self.imgs = pl.read_parquet(
            "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/train/*.parquet", parallel="row_groups"
        )
        self.targets = torch.load(
            "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/teacher/teacher_out_trans_homo.pt"
        ).to("cpu")

    def __getitem__(self, index):
        img_bytes = self.imgs.row(index)[0]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        target = self.targets[index]

        return convert_image(img, to_norm_tensor, device_str="cpu").squeeze().to(self.dtype), target.to(self.dtype)


class DatasetValStudent(torch.utils.data.Dataset):
    def __init__(self):
        self.dtype = torch.float32
        self.img = [
            load_image(
                "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/black_dog.jpg",
                resize_crop((224, 224), (224, 224)),
            )[0],
            load_image(
                "/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/default_image.jpg",
                resize_crop((224, 224), (224, 224)),
            )[0],
        ]
        self.target = [
            torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/black_dog_target.pt"),
            torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/micro_struct_target.pt"),
        ]

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, index):
        return self.img[index].squeeze().to(self.dtype), self.target[index].to(self.dtype)


class GenericDatasetStudent(torch.utils.data.Dataset):
    def __init__(self, base_path, split, dtype=torch.float32):
        self.img_paths = np.sort(glob.glob(f"{base_path}/{split}/imgs/*.png"))
        self.target_paths = np.sort(glob.glob(f"{base_path}/{split}/embeddings/*.pt"))
        self.targets = [torch.load(p) for p in self.target_paths]
        self.imgs = [load_image(p, resize_crop((224, 224), (224, 224)), device_str="cpu")[0] for p in self.img_paths]
        self.dtype = dtype

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.targets[index]  # torch.load(self.target_paths[index])
        return img.squeeze().to(self.dtype), target.to(self.dtype)
