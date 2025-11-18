import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from PIL import Image
from os import listdir

from dinosaw.utils import closest_crop, convert_image, to_norm_tensor

from typing import Literal

tr = closest_crop(224, 224, 14)


class EmbeddingDataset(Dataset):
    def __init__(self, root_dir: str, split: Literal["train", "val"]):
        img_dir = f"{root_dir}/{split}/imgs/"
        embed_dir = f"{root_dir}/{split}/embeddings/"
        self.img_paths = sorted([f"{img_dir}/{img}" for img in listdir(img_dir)])
        self.embedding_paths = sorted([f"{embed_dir}/{embed}" for embed in listdir(embed_dir)])

        self.transform = to_norm_tensor

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, index):
        pil_img = Image.open(f"{self.img_paths[index]}").convert("RGB")
        # img_tensor = TF.pil_to_tensor(img)
        # img_tensor = img_tensor.to(torch.float32)

        img_tensor = convert_image(pil_img, tr, True, False, device_str="cpu").squeeze()
        # img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        embedding = torch.load(self.embedding_paths[index], weights_only=True).squeeze()
        return img_tensor, embedding
