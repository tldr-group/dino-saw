import torch
from PIL import Image
import polars as pl
import io
from dinosaw.utils import convert_image, to_norm_tensor


class TeacherDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.df = pl.read_parquet('/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/train/*.parquet')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_bytes = self.df.row(index)[0]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        return convert_image(img, to_norm_tensor, device_str="cpu").squeeze()