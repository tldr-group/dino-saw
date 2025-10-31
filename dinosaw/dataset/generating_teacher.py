import polars as pl
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
from utils import convert_image, to_norm_tensor
from tqdm import tqdm
from translation import translate

torch.cuda.empty_cache()

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.df = pl.read_parquet('/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/train/*.parquet')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_bytes = self.df.row(index)[0]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        return convert_image(img, to_norm_tensor, device_str="cpu").squeeze()
    
from vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_MAP,
    FeatureType,
    MODEL_LIST,
)


vit_wrapper = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda")

loader = torch.utils.data.DataLoader(Dataset(), batch_size=1, num_workers=20, pin_memory=True)
with torch.inference_mode(): 
    res = []
    for image in tqdm(loader):
        #cleaned_channels = vit_wrapper.forward_features(batch.to("cuda", non_blocking=True), make_2D=True).cpu()
        #cleaned_channels[:,[47, 113, 117, 359], ...] = 0 # setting the channels with positinoal bias to zero

        res.append(translate(vit_wrapper, image.to("cuda", non_blocking=True), factor=1, show_progress=False))
        image.to("cpu")

torch.save(torch.cat(res), "teacher_out_trans_homo.pt")