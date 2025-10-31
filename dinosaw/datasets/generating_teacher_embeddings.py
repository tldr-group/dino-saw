import torch
from tqdm import tqdm
from dinosaw.utils import translate
from dinosaw.datasets import TeacherDataset
from dinosaw.models.vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_LIST,
)


def generate_teacher_dataset(save_path):
    vit_wrapper = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda")

    loader = torch.utils.data.DataLoader(TeacherDataset(), batch_size=1, num_workers=20, pin_memory=True)
    with torch.inference_mode(): 
        res = []
        for image in tqdm(loader):
            #cleaned_channels = vit_wrapper.forward_features(batch.to("cuda", non_blocking=True), make_2D=True).cpu()
            #cleaned_channels[:,[47, 113, 117, 359], ...] = 0 # setting the channels with positinoal bias to zero

            res.append(translate(vit_wrapper, image.to("cuda", non_blocking=True), factor=1, show_progress=False))
            image.to("cpu")

    torch.save(torch.cat(res), save_path)