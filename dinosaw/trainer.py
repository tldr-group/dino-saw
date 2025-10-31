import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import polars as pl
import matplotlib.pyplot as plt
from utils import convert_image, to_norm_tensor, load_image, resize_crop, do_2D_pca, plot_losses
from PIL import Image
import io
from models.lightning_model import PEModel, get_sinusoid_encoding
from torch import nn, optim
import torch
from models.vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_MAP,
    FeatureType,
    MODEL_LIST,
)



class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dtype = torch.float32
        self.imgs = pl.read_parquet('/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/train/*.parquet', parallel="row_groups")
        self.targets = torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/teacher/teacher_out_homo.pt")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_bytes = self.imgs.row(index)[0]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        target = self.targets[index, ::]

        return convert_image(img, to_norm_tensor, device_str="cpu").squeeze().to(self.dtype), target.to(self.dtype)
    
class DatasetVal(torch.utils.data.Dataset):
    def __init__(self):
        self.dtype = torch.float32
        self.img = [load_image("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/black_dog.jpg", resize_crop((224,224), (224,224)))[0], load_image("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/default_image.jpg", resize_crop((224,224), (224,224)))[0]]
        self.target = [torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/black_dog_target.pt"), torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/val/micro_struct_target.pt")]

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        return self.img[index].squeeze().to(self.dtype), self.target[index].to(self.dtype)
    
    
# def feed_batch_loss(model, opt, batch):
#     model.train()
#     opt.zero_grad()

#     input, target = batch
#     input, target = (
#         input.to("cuda").to(torch.float32),
#         target.to("cuda").to(torch.float32)
#     )
#     output = model.forward_features(input, make_2D=True)

#     loss = torch.nn.functional.mse_loss(output, target)

#     loss.backward()

#     opt.step()

#     input, target, output = (
#         input.to("cpu"),
#         target.to("cpu"),
#         output.to("cpu")
#     )

#     return loss.item()

# def vis(model: torch.nn.Module) -> None:
#     img = load_image("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/black_dog.jpg", resize_crop((224,224), (224,224)))[0]
#     img = img.to("cuda").to(torch.float32)

#     model.eval()
#     output = model.forward_features(img, make_2D=True).squeeze()

#     fig, axes = plt.subplots(1,2)
#     x=0
#     for ax, img in zip(axes.ravel(), [img.to("cpu").squeeze(), output.to("cpu").squeeze()]):
#         if x ==1:
#             img = do_2D_pca(img, n_components=3, post_norm="minmax")
#             ax.imshow(img)
#         else:
#             img = (img - img.min()) / (img.max()-img.min())
#             ax.imshow(img.transpose(0,2).transpose(0,1).float())
#         x+=1

        
#     fig.savefig("test.png")
#     plt.close()

def main():
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=32, num_workers=56, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(DatasetVal(), batch_size=2)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/',
        filename='test-{epoch:02d}-{val_loss:.2f}',
        mode="min"
    )
    
    trainer = L.Trainer(devices=1, max_epochs=70, gradient_clip_val=1.0, callbacks=checkpoint_callback) #, strategy="ddp"
    trainer.fit(model=PEModel(), train_dataloaders=loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()