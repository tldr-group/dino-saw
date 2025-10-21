import lightning as L
import torch
import polars as pl
import matplotlib.pyplot as plt
from utils import convert_image, to_norm_tensor, load_image, resize_crop, do_2D_pca, plot_losses
from PIL import Image
import io
from model import PEModel, get_sinusoid_encoding
from torch import nn, optim
import torch
from vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_MAP,
    FeatureType,
    MODEL_LIST,
)



class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs = pl.read_parquet('/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/train/*.parquet')
        self.targets = torch.load("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/Dataset/teacher/teacher_out.pt")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_bytes = self.imgs.row(index)[0]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        target = self.targets[index, ::]

        return convert_image(img, to_norm_tensor, device_str="cpu").squeeze().to(torch.float32), target.to(torch.float32)
    
def feed_batch_loss(model, opt, batch):
    model.train()
    opt.zero_grad()

    input, target = batch
    input, target = (
        input.to("cuda").to(torch.float32),
        target.to("cuda").to(torch.float32)
    )
    output = model.forward_features(input, make_2D=True)

    loss = torch.nn.functional.mse_loss(output, target)

    loss.backward()

    opt.step()

    input, target, output = (
        input.to("cpu"),
        target.to("cpu"),
        output.to("cpu")
    )

    return loss.item()

def vis(model: torch.nn.Module) -> None:
    img = load_image("/home/ab_aimd_anja_20884/Pawlowsky_Moritz/England/DINOMO/testing_dinov2/black_dog.jpg", resize_crop((224,224), (224,224)))[0]
    img = img.to("cuda").to(torch.float32)

    model.eval()
    output = model.forward_features(img, make_2D=True).squeeze()

    fig, axes = plt.subplots(1,2)
    x=0
    for ax, img in zip(axes.ravel(), [img.to("cpu").squeeze(), output.to("cpu").squeeze()]):
        if x ==1:
            img = do_2D_pca(img, n_components=3, post_norm="minmax")
            ax.imshow(img)
        else:
            img = (img - img.min()) / (img.max()-img.min())
            ax.imshow(img.transpose(0,2).transpose(0,1).float())
        x+=1

        
    fig.savefig("test.png")
    plt.close()

def main():
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=32, num_workers=40, pin_memory=True)

    # vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda").train()
        
    # new_pos_embedding = get_sinusoid_encoding(1369, 384).to(torch.float16) # needs to have the shape [1, 1369, 384]
    # new_pos_embedding

    # del vit.model.pos_embed
    # vit.model.pos_embed = new_pos_embedding.to("cuda")

    # N_EPOCHS, SAVE_PER = 10, 200

    # opt = torch.optim.Adam(vit.parameters(), lr=1e-3)
    # from tqdm import tqdm
    # train_losses, val_losses = [], []
    # best_val_loss = 1e10
    # for i in range(N_EPOCHS):
    #     epoch_loss = 0.0
    #     for batch in tqdm(loader):
    #         loss_val = feed_batch_loss(vit, opt, batch)
    #         epoch_loss += loss_val

    #         if i % SAVE_PER == 0:
    #             vis(vit)

    #             plot_losses(train_losses, None, "losses.png")

    #     print(f"[{i}/{N_EPOCHS}]: train={epoch_loss:.1f}")
    #     train_losses.append(epoch_loss)


    #     # scheduler.step(val_loss)

    #     # if val_loss < best_val_loss:
    #     #     obj = {"weights": net.state_dict(), "config": config_from_expriment(expr)}
    #     #     # todo: just save every 100 epochs?
    #     #     torch.save(net.state_dict(), f"{OUT_PATH}/best.pth")
    #     #     best_val_loss = val_loss


    trainer = L.Trainer(devices=1, max_epochs=10, gradient_clip_val=1.0) #, strategy="ddp"
    trainer.fit(model=PEModel(), train_dataloaders=loader)


if __name__ == "__main__":
    main()