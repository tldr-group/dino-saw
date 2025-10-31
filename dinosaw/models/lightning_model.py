import lightning as L
from torch import nn, optim
from torch.nn import functional as F
import torchvision.transforms.functional as v2
import torch
from torchvision.utils import make_grid
from vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_MAP,
    FeatureType,
    MODEL_LIST,
)
import matplotlib.pyplot as plt
from utils import do_2D_pca
import numpy as np
from overwriting_methods import _pos_embed_no_pos

def get_sinusoid_encoding(num_tokens, token_len):
    """ Make Sinusoid Encoding Table

        Args:
            num_tokens (int): number of tokens
            token_len (int): length of a token
            
        Returns:
            (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class PEModel(L.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_model(self):
        self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda").train()
        
        # new_pos_embedding = get_sinusoid_encoding(1369, 384).to(torch.float16) # needs to have the shape [1, 1369, 384]
        # new_pos_embedding
        # device = torch.get_device(self.vit.model.pos_embed)
        # del self.vit.model.pos_embed
        # self.vit.model.pos_embed = new_pos_embedding.to(device)

        import types
        self.vit._pos_embed = types.MethodType(_pos_embed_no_pos, self.vit)


    def training_step(self, batch):
        input, target = batch
        #input, target = input.to(torch.float32), target.to(torch.float32)
        #input = torch.nan_to_num(input, nan=0.0, posinf=1e6, neginf=-1e6)
        output = self.vit.forward_features(input, make_2D=True)
        #print(output, target.shape)

        loss = nn.functional.mse_loss(output, target)
        #print(loss)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)
        loss = nn.functional.mse_loss(output, target)
        self.log("val_loss", loss, sync_dist=True)
        
        tensorboard = self.logger.experiment
        #self.visualize(input, output)
        tensorboard.add_image("intermediate_output", self.gen_vis_grid(input, output, target), self.current_epoch)
        return loss
    
    def forward(self, input):
        return self.vit.forward_features(input, make_2D=True)
    
    def gen_vis_grid(self, input, output, target):
        res = []
        for input, output, target in zip(input, output, target):
            orig = F.normalize(input.to("cpu").squeeze())
            pred = v2.resize(torch.tensor(do_2D_pca(output.to("cpu").squeeze(), n_components=3, post_norm="minmax")).transpose(0,2).transpose(1,2), input.shape[-2:-1])
            target = v2.resize(torch.tensor(do_2D_pca(target.to("cpu").squeeze(), n_components=3, post_norm="minmax")).transpose(0,2).transpose(1,2), input.shape[-2:-1])
            res.append(torch.stack([orig, pred, target]))
        res = torch.concat(res, dim=0)
        return make_grid(
            res,
            3
        )

    def visualize(self, img, prediction):
        fig, axes = plt.subplots(1,2)
        x=0
        imgs = []
        for ax, img in zip(axes.ravel(), [img.to("cpu").squeeze(), prediction.to("cpu").squeeze()]):
            if x ==1:
                img = do_2D_pca(img, n_components=3, post_norm="minmax")
                ax.imshow(img)
            else:
                img = (img - img.min()) / (img.max()-img.min())
                ax.imshow(img.transpose(0,2).transpose(0,1).float())
            x+=1

            
        fig.savefig("test.png")
        plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer