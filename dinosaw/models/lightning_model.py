import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as v2
from dinosaw.models.vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_LIST,
)
from .example_overwrite import AlibiBlock
from dinosaw.utils import do_2D_pca, normalize
from .overwriting_methods import _pos_embed_no_pos
from functools import partial

# hyperparams for LoRA
lora_r = 12
lora_alpha = 4
lora_dropout = 0.05
lora_qkv = True
lora_proj = True
lora_mlp = True


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

class LoRALayer(torch.nn.Module):
    '''
        Layer that implements LoRA
    '''
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        '''
            implements addition between LoRA layer and linear layer for PEFT
        '''
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def add_lora(model):
        for param in model.model.parameters():
            param.requires_grad = False
        
        assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

        for layer in model.model.blocks:
            if lora_qkv:
                layer.attn.qkv = assign_lora(layer.attn.qkv)
            if lora_proj:
                layer.attn.proj = assign_lora(layer.attn.proj)
            if lora_mlp:
                layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
                layer.mlp.fc2 = assign_lora(layer.mlp.fc2)

class PEModel(L.LightningModule):
    def __init__(self, 
                 use_alibi: bool =  False, 
                 remove_pos_embed = False, 
                 loss_func: str =   "mse"
                ):
        super().__init__()
        self.use_alibi = use_alibi
        self.last_validation_batch = None
        self.remove_pos_embed = remove_pos_embed
        self.loss_func = loss_func

    def configure_model(self):
        if self.use_alibi:
            self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, block_fn=AlibiBlock, device="cuda").train()
        else:
            self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda").train()
        
        if self.remove_pos_embed:
            import types
            self.vit._pos_embed = types.MethodType(_pos_embed_no_pos, self.vit)
            self.vit.model.pos_embed = torch.nn.Parameter(torch.zeros_like(self.vit.model.pos_embed), requires_grad=False) #setting pos_encoding to zero without gradient

        #add_lora(self.vit)

        # new_pos_embedding = get_sinusoid_encoding(1369, 384).to(torch.float16) # needs to have the shape [1, 1369, 384]
        # new_pos_embedding
        # device = torch.get_device(self.vit.model.pos_embed)
        # del self.vit.model.pos_embed
        # self.vit.model.pos_embed = new_pos_embedding.to(device)

        # import types
        # self.vit._pos_embed = types.MethodType(_pos_embed_no_pos, self.vit)

    


    def training_step(self, batch):
        input, target = batch
        #input, target = input.to(torch.float32), target.to(torch.float32)
        #input = torch.nan_to_num(input, nan=0.0, posinf=1e6, neginf=-1e6)
        output = self.vit.forward_features(input, make_2D=True)
        #print(output, target.shape)

        
        loss = self.calc_loss(output, target)
        #print(loss)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)

        loss = self.calc_loss(output, target)
        self.log("val_loss", loss, sync_dist=True)
        
        
        #self.visualize(input, output)
        self.last_validation_batch = (input, output, target)
        return loss
    
    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        input, output, target = self.last_validation_batch
        tensorboard.add_image("intermediate_output", self.gen_vis_grid(input[:2], output[:2], target[:2]), self.current_epoch)
    
    def forward(self, input):
        return self.vit.forward_features(input, make_2D=True)
    
    def gen_vis_grid(self, input, output, target):
        res = []
        for input, output, target in zip(input, output, target):
            orig = normalize(input.to("cpu").squeeze())
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

    def calc_loss(self, output, target):
        '''
            calculates loss function for the model
        '''
        if self.loss_func == "cosine_similarity":
            loss = 1-torch.cosine_similarity(output, target, dim=-3)
        elif self.loss_func == "cosine_embedding":
            loss = nn.functional.cosine_embedding_loss(output.flatten(1), target.squeeze().flatten(1), torch.ones((input.shape[0])).to("cuda"))
        else:
            loss = nn.functional.mse_loss(output, target.squeeze())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer