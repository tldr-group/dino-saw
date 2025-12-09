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
    AlibiVitWrapper,
    MODEL_LIST,
)
from .alibi import AlibiSlopeType
#from .alibi import AlibiBlock
from dinosaw.utils import do_2D_pca, normalize
from .overwriting_methods import _pos_embed_no_pos
from functools import partial
from copy import deepcopy



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

def unfreeze_alibi_and_norms(model, unfreeze_layernorms=True, unfreeze_other_patterns=None):
    """
    Unfreeze alibi parameters and optionally LayerNorm params and other user-specified patterns.
    unfreeze_other_patterns: list of substrings in parameter names to unfreeze (e.g. ['head', 'proj'])
    """
    if unfreeze_other_patterns is None:
        unfreeze_other_patterns = []

    for name, p in model.named_parameters():
        # if "attn" in name:
        #     p.requires_grad = True
        if unfreeze_layernorms and (".norm" in name or "ln" in name or "layernorm" in name.lower() or "layer_norm" in name.lower() or "ls1" in name.lower() or "ls2" in name.lower()): #or ("mlp" in name.lower() and ("blocks.11" in name.lower() or "blocks.10" in name.lower()))
            p.requires_grad = True
        else:
            for pat in unfreeze_other_patterns:
                if pat in name:
                    p.requires_grad = True
                    break

class PEModel(L.LightningModule):
    def __init__(self, 
                 use_alibi: bool =  False, 
                 remove_pos_embed: bool = False, 
                 loss_func: str =   "mse",
                 slope_type: AlibiSlopeType = "fixed",
                 normalize:bool = True,
                 wrap: bool = True
                ):
        super().__init__()
        self.save_hyperparameters()
        self.use_alibi = use_alibi
        self.last_validation_batch = None
        self.remove_pos_embed = remove_pos_embed
        self.loss_func = loss_func
        self.slope_type = slope_type
        self.normalize = normalize
        self.wrap = wrap

    def configure_model(self):
        if self.use_alibi:
            #self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, block_fn=AlibiBlock, device="cuda").train()
            self.vit = AlibiVitWrapper(
                            MODEL_LIST[1], 
                            add_flash_attn=False, 
                            device="cuda", 
                            slope_type=self.slope_type,
                            normalize=self.normalize,
                            wrap=self.wrap
                            )
        else:
            self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda").train()
        
        # if self.remove_pos_embed:
        #     import types
        #     self.vit._pos_embed = types.MethodType(_pos_embed_no_pos, self.vit)
        #     self.vit.model.pos_embed = torch.nn.Parameter(torch.zeros_like(self.vit.model.pos_embed), requires_grad=False) #setting pos_encoding to zero without gradient

        self.vit.model.pos_embed.requires_grad = False
        
        if self.remove_pos_embed:
            self.vit.model.pos_embed.data.zero_()
        
        self.base_pos_embed = self.vit.model.pos_embed
        self.base_pos_embed.requires_grad = False

        # last_block = self.vit.model.blocks[-1]

        # new_block = deepcopy(last_block)
        # self.vit.model.blocks.append(new_block)

        # for p in self.vit.parameters():
        #     p.requires_grad = False

        # unfreeze_alibi_and_norms(model = self.vit, unfreeze_layernorms=True, unfreeze_other_patterns=["attn"])


    def training_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)#, attn_mask=self.attn_mask)
        
        loss = self.calc_loss(output, target)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, sync_dist=True)
        return loss


    def on_validation_epoch_start(self):
        # setting validation pos_embedding to zeros
        self.vit.model.pos_embed = torch.nn.Parameter(torch.zeros_like(self.vit.model.pos_embed), requires_grad=False)
        return super().on_validation_epoch_start()

    def validation_step(self, batch):
        input, target = batch
        output = self.vit.forward_features(input, make_2D=True)#, attn_mask=self.attn_mask)
        #print( self.vit.model.pos_embed)
        loss = self.calc_loss(output, target)
        self.log("val_loss", loss, sync_dist=True)
        
        
        #self.visualize(input, output)
        self.last_validation_batch = (input, output, target)
        return loss
    
    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        input, output, target = self.last_validation_batch
        self.vit.model.pos_embed = self.base_pos_embed
        tensorboard.add_image("intermediate_output", self.gen_vis_grid(input[:2], output[:2], target[:2]), self.current_epoch)
    
    # def on_validation_end(self):
    #     # returning it to normal
    #     return super().on_validation_end()
    
    def forward(self, input):
        return self.vit.forward_features(input, make_2D=True)#, attn_mask=attn_mask)
    
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
        if self.loss_func == "cosine_embedding":
            loss = nn.functional.cosine_embedding_loss(output.flatten(1), target.squeeze().flatten(1), torch.ones((output.shape[0])).to("cuda"))
        else:
            loss = nn.functional.mse_loss(output, target.squeeze())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-6)
        return optimizer