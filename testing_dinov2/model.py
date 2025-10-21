import lightning as L
from torch import nn, optim
import torch
from vit_wrapper import (
    PretrainedViTWrapper,
    MODEL_MAP,
    FeatureType,
    MODEL_LIST,
)
import numpy as np

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
        self.vit = PretrainedViTWrapper(MODEL_LIST[1], add_flash_attn=False, device="cuda").train()
        
        new_pos_embedding = get_sinusoid_encoding(1369, 384).to(torch.float16) # needs to have the shape [1, 1369, 384]
        new_pos_embedding

        del self.vit.model.pos_embed
        self.vit.model.pos_embed = new_pos_embedding.to("cuda")


    def training_step(self, batch):
        input, target = batch
        #input, target = input.to(torch.float32), target.to(torch.float32)
        #input = torch.nan_to_num(input, nan=0.0, posinf=1e6, neginf=-1e6)
        output = self.vit.forward_features(input, make_2D=True)
        #print(output, target.shape)

        loss = nn.functional.mse_loss(output, target)
        #print(loss)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer