import torch


#### not working from gpt
# def no_pos_forward_features(self, x):
#     """
#     Take the typical timm VisionTransformer.forward_features skeleton
#     and remove the x = x + self.pos_embed step.
#     This function intentionally keeps cls_token behavior; remove cls handling
#     if you want no cls token either.
#     """
#     B = x.shape[0]
#     # patch embedding -> (B, n_patches, embed_dim)
#     x = self.patch_embed(x)

#     print(x.shape) # b, h, w, c

#     if x.ndim == 4:
#         # flatten spatial dims and move channel to last dim
#         x = x.transpose(1, 3)      # (B, c, w, h)
#         x = x.flatten(2)           # (B, C, w*h)
#         x = x.transpose(1, 2)
#         print(x.shape)

#     # cls token handling (typical timm ViT)
#     if hasattr(self, 'cls_token'):
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#     # ---- skip positional embedding addition ----
#     # do not do: x = x + self.pos_embed
#     x = self.pos_drop(x)
#     # transformer blocks
#     for blk in self.blocks:
#         x = blk(x)
#     x = self.norm(x)
#     # return same thing the original forward_features would return
#     return x


# working from gpt
def _pos_embed_no_pos(self, x):
    """
    Replacement for _pos_embed that:
    - Flattens (B, H, W, C) -> (B, N, C)
    - Prepends reg and cls tokens if present
    - SKIPS adding absolute positional embeddings
    """
    B = x.shape[0]

    # flatten spatial dims
    if x.ndim == 4:  # (B, H, W, C)
        x = x.reshape(B, -1, x.shape[-1])  # -> (B, N, C)

    # add reg tokens if model has them
    if hasattr(self, "reg_tokens") and self.reg_tokens is not None:
        reg_tokens = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat((reg_tokens, x), dim=1)

    # add cls token if model has one
    if hasattr(self, "cls_token") and self.cls_token is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    # ---- do NOT add pos_embed ----
    return x