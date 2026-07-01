import torch
from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper, AlibiDV3Wrapper
from dinosaw.utils import load_image, do_2D_pca, to_numpy, closest_resize

from PIL import Image
import matplotlib.pyplot as plt

# model = AlibiVitWrapper(
#     # MODEL_LIST[1],
#     stride=14,
#     # add_flash_attn=False,
#     device="cuda:0",
#     slope_type="constant",
#     normalize=True,
#     wrap=True,
#     # chk_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
#     # conf_path="dinov3",
#     add_flash_attn=False,
#     n_reg_tokens=4,
# )
model = AlibiDV3Wrapper(
    # MODEL_LIST[1],
    stride=16,
    # add_flash_attn=False,
    device="cuda:0",
    slope_type="constant",
    normalize=True,
    wrap=True,
    chk_path="trained_models/dinov3_vits_patch16_plus_reg4.pth",
    conf_path="dinov3",
    add_flash_attn=False,
    n_reg_tokens=4,
    skip_overwrite=False,  # don't convert attention, since the loaded weights are already for an ALiBi model
)
model.to("cuda:0")

# weights = torch.load("trained_models/alibi_homog_dv2_vits14_reg.pth", weights_only=True, map_location="cuda:0")
# model.load_state_dict(weights)
# model.half()
model.eval()
# print(model.distance_matrix.device)

# model = PretrainedViTWrapper(MODEL_LIST[1], stride=4, add_flash_attn=False, device="cuda:0")

# img_fname = "NMC_2D_less_wide_crop.png"
img_fname = "000394.jpg"
# _img = Image.open(f"images/micro/{img_fname}").convert("RGB")
# img_fname = "diff_shapes_widest.png"
_img = Image.open(f"images/micro/{img_fname}").convert("RGB")
# _img = _img.resize((512, 512))

SF = 2
tr = closest_resize(SF * _img.height, SF * _img.width, 14)
img, _ = load_image(f"images/micro/{img_fname}", tr, to_half=False, device_str="cuda:0")
with torch.no_grad():
    emb = model.forward_features(img, make_2D=True)

emb_np = to_numpy(emb.squeeze(0))
print(emb_np.shape)

channels_to_blank = [47, 113, 117, 359]
emb_np[channels_to_blank, :, :] = 0

pca_emb = do_2D_pca(emb_np, n_components=3, post_norm="minmax")
print(pca_emb.shape)
plt.imsave("tmp/emb_518_.png", pca_emb)


# alibi_matrix = model.model.blocks[0].attn.distance_matrix.cpu().numpy()
# print(alibi_matrix.shape)
# plt.imsave("tmp/alibi_518.png", alibi_matrix, cmap="hot")
