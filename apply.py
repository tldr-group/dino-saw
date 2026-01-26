import torch
from dinosaw.models.vit_wrapper import MODEL_LIST, PretrainedViTWrapper, AlibiVitWrapper
from dinosaw.utils import load_image, do_2D_pca, to_numpy, closest_resize

from PIL import Image
import matplotlib.pyplot as plt

model = AlibiVitWrapper(
    MODEL_LIST[1],
    stride=14,
    add_flash_attn=False,
    device="cuda:0",
    slope_type="constant",
    normalize=True,
    wrap=True,
)

weights = torch.load("trained_models/alibi_dv2_vits14_reg.pth", weights_only=True, map_location="cuda:0")
model.load_state_dict(weights)
model.eval()

# model = PretrainedViTWrapper(MODEL_LIST[1], stride=4, add_flash_attn=False, device="cuda:0")

img_fname = "snowcat.png"
_img = Image.open(f"images/{img_fname}").convert("RGB")
_img = _img.resize((512, 512))


tr = closest_resize(_img.height, _img.width, 14)
img, _ = load_image(f"images/{img_fname}", tr, to_half=False, device_str="cuda:0")
with torch.no_grad():
    emb = model.forward_features(img, make_2D=True)

emb_np = to_numpy(emb.squeeze(0))
print(emb_np.shape)

pca_emb = do_2D_pca(emb_np, n_components=3, post_norm="minmax")
print(pca_emb.shape)
plt.imsave("tmp/emb_518.png", pca_emb)


alibi_matrix = model.model.blocks[0].attn.distance_matrix.cpu().numpy()
print(alibi_matrix.shape)
plt.imsave("tmp/alibi_518.png", alibi_matrix, cmap="hot")
