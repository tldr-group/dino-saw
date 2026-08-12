import torch
from PVW import WrapperRegistry
from dinosaw.utils import load_image, do_2D_pca, to_numpy, closest_resize

from PIL import Image
import matplotlib.pyplot as plt

from os import makedirs


model = WrapperRegistry.build(
    "alibi_coco_dinov2_s",
    device="cuda:0",
    checkpoint_path="models/checkpoints/trained/alibi_coco_dv2_vits14_reg4.pth",
)
# to run a dinov3 model, for example, uncomment this block
# all that's changed is the name of the model (see dinosaw.wrappers.register_models for the list) and the checkpoint path
# model = WrapperRegistry.build(
#     "alibi_dinov3_s+",
#     device="cuda:0",
#     checkpoint_path="models/checkpoints/trained/alibi_coco_dv3_vits16_reg4.pth",
#     model_conf_path="models/dinov3",  # this is only used for dv3 models
# )
model.to("cuda:0")
model.eval()

img_fname = "default_image.jpg"
_img = Image.open(f"tests/{img_fname}").convert("RGB")


SF = 1
tr = closest_resize(SF * _img.height, SF * _img.width, 14)
img, _ = load_image(f"tests/{img_fname}", tr, to_half=False, device_str="cuda:0")
with torch.no_grad():
    emb = model.forward_features(img, make_2D=True)

emb_np = to_numpy(emb.squeeze(0))

pca_emb = do_2D_pca(emb_np, n_components=3, post_norm="minmax")
print(pca_emb.shape)
makedirs("tmp", exist_ok=True)
plt.imsave("tmp/feats.png", pca_emb)
