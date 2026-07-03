import torch
from PVW import WrapperRegistry
from dinosaw.utils import load_image, do_2D_pca, to_numpy, closest_resize

from PIL import Image
import matplotlib.pyplot as plt

# problem: all your checkpoints are model.foo,
model = WrapperRegistry.build(
    "alibi_dinov3_s",
    device="cuda:0",
    checkpoint_path="models/checkpoints/trained/alibi_dv3_ms.pth",
    model_conf_path="models/dinov3",
)
model.to("cuda:0")
model.eval()

img_fname = "000394.jpg"
_img = Image.open(f"images/micro/{img_fname}").convert("RGB")


SF = 1
tr = closest_resize(SF * _img.height, SF * _img.width, 14)
img, _ = load_image(f"images/micro/{img_fname}", tr, to_half=False, device_str="cuda:0")
with torch.no_grad():
    emb = model.forward_features(img, make_2D=True)

emb_np = to_numpy(emb.squeeze(0))

pca_emb = do_2D_pca(emb_np, n_components=3, post_norm="minmax")
print(pca_emb.shape)
plt.imsave("tmp/feats.png", pca_emb)
