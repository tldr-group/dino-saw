from shutil import copy
from os import listdir, mkdir
from random import choice, seed
from PIL import Image

seed(42)
PATH_TO_ANNOTATED_SEM_DS = "/media/ronan/T7/micrograph_imagenet/data/annotated_sem/processed"
OUT_PATH = "notebooks/paper_figures_pdf/data/linear_probe/homog_micros"

try:
    mkdir(OUT_PATH)
except FileExistsError:
    pass


def longest_side_resize(img: Image.Image, target: int) -> Image.Image:
    longest_side = max(img.size)
    if longest_side == target:
        return img

    scale = target / longest_side
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    return img.resize(new_size, resample=Image.LANCZOS)

def center_crop(img: Image.Image, target: int) -> Image.Image:
    width, height = img.size
    left = (width - target) / 2
    top = (height - target) / 2
    right = (width + target) / 2
    bottom = (height + target) / 2
    return img.crop((left, top, right, bottom))

HOMOG_FOLDER_NAMES = ["Powder", "Fibres", "Particles", "Nanowires", "Porous_Sponge", "Films_Coated_Surface"]

N_IMGS = 100

n_imgs_per_folder = [N_IMGS // len(HOMOG_FOLDER_NAMES) for _ in HOMOG_FOLDER_NAMES]
remainder = N_IMGS % len(HOMOG_FOLDER_NAMES)
n_imgs_per_folder[-1] += remainder

i = 0
for j, folder_name in enumerate(HOMOG_FOLDER_NAMES):
    folder_path = f"{PATH_TO_ANNOTATED_SEM_DS}/{folder_name}"
    imgs = listdir(folder_path)
    selected_imgs = [choice(imgs) for _ in range(n_imgs_per_folder[j])]
    for img_name in selected_imgs:
        img = Image.open(f"{folder_path}/{img_name}").convert('L')
        img = longest_side_resize(img, 1024)  # Resize the longest side to 256 pixels
        shortest_side = min(img.size)
        img = center_crop(img, shortest_side)
        img.save(f"{OUT_PATH}/{i}_{img_name}")
        i += 1
print(f"Copied {i} images to {OUT_PATH}")
