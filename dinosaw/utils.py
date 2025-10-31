import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Literal
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


### Work from Ronan


NormType = Literal["minmax", "std", None]
norm_dict = {
    "minmax": MinMaxScaler(feature_range=(0, 1), clip=True, copy=False),
    "std": StandardScaler(copy=False),
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if len(arr.shape) == 4:
        arr = arr[0]
    return arr

# ========================= PCA STUFF =========================
def flatten(x: torch.Tensor | np.ndarray) -> np.ndarray:
    y: np.ndarray
    if type(x) == torch.Tensor:
        y = to_numpy(x)
    else:
        y = x  # type: ignore
    c, h, w = y.shape
    y = y.reshape((c, h * w))
    y = y.T
    return y


def do_pca(
    arr: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    # arr in shape (n_samples, n_features)
    if n_samples > -1:
        inds = np.arange(arr.shape[0])
        sample_inds = np.random.choice(inds, n_samples)
        train_data = arr[sample_inds]
    else:
        train_data = arr

    if pre_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[pre_norm]
        scaler.fit_transform(arr)
        # arr = scaler.transform(arr)
        # train_data = scaler.transform(train_data)

    pca = PCA(n_components=n_components)

    train_proj = pca.fit_transform(train_data)
    projection = pca.transform(arr)

    if post_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[post_norm]
        scaler.fit_transform(projection)
        # projection = scaler.transform(projection)
    return projection


def do_2D_pca(
    arr_2D: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    c, h, w = arr_2D.shape
    flat = flatten(arr_2D)
    proj = do_pca(flat, n_components, n_samples, pre_norm, post_norm)
    proj_2d = proj.reshape((h, w, n_components))
    return proj_2d

# ========================= INPUT TRANSFORMS =========================


MU, SIGMA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INV_MU = (-MU[0] / SIGMA[0], -MU[1] / SIGMA[1], -MU[2] / SIGMA[2])
INV_SIGMA = (1 / SIGMA[0], 1 / SIGMA[1], 1 / SIGMA[2])

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MU, std=SIGMA),
    ]
)

unnormalize = transforms.Normalize(
    mean=INV_MU,
    std=INV_SIGMA,
)


def closest_crop(h: int, w: int, patch_size: int = 14, to_tensor: bool = True) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose([transforms.CenterCrop((new_h, new_w)), to_norm_tensor])
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
            ]
        )
    return transform


def get_shortest_side_resize_dims(img_h: int, img_w: int, min_l: int) -> tuple[int, int]:
    if min(img_w, img_h) > min_l:
        sf = min(img_w / min_l, img_h / min_l)
    else:
        sf = max(min_l / img_w, min_l / img_h)
    return (int(max((img_h * sf), min_l)), int(max(img_w * sf, min_l)))


def resize_crop(resize_dims: tuple[int, int], crop_dims: tuple[int, int]) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dims),
            transforms.CenterCrop(crop_dims),
            to_norm_tensor,
        ]
    )
    return transform


def load_image(
    path: str,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
) -> tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor: torch.Tensor = convert_image(image, transform, to_gpu, to_half, batch)
    transformed_img = to_img(unnormalize(tensor.squeeze(0)))
    return tensor, transformed_img


def convert_image(
    img: Image.Image | torch.Tensor,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
    device_str: str = "cuda:0",
) -> torch.Tensor:
    tensor: torch.Tensor = transform(img)  # type: ignore
    if to_half:
        tensor = tensor.to(torch.float16)
    if to_gpu:
        tensor = tensor.to(device_str)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor




def plot_losses(train_loss: list[float], val_loss: list[float], out_path: str) -> None:
    epochs = np.arange(len(train_loss))
    plt.semilogy(epochs, train_loss, lw=2, label="train")
    #plt.semilogy(epochs, val_loss, lw=2, label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



### Work from Moritz

# ========================= translation TRANSFORMS =========================

def normalize(t_tens):
    return (t_tens - t_tens.min())/(t_tens.max() - t_tens.min())

def translate_tensor_wrap(x: torch.Tensor, step_x: int = 0, step_y: int = 0) -> torch.Tensor:
    """
    Circularly translate tensor using wrap boundary conditions.

    Args:
        x: Tensor of shape (B, C, H, W)
        step_x: horizontal shift in pixels; positive -> right
        step_y: vertical shift in pixels; positive -> down

    Returns:
        Tensor of same shape (B, C, H, W) translated with wrap (circular) boundary.
    """
    
    if not isinstance(step_x, (int,)) or not isinstance(step_y, (int,)):
        #print(type(step_x), type(step_y))
        raise ValueError("translate_tensor_wrap expects integer step_x and step_y for exact circular wrap.")
    # torch.roll uses order: dims (H dim index 2, W dim index 3)
    return torch.roll(x, shifts=(step_y, step_x), dims=(2, 3))


def revert_translation_wrap(feat: torch.Tensor, step_x: int = 0, step_y: int = 0, scale_factor: int = 14,
                            require_exact_division: bool = True) -> torch.Tensor:
    """
    Revert an image-space integer translation on a feature map using circular wrapping.
    The original translation is in *image pixels*; we convert to feature pixels by dividing by scale_factor.

    Args:
        feat: Tensor of shape (B, C, Hf, Wf)  (e.g. (1, 384, H//14, W//14))
        step_x: original image translation (pixels, int)
        step_y: original image translation (pixels, int)
        scale_factor: integer scaling between image and feature map spatial dims (e.g. 14)
        require_exact_division: if True, require that step_x/scale_factor and step_y/scale_factor are integers.
                                If False, the feature shift will be rounded to nearest integer.

    Returns:
        Tensor of same shape (B, C, Hf, Wf) where the translation is reverted using circular wrap.
    """
    if not isinstance(step_x, (int,)) or not isinstance(step_y, (int,)):
        raise ValueError("step_x and step_y must be integers (image pixels).")

    # convert to feature-map pixels (float)
    fx = step_x / float(scale_factor)
    fy = step_y / float(scale_factor)

    # check divisibility if requested
    if require_exact_division:
        if not fx.is_integer() or not fy.is_integer():
            raise ValueError(f"Translation {step_x},{step_y} not divisible by scale_factor={scale_factor}. "
                             "Resulting feature shift would be fractional. "
                             "Set require_exact_division=False to allow rounding.")
        fx = int(fx)
        fy = int(fy)
    else:
        # round to nearest integer feature shift
        fx = int(round(fx))
        fy = int(round(fy))

    # to revert the original translation we roll in the opposite direction
    return torch.roll(feat, shifts=(-fy, -fx), dims=(2, 3))

def translate_14(model, img, x_step, y_step, plot:bool=False):
    '''
    step refers to the step in the feature tensor
    '''
    if isinstance(x_step, np.ndarray) and isinstance(x_step, np.ndarray):
        x_step, y_step = x_step*14, y_step*14
        tensor_list = []
        for x_step_, y_step_ in zip(x_step, y_step):
            tensor_list.append(translate_tensor_wrap(img, step_x=int(x_step_), step_y=int(y_step_)))
        #tensor_list.reverse()
        batched_tensor = torch.concat(tensor_list, dim=0)
        #print(f"{batched_tensor.shape=}")

        #inference
        with torch.inference_mode(): 
            pred = model.forward_features(batched_tensor.float(), make_2D=True)#.to("cpu").to("cuda")
        #print(f"{pred.shape}")

        res_list = []
        for x_step_, y_step_, pred_ in zip (x_step, y_step, pred):
            res_list.append(revert_translation_wrap(pred_.unsqueeze(0), step_x=int(x_step_), step_y=int(y_step_), scale_factor=14, require_exact_division=False))#.detach().cpu().squeeze().transpose(2, 0).float()
        
        batched_res = torch.concat(res_list, dim=0)
        return batched_res
    else:
        x_step, y_step = x_step*14, y_step*14
        t_tens = translate_tensor_wrap(img, step_x=x_step, step_y=y_step)#.to("cpu")
        #print(t_tens.shape)

        # inference
        with torch.inference_mode(): 
            pred = model.forward_features(t_tens.float(), make_2D=True)#.to("cpu").to("cuda")
        #print(f"{pred.shape}")

        un_ten = revert_translation_wrap(pred, step_x=x_step, step_y=y_step, scale_factor=14, require_exact_division=False)#.detach().cpu().squeeze().transpose(2, 0).float()
    
        if plot:
            plt_t_tens= t_tens.clone().detach().cpu().squeeze().transpose(2, 0).float()
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            ax[0].imshow(normalize(img.detach().cpu().squeeze().transpose(2, 0).float()))
            ax[0].set_title("orig image")
            ax[1].imshow(normalize(plt_t_tens))
            ax[1].set_title("translated image")
            ax[2].imshow(do_2D_pca(pred.squeeze(), post_norm="minmax"))
            ax[2].set_title("translated_prediction")
            ax[3].imshow(do_2D_pca(un_ten.squeeze(), post_norm="minmax"))
            ax[3].set_title("untranslated prediction image")

        return un_ten


import time
def batched_translate_and_predict(model, img, x_step, y_step, scale_factor=14, device="cuda"):
    # ensure img has batch dim
    if img.dim() == 3:
        img = img.unsqueeze(0)   # (1, C, H, W)
    # move once to device
    img = img.to(device)

    # convert steps to ints and scale if needed
    x_steps = [int(x * 14) for x in x_step]  # or x_step already scaled
    y_steps = [int(y * 14) for y in y_step]

    start = time.perf_counter()
    # Build batched translated tensor (on device)
    translated_list = [torch.roll(img, shifts=(ys, xs), dims=(2, 3)) 
                       for xs, ys in zip(x_steps, y_steps)]
    #print("time for rolling =", time.perf_counter() - start)
    batched_tensor = torch.cat(translated_list, dim=0)   # shape (N, C, H, W)

    # Inference in single batch on GPU
    start = time.perf_counter()
    with torch.inference_mode():
        pred = model.forward_features(batched_tensor.float(), make_2D=True)  # stays on device
    print(pred.shape)
    #print("time for inference =", time.perf_counter() - start)

    # Revert translations on the predictions (inverse roll)
    # assume pred shape (N, featC, featH, featW)
    start = time.perf_counter()
    reverted_list = [torch.roll(p.unsqueeze(0), shifts=(-ys//14, -xs//14), dims=(2, 3))
                     for p, xs, ys in zip(pred, x_steps, y_steps)]
    batched_res = torch.cat(reverted_list, dim=0)  # (N, featC, featH, featW)
    #print("time for unrolling =", time.perf_counter() - start)

    # optional: move to cpu if you need
    return batched_res

def translate(model, img, factor=4, show_progress=True):
    from tqdm import tqdm

    from itertools import batched
    img_size=img.shape[-1]

    x = np.array(range(1, img_size//(factor*14)))*factor

    y = x.copy()
    x_steps, y_steps = np.meshgrid(x, y)
    #print()

    x_batches = list(batched(x_steps.flatten(), x.shape[0]))
    y_batches = list(batched(y_steps.flatten(), x.shape[0]))

    #print(x_batches, y_batches)



    res = torch.zeros((1, 384, img_size//14, img_size//14)).to("cuda")
    # for x, y in tqdm(zip(x_batches, y_batches)):
    #     #print(res.shape, translate_14(img, x_step=int(x), y_step=int(y)).)
    #     start = time.perf_counter()        
    #     res = batched_translate_and_predict(img, x_step=np.array(x), y_step=np.array(y)).sum(0, keepdim=True)
    #     #print("time until_res_add", time.perf_counter() - start)
    #     #print(intermediate.shape)

    if show_progress:
        for x, y in tqdm(zip(x_steps.flatten().astype(int), y_steps.flatten().astype(int))):
            #print(res.shape, translate_14(img, x_step=int(x), y_step=int(y)).)
            res += translate_14(model, img, x_step=int(x), y_step=int(y))#.cpu()
    else:
        for x, y in zip(x_steps.flatten().astype(int), y_steps.flatten().astype(int)):
            #print(res.shape, translate_14(img, x_step=int(x), y_step=int(y)).)
            res += translate_14(model, img, x_step=int(x), y_step=int(y))#.cpu()
    mean_res = res/x_steps.flatten().shape[0]
    return mean_res


# =========================== linear probing ==================================

def get_ramp(dir: Literal['lr', 'ud', 'diag', 'radial'], h, w) -> np.ndarray:
    """Generate a ramp mask in the specified direction."""
    if dir == 'lr':
        return np.tile(np.linspace(0, 1, w), (h, 1))
    elif dir == 'ud':
        return np.tile(np.linspace(0, 1, h), (w, 1)).T
    elif dir == 'diag':
        return np.linspace(0, 1, max(h, w))[:h, None] + np.linspace(0, 1, max(h, w))[:w]
    elif dir == 'radial':
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        cy, cx = h // 2, w // 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / r.max()
        return  1 - r_norm
    else:
        raise ValueError("Direction must be 'lr', 'ud', or 'diag'.")


def gen_sample_mask(shape: tuple[int, int], step: int = 4, cutoff_frac: float=1.0) -> np.ndarray:
    """Generate a sampling mask for an array, taking every `step`-th element."""
    cutoff_y, cutoff_x = int(shape[0] * cutoff_frac), int(shape[1] * cutoff_frac)
    mask = np.zeros(shape, dtype=bool)
    mask[0:cutoff_y:step, 0:cutoff_x:step] = True
    return mask


def linear_probe(feats: np.ndarray, target: np.ndarray, sample_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Train a linear regression model on sampled features to predict the target."""
    c, h, w = feats.shape
    X = feats[:, sample_mask].T  # Shape: (num_samples, c)
    y = target[sample_mask]       # Shape: (num_samples,)

    model = LinearRegression()
    model.fit(X, y)

    # Predict on all features
    X_full = feats.reshape(c, -1).T  # Shape: (h*w, c)
    y_pred = model.predict(X_full)    # Shape: (h*w,)

    score = model.score(X_full, target.flatten())  # R^2 score on training data
    return y_pred.reshape(h, w), float(score)



def probe(input_preds: list, remove_channels: list, titles: list, ramp='diag', mask_step=6, mask_cutoff_frac=0.8):
    input_preds_, remove_channels_, titles_ = input_preds.copy(), remove_channels.copy(), titles.copy()
    N_COLS = len(input_preds_)+1
    WIDTH=4
    input_np = []
    for inp, remove in zip(input_preds_, remove_channels_):
        if remove:
            inp[:, [47, 113, 117, 359], :, :] = 0
        input_np.append(to_numpy(inp))
    
    c, h, w = input_np[0].shape
    sample_mask = gen_sample_mask((h, w), step=mask_step, cutoff_frac=mask_cutoff_frac)
    ramp = get_ramp(ramp, h=h, w=w)

    input_np.insert(0, ramp)
    titles_.insert(0, "Ramp")
    
    fig, axs = plt.subplots(1, N_COLS, figsize=(WIDTH*N_COLS, WIDTH))
    for arr, ax, title in zip(input_np, axs, titles_):
        if title != 'Ramp':
            print(arr.shape)
            res, score = linear_probe(arr, ramp, sample_mask)
            title += f'\n (R²: {score:.3f})'
        else:
            res = arr
            title += f'\n'
        ax.imshow(res)
        ax.axis('off')
        ax.set_title(title)