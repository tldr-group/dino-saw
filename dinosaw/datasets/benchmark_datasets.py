import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import numpy as np
from dinosaw.utils import load_image, resize_crop
import dinosaw.utils as utils
import geobench
import glob
import cv2

# ADE20K
import random
import os


from typing import Literal

Mode = Literal["train", "val"]
Satellites = Literal[
    "m-cashew-plant",
    "m-pv4ger-seg",
    "m-SA-crop-type",
    "m-chesapeake",
    "m-NeonTree",
    "m-nz-cattle",
]


class VOC_Dataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        set_255_0: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_path = base_path
        self.mode = mode
        self.img_size = img_size
        self.set_255_0 = set_255_0
        self.dtype = dtype

        self.img_paths, self.target_paths = self.get_paths()

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        # loading images
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
            device_str="cpu",
        )[0].squeeze()

        target = self.load_voc_target(
            self.target_paths[index],
            img_size=self.img_size,
            set_255_0=self.set_255_0,
        )

        return img.to(self.dtype), target  # .to(self.dtype)

    def get_paths(self):
        match self.mode:
            case "train":
                txt_file = "/train.txt"
            case "val":
                txt_file = "/val.txt"
            case _:
                raise Exception(f"Unkown mode {_}")

        img_paths, target_paths = [], []
        for line in open(self.base_path + txt_file, "r").readlines():
            img_paths.append(self.base_path + "/JPEGImages/" + line.strip() + ".jpg")
            target_paths.append(
                self.base_path + "/SegmentationClass/" + line.strip() + ".png"
            )

        return img_paths, target_paths  # [0:2]

    def load_voc_target(self, path: str, img_size: int, set_255_0: bool = False):
        img = Image.open(path)
        transform = T.Compose(
            [
                v2.Resize((img_size, img_size)),
                v2.CenterCrop((img_size, img_size)),
                v2.ToTensor(),
            ]
        )
        img_transformed = torch.tensor(np.array(transform(img)))

        if set_255_0:
            img_transformed[img_transformed == 255] = 0

        # target = to_VOC_label(img_transformed)

        target = img_transformed

        return target.long()

    def to_VOC_label(self, mask: torch.Tensor):
        num_classes = 21
        if len(mask.shape) == 2:
            label = torch.zeros(
                (num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[class_, :, :] = mask == class_
        elif len(mask.shape) == 3:
            label = torch.zeros(
                (mask.shape[0], num_classes, mask.shape[-2], mask.shape[-1]),
                dtype=torch.long,
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[:, class_, :, :] = mask == class_
        return label


class VOC07_Dataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        set_255_0: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_path = base_path
        self.mode = mode
        self.img_size = img_size
        self.set_255_0 = set_255_0
        self.dtype = dtype

        self.img_paths, self.target_paths = self.get_paths()

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        # loading images
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
            device_str="cpu",
        )[0].squeeze()

        target = self.load_voc_target(
            self.target_paths[index],
            img_size=self.img_size,
            set_255_0=self.set_255_0,
        ).squeeze()

        return img.to(self.dtype), target  # .to(self.dtype)

    def get_paths(self):
        match self.mode:
            case "train":
                txt_file = "/train.txt"
            case "val":
                txt_file = "/val.txt"
            case _:
                raise Exception(f"Unkown mode {_}")

        img_paths, target_paths = [], []
        for line in open(self.base_path + txt_file, "r").readlines():
            img_paths.append(self.base_path + "/JPEGImages/" + line.strip() + ".jpg")
            target_paths.append(
                self.base_path + "/SegmentationClass/" + line.strip() + ".png"
            )

        return img_paths, target_paths  # [0:2]

    def load_voc_target(self, path: str, img_size: int, set_255_0: bool = False):
        img = Image.open(path)
        transform = T.Compose(
            [
                v2.Resize((img_size, img_size)),
                v2.CenterCrop((img_size, img_size)),
                v2.ToDtype(torch.long, scale=True),
            ]
        )
        img_transformed = torch.tensor(np.array(transform(img)))

        if set_255_0:
            img_transformed[img_transformed == 255] = 0

        # target = to_VOC_label(img_transformed)

        target = img_transformed

        return target.long()

    def to_VOC_label(self, mask: torch.Tensor):
        num_classes = 21
        if len(mask.shape) == 2:
            label = torch.zeros(
                (num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[class_, :, :] = mask == class_
        elif len(mask.shape) == 3:
            label = torch.zeros(
                (mask.shape[0], num_classes, mask.shape[-2], mask.shape[-1]),
                dtype=torch.long,
            )

            for class_ in range(num_classes):  # 21 == num classes
                label[:, class_, :, :] = mask == class_
        return label


class ADE20KDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        img_size: int = 518,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.img_size = img_size

        self.img_paths, self.target_paths = get_paths(base_path=base_path, mode=mode)
        if len(self.img_paths) == 0:
            raise Exception(f"Dataset has 0 entrys")

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        img = load_image(
            self.img_paths[index],
            transform=resize_crop(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            ),
            device_str="cpu",
        )[0].squeeze()

        target = self.load_ade20k_target(
            path=self.target_paths[index], img_size=self.img_size
        )

        return img.to(self.dtype), target

    def get_paths(self, base_path: str, mode: Mode):
        match mode:
            case "train":
                img_paths = np.sort(
                    glob.glob(f"{base_path}/ADEChallengeData2016/images/training/*.jpg")
                )
                target_paths = np.sort(
                    glob.glob(
                        f"{base_path}/ADEChallengeData2016/annotations/training/*.png"
                    )
                )
            case "val":
                img_paths = np.sort(
                    glob.glob(
                        f"{base_path}/ADEChallengeData2016/images/validation/*.jpg"
                    )
                )
                target_paths = np.sort(
                    glob.glob(
                        f"{base_path}/ADEChallengeData2016/annotations/validation/*.png"
                    )
                )

        return img_paths, target_paths

    def to_ADE20K_label(self, mask: torch.Tensor):
        num_classes = 150
        label = torch.zeros(
            (num_classes, mask.shape[-2], mask.shape[-1]), dtype=torch.long
        )

        for class_ in range(num_classes):
            label[class_, :, :] = mask == class_
        return label

    def load_ade20k_target(self, path: str, img_size: int):
        target = Image.open(path)
        transform = v2.Compose(
            [v2.Resize((img_size, img_size)), v2.CenterCrop((img_size, img_size))]
        )
        target_transformed = pil_to_tensor(transform(target)).squeeze()
        # print(target_transformed.max(), target_transformed.min())
        # label = to_ADE20K_label(target_transformed)
        return target_transformed.long()


class DatasetADE_NEW(Dataset):
    def __init__(self, base_path: str, mode: Mode, img_size=518, max_sample: int = -1):
        txt = f"{base_path}/ADE20K_object150_{mode}.txt"
        # mode_folder = "training" if mode == "train" else "validation"
        self.root_img = f"{base_path}/images"
        self.root_seg = f"{base_path}/annotations"
        self.imgSize = img_size
        self.segSize = img_size
        self.is_train = mode == "train"

        # mean and std
        # self.img_transform = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])

        self.list_sample = [x.rstrip() for x in open(txt, "r")]

        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print("# samples: {}".format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h, w = img.shape[-2], img.shape[-1]

        if is_train:
            # random scale
            scale = random.random() + 0.5  # 0.5-1.5
            scale = max(scale, 1.0 * cropSize / (min(h, w) - 1))
        else:
            # scale to crop size
            scale = 1.0 * cropSize / (min(h, w) - 1)

        img_scale = torch.nn.functional.interpolate(
            img, scale_factor=scale, mode="bilinear"
        )
        seg_scale = torch.nn.functional.interpolate(
            seg.unsqueeze(0), scale_factor=scale, mode="nearest"
        ).squeeze()

        h_s, w_s = img_scale.shape[-2], img_scale.shape[-1]
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
        else:
            # center crop
            x1 = (w_s - cropSize) // 2
            y1 = (h_s - cropSize) // 2
        # print(f"{seg_scale.shape=}, {img_scale.shape=}")
        img_crop = img_scale[:, :, y1 : y1 + cropSize, x1 : x1 + cropSize]
        seg_crop = seg_scale[y1 : y1 + cropSize, x1 : x1 + cropSize]
        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img.flip(-1)
        seg_flip = seg.flip(-1)
        return img_flip, seg_flip

    def __getitem__(self, index):
        img_basename = self.list_sample[index]
        path_img = os.path.join(self.root_img, img_basename)
        path_seg = os.path.join(self.root_seg, img_basename.replace(".jpg", ".png"))

        assert os.path.exists(path_img), "[{}] does not exist".format(path_img)
        assert os.path.exists(path_seg), "[{}] does not exist".format(path_seg)

        # load image and label

        img = v2.functional.to_tensor(Image.open(path_img).convert("RGB")).unsqueeze(0)
        seg = torch.tensor(cv2.imread(path_seg, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)
        # print(f"{seg.shape=}, {img.shape=}")
        # print(seg.max())
        assert img.ndim == 4
        assert seg.ndim == 3
        assert img.shape[-1] == seg.shape[-1]
        assert img.shape[-2] == seg.shape[-2]

        # random scale, crop, flip
        if self.imgSize > 0:
            img, seg = self._scale_and_crop(img, seg, self.imgSize, self.is_train)
            if random.choice([-1, 1]) > 0:
                img, seg = self._flip(img, seg)

        # image to float
        img = img.to(torch.float32)
        image = img.squeeze()

        # if self.segSize > 0:
        #     seg = imresize(seg, (self.segSize, self.segSize), interp="nearest")

        # label to int from -1 to 149
        segmentation = seg.long() - 1

        # to torch tensor
        # image = torch.from_numpy(img)
        # segmentation = torch.from_numpy(seg)
        # except Exception as e:
        #     print("Failed loading image/segmentation [{}]: {}".format(path_img, e))
        #     # dummy data
        #     image = torch.zeros(3, self.imgSize, self.imgSize)
        #     segmentation = -1 * torch.ones(self.segSize, self.segSize).long()
        #     return image, segmentation, img_basename

        # substracted by mean and divided by std
        # image = self.img_transform(image)

        if len(segmentation.shape) != 2 or len(image.shape) != 3:
            raise Exception(
                f"{segmentation.shape=} != 2 or {image.shape=} != 3 for {img_basename}"
            )

        return image, segmentation  # , img_basename

    def __len__(self):
        return len(self.list_sample)


class GeoBenchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        benchmark_name: Satellites,
        mode: Mode,
        size=518,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.size = size

        self.benchmark_name = benchmark_name

        for task in geobench.task_iterator(benchmark_name="segmentation_v1.0"):
            if task.dataset_name == benchmark_name:
                print(f"Task {task.dataset_name}:\n  {task}\n")
                self.dataset = task.get_dataset(
                    split=mode, band_names=("red", "green", "blue")
                )

        # shape = self.dataset[0].label.data.shape
        self.tr = v2.Compose(
            [
                utils.closest_resize(self.size, self.size, to_tensor=False),
                v2.Normalize(mean=utils.MU, std=utils.SIGMA),
            ]
        )
        sub_h: int = self.size % 14
        sub_w: int = self.size % 14
        new_h, new_w = self.size - sub_h, self.size - sub_w
        self.target_tr = v2.Compose(
            [
                v2.Resize(
                    (new_h, new_w), interpolation=v2.InterpolationMode.NEAREST_EXACT
                )
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def get_target(self, index):
        target = utils.convert_image(
            torch.tensor(self.dataset[index].label.data).unsqueeze(0),
            self.target_tr,
            to_half=False,
            device_str="cpu",
        ).squeeze()

        return target.cpu()

    def get_image(self, index):
        bands = []
        for band in self.dataset[index].bands:
            if band.band_info.name[-5:].strip() in ("- Red", "Red", "Green", "Blue"):
                bands.append(torch.tensor(band.data))
                # print(band.band_info.name)

        # print(f"{bands=}")

        # flipping to right RGB color orientation
        if self.benchmark_name in ("m-cashew-plant", "m-SA-crop-type"):
            image = utils.convert_image(
                torch.stack(bands).flip(0).float(),
                self.tr,
                to_half=False,
                device_str="cpu",
            )
        else:
            image = utils.convert_image(
                torch.stack(bands).float(), self.tr, to_half=False, device_str="cpu"
            )

        return image.squeeze().cpu()

    def __getitem__(self, index):
        image = self.get_image(index)
        target = self.get_target(index)

        return image.to(
            self.dtype
        ), target.long() - 1  # shifting targets to have ignored index at -1


class GF7(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        mode: Mode,
        size=518,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.size = size
        self.mode = mode
        self.base_path = base_path

        self.img_paths, self.target_paths = self.get_paths()

        self.tr = v2.Compose(
            [utils.closest_resize(self.size, self.size, to_tensor=True)]
        )
        sub_h: int = self.size % 14
        sub_w: int = self.size % 14
        new_h, new_w = self.size - sub_h, self.size - sub_w
        self.target_tr = v2.Compose(
            [
                v2.Resize(
                    (new_h, new_w), interpolation=v2.InterpolationMode.NEAREST_EXACT
                )
            ]
        )

    def get_paths(self):
        return (
            sorted(glob.glob(f"{self.base_path}/{self.mode}/image/*tif")),
            sorted(glob.glob(f"{self.base_path}/{self.mode}/label/*tif")),
        )

    def load_target(self, index):
        target = torch.tensor(
            np.array(Image.open(self.target_paths[index])) / 255
        ).long()
        target = self.target_tr(target.unsqueeze(0)).squeeze()

        return target

    def load_image(self, index):
        img = Image.open(self.img_paths[index])
        image = utils.convert_image(
            img, self.tr, to_gpu=False, to_half=False, device_str="cpu"
        )
        return image.squeeze()

    def __getitem__(self, index):
        image = self.load_image(index)
        target = self.load_target(index)

        return image.to(self.dtype), target

    def __len__(self):
        if len(self.img_paths) != len(self.target_paths):
            raise ValueError(
                f"Number of images and targets do not match. Images: {len(self.img_paths)}, Targets: {len(self.target_paths)}"
            )
        else:
            return len(self.img_paths)


if __name__ == "__main__":
    ds = VOC_Dataset(
        "/home/pawlo/Arbeit/positional_bias/dino-saw/Datasets/VOC",
        mode="train",
        set_255_0=True,
    )
