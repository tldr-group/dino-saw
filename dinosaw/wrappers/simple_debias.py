import torch
import torch.nn.functional as F
from PIL import Image
from PVW import PretrainedViTWrapper
from PVW.types import Architectures, ImageTransform, ViTBackbone
from PVW.wrapper import closest_resize
from torch import Tensor
from torchvision.transforms.functional import normalize


class ChannelBlankedWrapper(PretrainedViTWrapper):
    def __init__(
        self,
        vit: ViTBackbone,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        channels_to_blank: list[int] | None = None,
        transform: ImageTransform = closest_resize,
        name: str = "",
        arch_name: Architectures | None = None,
    ):
        super().__init__(
            vit=vit,
            device=device,
            dtype=dtype,
            transform=transform,
            name=name,
            arch_name=arch_name,
        )
        self.channels_to_blank = channels_to_blank

    def forward_features(
        self,
        x: Tensor | Image.Image | list[Image.Image],
        make_2D: bool = True,
    ) -> Tensor:
        f = super().forward_features(x, make_2D=make_2D)

        if self.channels_to_blank is not None:
            f[:, self.channels_to_blank] = 0.0
        return f


class TransformAverageWrapper(PretrainedViTWrapper):
    def forward_features(self, x: Tensor | Image.Image | list[Image.Image], make_2D: bool = True) -> Tensor:
        x_t: Tensor = self.preprocess_input(x)
        x_flip_lr = torch.flip(x_t, dims=[3])
        x_flip_ud = torch.flip(x_t, dims=[2])
        x_rot_90 = torch.rot90(x_t, k=1, dims=[2, 3])

        f = super().forward_features(x_t, make_2D=make_2D)
        for tr_input in [x_flip_lr, x_flip_ud, x_rot_90]:
            f += super().forward_features(tr_input, make_2D=make_2D)
        f /= 4.0

        return f


class DebiasedViTWrapper(PretrainedViTWrapper):
    def forward_features(
        self, x: torch.Tensor, make_2D: bool = False, svd_components: int = 16, **kwargs
    ) -> torch.Tensor:
        x_tensor = self.preprocess_input(x)
        _, _, h, w = x_tensor.shape
        noise_img = normalize(
            torch.zeros(1, 3, h, w),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).to(x_tensor.device)

        noise_fmap = super().forward_features(noise_img, make_2D=False)
        N = F.normalize(noise_fmap, p=2, dim=1)

        # Center across spatial locations (not batch)
        E = N - N.mean(dim=(2,), keepdim=True)
        # SVD: E is [1, C, HW], squeeze batch
        E = E.squeeze(0)
        # E: [C, HW]
        basis, _, _ = torch.linalg.svd(E, full_matrices=False)
        # Limit to top-k components
        basis = basis[:, :svd_components]

        base_feats = super().forward_features(x_tensor, make_2D=True)
        B, C, H, W = base_feats.shape
        X = base_feats.reshape(B, C, H * W)
        X = F.normalize(X, p=2, dim=1)

        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp, X)

        if make_2D:
            X_deb = X_deb.reshape(B, C, H, W)
            X_deb = F.normalize(X_deb, p=2, dim=1)
        else:
            X_deb = F.normalize(X_deb, p=2, dim=1)

        return X_deb
