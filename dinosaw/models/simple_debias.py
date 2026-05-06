"""
Reimplementation of positional debiasing approach from 'INSID3: Training-Free In-Context Segmentation with DINOv3'

Extracts features of zero-tensor and projects other features to be orthogonal to top-k SVD components of zero-tensor features.

Source: 'https://github.com/visinf/INSID3/blob/main/models/insid3.py'
"""

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from dinosaw.models.vit_wrapper import PretrainedViTWrapper


class DebiasedViTWrapper(PretrainedViTWrapper):
    def forward_features(
        self, x: torch.Tensor, make_2D: bool = False, add_reg: bool = False, svd_components: int = 16
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        noise_img = normalize(
            torch.zeros(1, 3, h, w),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).to(x.device)
        noise_fmap = super().forward_features(noise_img, False, False)
        N = F.normalize(noise_fmap, p=2, dim=1)

        # Center across spatial locations (not batch)
        E = N - N.mean(dim=(2,), keepdim=True)
        # SVD: E is [1, C, HW], squeeze batch
        E = E.squeeze(0)
        # E: [C, HW]
        basis, _, _ = torch.linalg.svd(E, full_matrices=False)
        # Limit to top-k components
        basis = basis[:, :svd_components]

        base_feats = super().forward_features(x, True, add_reg)
        B, C, H, W = base_feats.shape
        X = base_feats.reshape(B, C, H * W)
        X = F.normalize(X, p=2, dim=1)

        P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        X_deb = torch.matmul(P_perp, X)
        X_deb = X_deb.reshape(B, C, H, W)

        X_deb = F.normalize(X_deb, p=2, dim=1)

        if make_2D:
            X_deb = X_deb.reshape(B, C, H, W)

        return X_deb


if __name__ == "__main__":
    d = DebiasedViTWrapper(
        model_identifier="vit_small_patch14_reg4_dinov2.lvd142m", device="cuda:0", add_flash_attn=False
    )
    x = torch.ones((1, 3, 518, 518), device="cuda:0")
    d.forward_features(x, True)
    print(x.shape)
