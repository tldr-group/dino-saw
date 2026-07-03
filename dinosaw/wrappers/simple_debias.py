import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PVW import PretrainedViTWrapper, WrapperRegistry, WrapperConfig, BackboneConfig


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


# Register Debiased ViT
WrapperRegistry.register(
    "dv2_db",
    WrapperConfig(
        backbone_cfg=BackboneConfig(backbone_type="timm", model_arch="dinov2_s"),
        wrapper_class=DebiasedViTWrapper,
    ),
)
