"""Frozen timm DINOv3-S visual frontend and a time-invariant target space."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FrozenDINOv3(nn.Module):
    """Patch-token interface for the exact timm model requested by the user.

    Expected reference shape on 256x256: [B,261,384] = 5 prefix + 256 patches.
    The *last* H/16*W/16 tokens are spatial patches for this timm ViT.
    """

    def __init__(self, model_name="vit_small_patch16_dinov3.lvd1689m", *,
                 pretrained=True, model_factory=None) -> None:
        super().__init__()
        if model_factory is None:
            try:
                import timm
            except ImportError as exc:
                raise RuntimeError("请先运行 `uv sync` 安装 timm") from exc
            model_factory = timm.create_model
        self.model_name = model_name
        self.backbone = model_factory(model_name, pretrained=pretrained, num_classes=0)
        self.backbone.requires_grad_(False).eval()
        self.feature_dim = int(getattr(self.backbone, "num_features", 384))
        patch = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", (16, 16))
        self.patch_size = (int(patch), int(patch)) if isinstance(patch, int) else tuple(patch)
        self.preprocessing = None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # 永远冻结 backbone / 关闭其 dropout
        return self

    def make_transform(self):
        import timm
        cfg = timm.data.resolve_model_data_config(self.backbone)
        return timm.data.create_transform(**cfg, is_training=False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images 应为 [B,3,H,W]，实际 {tuple(images.shape)}")
        ph, pw = self.patch_size
        h, w = images.shape[-2:]
        if h % ph or w % pw:
            raise ValueError(f"图像尺寸 {(h,w)} 必须能被 patch_size {(ph,pw)} 整除")
        grid = (h // ph, w // pw)
        features = self.backbone.forward_features(images)
        if features.ndim != 3:
            raise RuntimeError(f"DINOv3 forward_features 应返回 [B,N,C]，实际 {features.shape}")
        n_patches = grid[0] * grid[1]
        if features.shape[1] < n_patches:
            raise RuntimeError("DINOv3 token 长度不足以提取全部空间 patches")
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError("DINOv3 特征维度与 num_features 不一致")
        # 对此 timm DINOv3，前面的 CLS / Register 是特殊 token，不能当成空间格点。
        return features[:, -n_patches:, :].detach(), grid


class FixedSpatialTarget(nn.Module):
    """Stable spatial coordinates for short/goal latent difference targets.

    Only deterministic adaptive average pooling; never condition target coordinates
    on current observation, task, time or memory. This does NOT solve optical
    correspondence across large motion; it only ensures a fixed token grid.
    """

    def __init__(self, target_hw: int = 4):
        super().__init__()
        self.target_hw = target_hw

    def forward(self, patches: Tensor, grid: tuple[int, int] | None = None) -> Tensor:
        if patches.ndim != 3:
            raise ValueError("patches 必须为 [B,P,C]")
        b, n, c = patches.shape
        if grid is None:
            side = math.isqrt(n)
            if side * side != n:
                raise ValueError("非正方 patch grid 请显式提供 (H,W)")
            grid = (side, side)
        if grid[0] * grid[1] != n:
            raise ValueError(f"grid {grid} 与 patches 数量 {n} 不匹配")
        feature = patches.transpose(1, 2).reshape(b, c, *grid)
        pooled = F.adaptive_avg_pool2d(feature, (self.target_hw, self.target_hw))
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def residual_targets(self, current: Tensor, short: Tensor, goal: Tensor,
                         grid: tuple[int, int]):
        z0 = self(current, grid)
        return self(short, grid)-z0, self(goal, grid)-z0
