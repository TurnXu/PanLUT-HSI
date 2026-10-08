"""Pan-LUT高光谱基线使用的训练损失。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import PanLUTHSI


def spectral_angle_loss(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    """计算逐像素光谱角损失，降低不同波段比例关系的失真。"""
    dot_product = (prediction * target).sum(dim=1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    cosine = dot_product / (prediction_norm * target_norm + epsilon)
    return torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)).mean()


class PanLUTLoss(nn.Module):
    def __init__(
        self,
        reconstruction: str = "mse",
        smoothness_weight: float = 1e-4,
        monotonicity_weight: float = 10.0,
        spectral_angle_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if reconstruction not in {"mse", "l1"}:
            raise ValueError("reconstruction只支持mse或l1")
        self.reconstruction = reconstruction
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight
        self.spectral_angle_weight = spectral_angle_weight

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        model: PanLUTHSI,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.reconstruction == "mse":
            reconstruction_loss = F.mse_loss(prediction, target)
        else:
            reconstruction_loss = F.l1_loss(prediction, target)

        smoothness_loss, monotonicity_loss = model.lut_regularization()
        angle_loss = prediction.new_zeros(())
        if self.spectral_angle_weight > 0:
            angle_loss = spectral_angle_loss(prediction, target)

        total = (
            reconstruction_loss
            + self.smoothness_weight * smoothness_loss
            + self.monotonicity_weight * monotonicity_loss
            + self.spectral_angle_weight * angle_loss
        )
        components = {
            "total": total.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "smoothness": smoothness_loss.detach(),
            "monotonicity": monotonicity_loss.detach(),
            "spectral_angle": angle_loss.detach(),
        }
        return total, components
