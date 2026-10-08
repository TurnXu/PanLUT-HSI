"""模拟与真实数据使用的融合质量指标。"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from q2n import calc_q2n


def psnr(prediction: Tensor, target: Tensor, maximum: float = 1.0) -> float:
    error = F.mse_loss(prediction, target).item()
    return float("inf") if error == 0 else 10.0 * math.log10(maximum**2 / error)


def sam(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> float:
    dot = (prediction * target).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(prediction, dim=1)
        * torch.linalg.vector_norm(target, dim=1)
    )
    angle = torch.acos((dot / (denominator + epsilon)).clamp(-1.0, 1.0))
    return angle.mean().item()


def ergas(prediction: Tensor, target: Tensor, scale_ratio: int = 8) -> float:
    spatial_dims = (-2, -1)
    band_rmse = torch.sqrt(((prediction - target) ** 2).mean(dim=spatial_dims))
    band_mean = target.mean(dim=spatial_dims).abs().clamp_min(1e-8)
    value = 100.0 / scale_ratio * torch.sqrt(((band_rmse / band_mean) ** 2).mean())
    return value.item()


def ssim(prediction: Tensor, target: Tensor, window_size: int = 11) -> float:
    """对每个光谱通道计算局部SSIM并取平均。"""
    channels = prediction.shape[1]
    padding = window_size // 2
    kernel = prediction.new_ones((channels, 1, window_size, window_size)) / window_size**2
    mean_prediction = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mean_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    prediction_variance = F.conv2d(prediction**2, kernel, padding=padding, groups=channels) - mean_prediction**2
    target_variance = F.conv2d(target**2, kernel, padding=padding, groups=channels) - mean_target**2
    covariance = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mean_prediction * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_prediction * mean_target + c1)
        * (2 * covariance + c2)
        / ((mean_prediction**2 + mean_target**2 + c1) * (prediction_variance + target_variance + c2))
    )
    return score.mean().item()


def evaluate_full_reference(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.clamp(0.0, 1.0)
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "sam": sam(prediction, target),
        "ergas": ergas(prediction, target),
    }


def _quality_index(image_a: Tensor, image_b: Tensor, patch_size: int = 16) -> Tensor:
    """按原工程实现计算分块通用图像质量指数。"""
    height = min(image_a.shape[-2], image_b.shape[-2])
    width = min(image_a.shape[-1], image_b.shape[-1])
    height = height // patch_size * patch_size
    width = width // patch_size * patch_size
    if height == 0 or width == 0:
        raise ValueError("图像尺寸小于QNR分块尺寸")
    image_a = image_a[..., :height, :width]
    image_b = image_b[..., :height, :width]
    batch, channels = image_a.shape[:2]
    blocks_a = image_a.reshape(
        batch, channels, height // patch_size, patch_size, width // patch_size, patch_size
    )
    blocks_b = image_b.reshape(
        batch, channels, height // patch_size, patch_size, width // patch_size, patch_size
    )
    mean_a = blocks_a.mean(dim=(3, 5))
    mean_b = blocks_b.mean(dim=(3, 5))
    variance_a = blocks_a.var(dim=(3, 5), unbiased=False)
    variance_b = blocks_b.var(dim=(3, 5), unbiased=False)
    covariance = (
        (blocks_a - mean_a[..., None, :, None])
        * (blocks_b - mean_b[..., None, :, None])
    ).mean(dim=(3, 5))
    score = 4 * covariance * mean_a * mean_b / (
        (variance_a + variance_b) * (mean_a.square() + mean_b.square()) + 1e-4
    )
    # 保留通道维，平均批次和所有空间块，与原工程average=False行为一致。
    return score.mean(dim=(0, 2, 3))


def qnr_mosaic(
    fused: Tensor,
    mosaic: Tensor,
    pan: Tensor,
    patch_size: int = 32,
    spatial_ratio: int = 2,
    spectral_power: float = 1.0,
    spatial_power: float = 0.5,
    alpha: float = 1.0,
    beta: float = 3.0,
) -> dict[str, float]:
    """复用原工程QNR定义评价无真值真实数据。"""
    if mosaic.shape[1] == 1:
        low_hsi = F.pixel_unshuffle(mosaic, 4)
    elif mosaic.shape[1] == fused.shape[1]:
        # generate.py按原流程保存解包后的16通道低分辨率观测。
        low_hsi = mosaic
    else:
        raise ValueError("mosaic必须是单通道原始排列或16通道解包数据")
    spectral_differences = []
    for band in range(fused.shape[1]):
        other = [index for index in range(fused.shape[1]) if index != band]
        fused_q = _quality_index(
            fused[:, band : band + 1].expand(-1, len(other), -1, -1), fused[:, other], patch_size
        )
        low_q = _quality_index(
            low_hsi[:, band : band + 1].expand(-1, len(other), -1, -1),
            low_hsi[:, other],
            patch_size,
        )
        spectral_differences.append((fused_q - low_q).abs().pow(spectral_power).mean())
    d_lambda = torch.stack(spectral_differences).mean().pow(1.0 / spectral_power)

    pan_full = pan.expand(-1, fused.shape[1], -1, -1)
    fused_down = F.interpolate(fused, scale_factor=1 / spatial_ratio, mode="bicubic", align_corners=False)
    pan_down = F.interpolate(pan, scale_factor=1 / spatial_ratio, mode="bicubic", align_corners=False)
    pan_down = pan_down.expand(-1, fused.shape[1], -1, -1)
    spatial_difference = (
        _quality_index(fused, pan_full, patch_size)
        - _quality_index(fused_down, pan_down, patch_size)
    ).abs()
    d_s = spatial_difference.pow(spatial_power).mean().pow(1.0 / spatial_power)
    qnr = (1.0 - d_lambda).clamp_min(0).pow(alpha) * (1.0 - d_s).clamp_min(0).pow(beta)
    return {"qnr": qnr.item(), "d_lambda": d_lambda.item(), "d_s": d_s.item()}


# 下列接口沿用模板命名，便于原有脚本直接调用。
def calc_psnr(hrms: Tensor, fused: Tensor) -> Tensor:
    return fused.new_tensor(psnr(fused, hrms))


def calc_ssim(hrms: Tensor, fused: Tensor) -> Tensor:
    return fused.new_tensor(ssim(fused, hrms))


def calc_sam(hrms: Tensor, fused: Tensor) -> Tensor:
    return fused.new_tensor(sam(fused, hrms) * 180.0 / math.pi)


def calc_ergas(hrms: Tensor, fused: Tensor, scale_factor: int = 8) -> Tensor:
    return fused.new_tensor(ergas(fused, hrms, scale_ratio=scale_factor))


def calc_qnr_mosaic(
    fused: Tensor,
    mosaic: Tensor,
    pan: Tensor,
    msfa_kernel: Tensor | None = None,
    patch_size: int = 32,
    scale_factor: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    del msfa_kernel
    values = qnr_mosaic(
        fused,
        mosaic,
        pan,
        patch_size=patch_size,
        spatial_ratio=scale_factor,
    )
    return (
        fused.new_tensor(values["qnr"]),
        fused.new_tensor(values["d_lambda"]),
        fused.new_tensor(values["d_s"]),
    )
