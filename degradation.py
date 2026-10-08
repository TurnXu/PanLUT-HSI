"""与模板工程一致的可微MSFA和PAN退化模型。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def build_msfa_kernel(msfa_size: int = 4, spatial_ratio: int = 2) -> Tensor:
    """构造原工程使用的16波段MSFA平均采样核。"""
    bands = msfa_size**2
    kernel_size = msfa_size * spatial_ratio
    kernel = torch.zeros(bands, 1, kernel_size, kernel_size)
    weight = 1.0 / spatial_ratio**2
    for row in range(msfa_size):
        for column in range(msfa_size):
            band = row * msfa_size + column
            y = row * spatial_ratio
            x = column * spatial_ratio
            kernel[band, 0, y : y + spatial_ratio, x : x + spatial_ratio] = weight
    return kernel


def degrade_to_mosaic(
    high_resolution_hsi: Tensor,
    msfa_kernel: Tensor,
    msfa_size: int = 4,
) -> Tensor:
    """把16波段高分辨率结果退化为单通道MSFA马赛克。"""
    bands = high_resolution_hsi.shape[1]
    if bands != msfa_size**2:
        raise ValueError("高光谱通道数必须等于msfa_size平方")
    kernel = msfa_kernel.to(
        device=high_resolution_hsi.device,
        dtype=high_resolution_hsi.dtype,
    )
    sampled = F.conv2d(
        high_resolution_hsi,
        kernel,
        stride=kernel.shape[-1],
        groups=bands,
    )
    return F.pixel_shuffle(sampled, upscale_factor=msfa_size)


class LearnableSpectralResponse(nn.Module):
    """原真实数据训练中的可学习16波段到PAN响应。"""

    def __init__(self, spectral_bands: int = 16) -> None:
        super().__init__()
        self.projection = nn.Conv2d(spectral_bands, 1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1.0 / spectral_bands)

    def forward(self, high_resolution_hsi: Tensor) -> Tensor:
        return self.projection(high_resolution_hsi)
