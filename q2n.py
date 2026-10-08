"""多光谱融合的Q2n质量指标。"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def calc_q2n(
    ground_truth: Tensor,
    fused: Tensor,
    Q_blocks_size: int,
    Q_shift: int,
) -> tuple[Tensor, Tensor]:
    """计算Q2n及其分块质量图，输入形状均为[C,H,W]。"""
    if ground_truth.shape != fused.shape or ground_truth.ndim != 3:
        raise ValueError("Q2n输入必须是形状一致的[C,H,W]张量")
    if Q_blocks_size < 2 or Q_shift < 1:
        raise ValueError("Q_blocks_size至少为2，Q_shift至少为1")

    reference = ground_truth.float()
    estimate = fused.float()
    channels, height, width = reference.shape
    steps_y = max(1, math.ceil(height / Q_shift))
    steps_x = max(1, math.ceil(width / Q_shift))
    padded_height = (steps_y - 1) * Q_shift + Q_blocks_size
    padded_width = (steps_x - 1) * Q_shift + Q_blocks_size
    pad_y = max(0, padded_height - height)
    pad_x = max(0, padded_width - width)
    if pad_y or pad_x:
        padding = (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
        reference = F.pad(reference, padding, mode="reflect")
        estimate = F.pad(estimate, padding, mode="reflect")

    padded_channels = 1 << (channels - 1).bit_length()
    if padded_channels != channels:
        zeros = reference.new_zeros((padded_channels - channels, *reference.shape[1:]))
        reference = torch.cat((reference, zeros), dim=0)
        estimate = torch.cat((estimate, zeros.clone()), dim=0)
        channels = padded_channels

    values = reference.new_zeros((steps_y, steps_x, channels))
    for row in range(steps_y):
        for column in range(steps_x):
            top = row * Q_shift
            left = column * Q_shift
            reference_block = reference[
                :, top : top + Q_blocks_size, left : left + Q_blocks_size
            ]
            estimate_block = estimate[
                :, top : top + Q_blocks_size, left : left + Q_blocks_size
            ]
            values[row, column] = _onions_quality(reference_block, estimate_block)

    quality_map = torch.sqrt(torch.sum(values.square(), dim=2))
    return quality_map.mean(), quality_map


def _normalize_block(block: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean = block.mean()
    standard_deviation = block.std(unbiased=False)
    if standard_deviation.abs() < 1e-12:
        return block - mean + 1.0, mean, standard_deviation
    return (block - mean) / standard_deviation + 1.0, mean, standard_deviation


def _onions_quality(reference: Tensor, estimate: Tensor) -> Tensor:
    estimate = estimate.clone()
    if estimate.shape[0] > 1:
        estimate[1:] = -estimate[1:]

    normalized_reference = []
    normalized_estimate = []
    for channel in range(reference.shape[0]):
        reference_channel, mean, deviation = _normalize_block(reference[channel : channel + 1])
        estimate_channel = estimate[channel : channel + 1]
        sign = 1.0 if channel == 0 else -1.0
        signed_estimate = estimate_channel if channel == 0 else -estimate_channel
        if deviation.abs() < 1e-12:
            estimate_channel = sign * (signed_estimate - mean + 1.0)
        else:
            estimate_channel = sign * ((signed_estimate - mean) / deviation + 1.0)
        normalized_reference.append(reference_channel)
        normalized_estimate.append(estimate_channel)

    reference = torch.cat(normalized_reference, dim=0)
    estimate = torch.cat(normalized_estimate, dim=0)
    mean_reference = reference.mean(dim=(1, 2))
    mean_estimate = estimate.mean(dim=(1, 2))
    modulus_reference_mean = torch.linalg.vector_norm(mean_reference)
    modulus_estimate_mean = torch.linalg.vector_norm(mean_estimate)
    modulus_reference = torch.linalg.vector_norm(reference, dim=0)
    modulus_estimate = torch.linalg.vector_norm(estimate, dim=0)
    pixels = reference.shape[1] * reference.shape[2]
    correction = pixels / (pixels - 1)

    mean_product = modulus_reference_mean * modulus_estimate_mean
    mean_energy = modulus_reference_mean.square() + modulus_estimate_mean.square()
    covariance_energy = correction * (
        modulus_reference.square().mean()
        + modulus_estimate.square().mean()
        - mean_energy
    )
    mean_bias = 2.0 * mean_product / mean_energy.clamp_min(1e-12)
    if covariance_energy.abs() < 1e-12:
        quality = reference.new_zeros(reference.shape[0])
        quality[-1] = mean_bias
        return quality

    pixel_product = _onion_multiply_2d(reference, estimate)
    mean_onion_product = _onion_multiply(mean_reference, mean_estimate)
    covariance = correction * pixel_product.mean(dim=(1, 2)) - correction * mean_onion_product
    return covariance * mean_bias * (2.0 / covariance_energy)


def _conjugate(value: Tensor) -> Tensor:
    if value.shape[0] <= 1:
        return value
    return torch.cat((value[:1], -value[1:]), dim=0)


def _onion_multiply_2d(first: Tensor, second: Tensor) -> Tensor:
    channels = first.shape[0]
    if channels == 1:
        return first * second
    half = channels // 2
    a, b = first[:half], first[half:]
    c, d = second[:half], second[half:]
    if channels == 2:
        return torch.cat((a * c - d * b, a * d + c * b), dim=0)
    real = _onion_multiply_2d(a, c) - _onion_multiply_2d(d, _conjugate(b))
    imaginary = _onion_multiply_2d(_conjugate(a), d) + _onion_multiply_2d(c, b)
    return torch.cat((real, imaginary), dim=0)


def _onion_multiply(first: Tensor, second: Tensor) -> Tensor:
    channels = first.shape[0]
    if channels == 1:
        return first * second
    half = channels // 2
    a, b = first[:half], first[half:]
    c, d = second[:half], second[half:]
    if channels == 2:
        return torch.cat((a * c - d * b, a * d + c * b))
    real = _onion_multiply(a, c) - _onion_multiply(d, _conjugate(b))
    imaginary = _onion_multiply(_conjugate(a), d) + _onion_multiply(c, b)
    return torch.cat((real, imaginary))
