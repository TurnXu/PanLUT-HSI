"""PanLUT-HSI参数量与理论查表运算量统计。"""

from __future__ import annotations

from typing import Any

from model import PanLUTHSI


def count_parameters(model: PanLUTHSI) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def estimate_panlut_operations(
    model: PanLUTHSI,
    batch_size: int,
    pan_height: int,
    pan_width: int,
) -> dict[str, Any]:
    """估算乘加和LUT顶点访问次数，不执行大图前向。"""
    if pan_height % 8 or pan_width % 8:
        raise ValueError("PAN高宽必须是8的整数倍")
    high_pixels = batch_size * pan_height * pan_width
    low_pixels = batch_size * (pan_height // 8) * (pan_width // 8)
    encoder_macs = low_pixels * model.spectral_bands * model.latent_bands
    decoder_macs = high_pixels * model.latent_bands * model.spectral_bands
    pglut_vertices = high_pixels * (2**5)
    sdlut_vertices = high_pixels * (model.latent_bands + 1) * 4 * (2**4)
    aolut_vertices = high_pixels * (2**5)
    return {
        "encoder_macs": encoder_macs,
        "decoder_macs": decoder_macs,
        "convolution_macs": encoder_macs + decoder_macs,
        "pglut_vertex_reads": pglut_vertices,
        "sdlut_vertex_reads": sdlut_vertices,
        "aolut_vertex_reads": aolut_vertices,
        "lut_vertex_reads": pglut_vertices + sdlut_vertices + aolut_vertices,
    }


def format_count(value: int) -> str:
    for suffix, scale in (("T", 10**12), ("G", 10**9), ("M", 10**6), ("K", 10**3)):
        if value >= scale:
            return f"{value / scale:.3f}{suffix}"
    return str(value)
