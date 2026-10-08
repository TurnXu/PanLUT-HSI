"""16波段高光谱全色融合的 Pan-LUT 迁移模型。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lut_layers import MultiLinearLUT, SpatialDetailLUT


class PanLUTHSI(nn.Module):
    """实现16→4→Pan-LUT→4→16的端到端基线。"""

    def __init__(
        self,
        spectral_bands: int = 16,
        latent_bands: int = 4,
        bins: int = 9,
        msfa_size: int = 4,
        chunk_pixels: int = 65536,
        backend: str = "torch",
        official_root: str = "../Pan-LUT-main",
        input_order: str = "spectral_pan",
        share_sdlut_directions: bool = True,
        regularization_reduction: str = "sum",
    ) -> None:
        super().__init__()
        if latent_bands != 4:
            raise ValueError("当前Pan-LUT接口固定为4个潜在光谱通道")
        if spectral_bands % latent_bands != 0:
            raise ValueError("spectral_bands必须能够被latent_bands整除")

        self.spectral_bands = spectral_bands
        self.latent_bands = latent_bands
        self.msfa_size = msfa_size
        self.backend = backend
        self.input_order = input_order
        self.regularization_reduction = regularization_reduction
        if input_order not in {"spectral_pan", "pan_spectral"}:
            raise ValueError("input_order必须为spectral_pan或pan_spectral")
        if regularization_reduction not in {"sum", "mean"}:
            raise ValueError("regularization_reduction必须为sum或mean")

        self.encoder = nn.Conv2d(spectral_bands, latent_bands, 1, bias=False)
        self.decoder = nn.Conv2d(latent_bands, spectral_bands, 1, bias=False)

        spectral_identity = (0, 1, 2, 3) if input_order == "spectral_pan" else (1, 2, 3, 4)
        if backend == "torch":
            self.pglut = MultiLinearLUT(
                input_dims=5,
                output_channels=5,
                bins=bins,
                identity_channels=(0, 1, 2, 3, 4),
                chunk_pixels=chunk_pixels,
            )
            self.sdlut = SpatialDetailLUT(
                bins=bins,
                chunk_pixels=chunk_pixels,
                share_directions=share_sdlut_directions,
            )
            self.aolut = MultiLinearLUT(
                input_dims=5,
                output_channels=4,
                bins=bins,
                identity_channels=spectral_identity,
                chunk_pixels=chunk_pixels,
            )
        elif backend == "official_cuda":
            from official_ops import (
                OfficialFiveDimensionalLUT,
                OfficialSpatialDetailLUT,
                load_official_transforms,
            )

            transforms = load_official_transforms(official_root)
            self.pglut = OfficialFiveDimensionalLUT(
                output_channels=5,
                bins=bins,
                identity_channels=(0, 1, 2, 3, 4),
                transform=transforms["pglut"],
            )
            self.sdlut = OfficialSpatialDetailLUT(
                channels=5,
                bins=bins,
                transforms=transforms,
                share_directions=share_sdlut_directions,
            )
            self.aolut = OfficialFiveDimensionalLUT(
                output_channels=4,
                bins=bins,
                identity_channels=spectral_identity,
                transform=transforms["pglut"],
            )
        else:
            raise ValueError("backend必须为torch或official_cuda")
        self._initialize_spectral_projection()

    def _initialize_spectral_projection(self) -> None:
        """按连续波段分组初始化，使模型初始状态接近分组平均与复制。"""
        group_size = self.spectral_bands // self.latent_bands
        with torch.no_grad():
            self.encoder.weight.zero_()
            self.decoder.weight.zero_()
            for latent_index in range(self.latent_bands):
                start = latent_index * group_size
                stop = start + group_size
                self.encoder.weight[latent_index, start:stop, 0, 0] = 1.0 / group_size
                self.decoder.weight[start:stop, latent_index, 0, 0] = 1.0

    def decode_mosaic(self, mosaic: Tensor) -> Tensor:
        """把单通道4×4 MSFA马赛克重排为16个低分辨率光谱通道。"""
        if mosaic.ndim != 4 or mosaic.shape[1] != 1:
            raise ValueError("mosaic必须是[B,1,H,W]张量")
        if mosaic.shape[-2] % self.msfa_size or mosaic.shape[-1] % self.msfa_size:
            raise ValueError("马赛克高宽必须能被msfa_size整除")
        decoded = F.pixel_unshuffle(mosaic, self.msfa_size)
        if decoded.shape[1] != self.spectral_bands:
            raise ValueError(
                f"马赛克解码得到{decoded.shape[1]}通道，但模型期望{self.spectral_bands}通道"
            )
        return decoded

    def forward_from_hsi(self, low_resolution_hsi: Tensor, pan: Tensor) -> Tensor:
        """从已经解码的16通道低分辨率高光谱图执行融合。"""
        if low_resolution_hsi.ndim != 4:
            raise ValueError("low_resolution_hsi必须是[B,C,H,W]张量")
        if low_resolution_hsi.shape[1] != self.spectral_bands:
            raise ValueError("low_resolution_hsi通道数与模型配置不一致")
        if pan.ndim != 4 or pan.shape[1] != 1:
            raise ValueError("pan必须是[B,1,H,W]张量")
        if low_resolution_hsi.shape[0] != pan.shape[0]:
            raise ValueError("高光谱与PAN批量大小不一致")

        latent = self.encoder(low_resolution_hsi)
        latent = F.interpolate(
            latent,
            size=pan.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        if self.input_order == "spectral_pan":
            pan_lut_input = torch.cat((latent, pan), dim=1)
        else:
            pan_lut_input = torch.cat((pan, latent), dim=1)
        pan_lut_input = pan_lut_input.clamp(0.0, 1.0)

        spectral_features = self.pglut(pan_lut_input).clamp(0.0, 1.0)
        spatial_features = self.sdlut(spectral_features).clamp(0.0, 1.0)
        latent_output = self.aolut(spatial_features).clamp(0.0, 1.0)
        return self.decoder(latent_output)

    def forward(self, mosaic: Tensor, pan: Tensor) -> Tensor:
        low_resolution_hsi = self.decode_mosaic(mosaic)
        return self.forward_from_hsi(low_resolution_hsi, pan)

    def lut_regularization(self) -> tuple[Tensor, Tensor]:
        """汇总三个LUT的平滑和单调正则项。"""
        if self.backend == "torch":
            smoothness = (
                self.pglut.smoothness_loss(self.regularization_reduction)
                + self.sdlut.smoothness_loss(self.regularization_reduction)
                + self.aolut.smoothness_loss(self.regularization_reduction)
            )
            monotonicity = (
                self.pglut.monotonicity_loss(self.regularization_reduction)
                + self.sdlut.monotonicity_loss(self.regularization_reduction)
                + self.aolut.monotonicity_loss(self.regularization_reduction)
            )
        else:
            smoothness = (
                self.pglut.smoothness_loss(self.regularization_reduction)
                + self.sdlut.smoothness_loss(self.regularization_reduction)
                + self.aolut.smoothness_loss(self.regularization_reduction)
            )
            monotonicity = (
                self.pglut.monotonicity_loss(self.regularization_reduction)
                + self.sdlut.monotonicity_loss(self.regularization_reduction)
                + self.aolut.monotonicity_loss(self.regularization_reduction)
            )
        return smoothness, monotonicity
