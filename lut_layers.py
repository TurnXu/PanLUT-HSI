"""可学习多维查找表及其纯PyTorch插值实现。"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiLinearLUT(nn.Module):
    """使用多线性插值的可学习N维查找表。

    输入和查找表索引均约定在 ``[0, 1]`` 范围内。前向过程按像素分块，
    防止五维插值在大图上同时展开32个顶点而造成显存峰值过高。
    """

    def __init__(
        self,
        input_dims: int,
        output_channels: int,
        bins: int = 9,
        identity_channels: Sequence[int] | None = None,
        chunk_pixels: int = 65536,
    ) -> None:
        super().__init__()
        if input_dims < 1:
            raise ValueError("input_dims必须大于0")
        if output_channels < 1:
            raise ValueError("output_channels必须大于0")
        if bins < 2:
            raise ValueError("bins必须至少为2")
        if chunk_pixels < 1:
            raise ValueError("chunk_pixels必须大于0")

        self.input_dims = input_dims
        self.output_channels = output_channels
        self.bins = bins
        self.chunk_pixels = chunk_pixels

        table_shape = (output_channels,) + (bins,) * input_dims
        self.table = nn.Parameter(torch.empty(table_shape))
        self.reset_parameters(identity_channels)

        strides = [bins ** power for power in reversed(range(input_dims))]
        self.register_buffer(
            "linear_strides",
            torch.tensor(strides, dtype=torch.long),
            persistent=False,
        )

    def reset_parameters(self, identity_channels: Sequence[int] | None) -> None:
        """用指定输入维度初始化每个输出通道，形成稳定的近似恒等映射。"""
        if identity_channels is None:
            nn.init.uniform_(self.table, 0.0, 1.0)
            return
        if len(identity_channels) != self.output_channels:
            raise ValueError("identity_channels长度必须等于输出通道数")

        axes = torch.meshgrid(
            *[torch.linspace(0.0, 1.0, self.bins) for _ in range(self.input_dims)],
            indexing="ij",
        )
        initialized = torch.stack([axes[index] for index in identity_channels], dim=0)
        with torch.no_grad():
            self.table.copy_(initialized)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("LUT输入必须是[B,C,H,W]四维张量")
        if inputs.shape[1] != self.input_dims:
            raise ValueError(
                f"LUT期望{self.input_dims}个输入通道，实际得到{inputs.shape[1]}个"
            )

        batch, _, height, width = inputs.shape
        flattened = inputs.clamp(0.0, 1.0).flatten(2)
        outputs = []
        for start in range(0, height * width, self.chunk_pixels):
            stop = min(start + self.chunk_pixels, height * width)
            outputs.append(self._interpolate_chunk(flattened[:, :, start:stop]))
        output = torch.cat(outputs, dim=-1)
        return output.reshape(batch, self.output_channels, height, width)

    def _interpolate_chunk(self, values: Tensor) -> Tensor:
        scaled = values * (self.bins - 1)
        lower = torch.floor(scaled).long().clamp(0, self.bins - 2)
        fraction = scaled - lower.to(scaled.dtype)

        batch, _, pixels = values.shape
        flat_table = self.table.reshape(self.output_channels, -1)
        result = values.new_zeros((batch, self.output_channels, pixels))

        for corner in itertools.product((0, 1), repeat=self.input_dims):
            corner_tensor = lower.new_tensor(corner).view(1, self.input_dims, 1)
            indices = lower + corner_tensor
            linear_indices = (indices * self.linear_strides.view(1, -1, 1)).sum(dim=1)

            weights = values.new_ones((batch, pixels))
            for axis, is_upper in enumerate(corner):
                axis_fraction = fraction[:, axis]
                weights = weights * (axis_fraction if is_upper else 1.0 - axis_fraction)

            selected = torch.gather(
                flat_table.unsqueeze(0).expand(batch, -1, -1),
                dim=2,
                index=linear_indices.unsqueeze(1).expand(-1, self.output_channels, -1),
            )
            result = result + selected * weights.unsqueeze(1)
        return result

    def smoothness_loss(self, reduction: str = "mean") -> Tensor:
        """惩罚相邻LUT节点的剧烈变化。"""
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction必须为sum或mean")
        losses = []
        for axis in range(self.input_dims):
            difference = torch.diff(self.table, dim=axis + 1)
            penalty = difference.square()
            losses.append(penalty.sum() if reduction == "sum" else penalty.mean())
        return torch.stack(losses).sum()

    def monotonicity_loss(self, reduction: str = "mean") -> Tensor:
        """惩罚输入增大而输出减小的LUT区间。"""
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction必须为sum或mean")
        losses = []
        for axis in range(self.input_dims):
            difference = torch.diff(self.table, dim=axis + 1)
            penalty = F.relu(-difference)
            losses.append(penalty.sum() if reduction == "sum" else penalty.mean())
        return torch.stack(losses).sum()


class SpatialDetailLUT(nn.Module):
    """共享四维LUT，通过四个方向的邻域连续注入空间细节。"""

    DIRECTIONS: tuple[tuple[tuple[int, int], ...], ...] = (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (0, -1), (1, -1)),
        ((0, 0), (0, -1), (-1, 0), (-1, -1)),
        ((0, 0), (-1, 0), (0, 1), (-1, 1)),
    )

    def __init__(
        self,
        bins: int = 9,
        chunk_pixels: int = 65536,
        share_directions: bool = True,
    ) -> None:
        super().__init__()
        self.share_directions = share_directions

        def make_lut() -> MultiLinearLUT:
            return MultiLinearLUT(
                input_dims=4,
                output_channels=1,
                bins=bins,
                identity_channels=(0,),
                chunk_pixels=chunk_pixels,
            )

        if share_directions:
            self.luts = nn.ModuleList([make_lut()])
        else:
            self.luts = nn.ModuleList([make_lut() for _ in self.DIRECTIONS])

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for direction_index, offsets in enumerate(self.DIRECTIONS):
            lut = self.luts[0] if self.share_directions else self.luts[direction_index]
            output = self._directional_lookup(output, offsets, lut)
        return output

    def _directional_lookup(
        self,
        inputs: Tensor,
        offsets: Iterable[tuple[int, int]],
        lut: MultiLinearLUT,
    ) -> Tensor:
        batch, channels, height, width = inputs.shape
        padded = F.pad(inputs, (1, 1, 1, 1), mode="replicate")
        neighbors = []
        for delta_y, delta_x in offsets:
            y_start = 1 + delta_y
            x_start = 1 + delta_x
            neighbors.append(
                padded[:, :, y_start : y_start + height, x_start : x_start + width]
            )
        neighborhoods = torch.stack(neighbors, dim=2)
        neighborhoods = neighborhoods.reshape(batch * channels, 4, height, width)
        transformed = lut(neighborhoods)
        return transformed.reshape(batch, channels, height, width)

    def smoothness_loss(self, reduction: str = "mean") -> Tensor:
        return torch.stack([lut.smoothness_loss(reduction) for lut in self.luts]).sum()

    def monotonicity_loss(self, reduction: str = "mean") -> Tensor:
        return torch.stack([lut.monotonicity_loss(reduction) for lut in self.luts]).sum()
