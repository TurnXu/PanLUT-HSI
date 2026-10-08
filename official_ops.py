"""对Pan-LUT-main官方CUDA查表算子的薄封装。

该模块不负责编译扩展。只有在原目录中的PGLUT、SDLUT及三个旋转SDLUT
已经完成编译安装后，才可选择 ``backend=official_cuda``。
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _official_search_paths(official_root: Path) -> list[Path]:
    return [
        official_root / "PGLUT" / "PGLUT_transform",
        official_root / "SDLUT" / "SDLUT_transform",
        official_root / "SDLUT" / "SDLUT90_transform",
        official_root / "SDLUT" / "SDLUT180_transform",
        official_root / "SDLUT" / "SDLUT270_transform",
    ]


def load_official_transforms(official_root: str | Path) -> dict[str, Callable]:
    """加载官方扩展；缺失时给出可定位的错误信息。"""
    root = Path(official_root).resolve()
    for path in _official_search_paths(root):
        if not path.exists():
            raise FileNotFoundError(f"官方算子目录不存在：{path}")
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)

    module_names = {
        "pglut": ("PGLUT", "PGLUT_transform"),
        "sd0": ("SDLUT", "SDLUT_transform"),
        "sd90": ("SD90LUT", "SD90LUT_transform"),
        "sd180": ("SD180LUT", "SD180LUT_transform"),
        "sd270": ("SD270LUT", "SD270LUT_transform"),
    }
    transforms: dict[str, Callable] = {}
    for key, (module_name, attribute_name) in module_names.items():
        try:
            module = importlib.import_module(module_name)
            transforms[key] = getattr(module, attribute_name)
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"无法加载官方CUDA算子{module_name}.{attribute_name}。"
                "请先在对应目录编译扩展，并确保PyTorch与CUDA工具链匹配。"
            ) from error
    return transforms


def _identity_table(
    input_dims: int,
    output_channels: int,
    bins: int,
    identity_channels: Sequence[int],
) -> Tensor:
    """按官方CUDA的第一输入维最快布局构造恒等初始化表。"""
    total_nodes = bins**input_dims
    linear = torch.arange(total_nodes, dtype=torch.long)
    coordinates = []
    for axis in range(input_dims):
        coordinates.append(((linear // (bins**axis)) % bins).float() / (bins - 1))
    flat = torch.stack([coordinates[index] for index in identity_channels], dim=0)
    return flat.reshape((1, output_channels) + (bins,) * input_dims)


class OfficialFiveDimensionalLUT(nn.Module):
    def __init__(
        self,
        output_channels: int,
        bins: int,
        identity_channels: Sequence[int],
        transform: Callable,
    ) -> None:
        super().__init__()
        self.table = nn.Parameter(
            _identity_table(5, output_channels, bins, identity_channels)
        )
        self.transform = transform

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[0] != 1:
            raise ValueError("官方PGLUT CUDA内核仅安全支持batch_size=1")
        return self.transform(inputs.contiguous(), self.table.contiguous())

    def smoothness_loss(self, reduction: str = "sum") -> Tensor:
        return _table_regularization(self.table, 5, "smoothness", reduction)

    def monotonicity_loss(self, reduction: str = "sum") -> Tensor:
        return _table_regularization(self.table, 5, "monotonicity", reduction)


class OfficialSpatialDetailLUT(nn.Module):
    """依次调用官方0°、90°、180°、270°空间查表算子。"""

    def __init__(
        self,
        channels: int,
        bins: int,
        transforms: dict[str, Callable],
        share_directions: bool,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.share_directions = share_directions
        self.transforms = [
            transforms["sd0"],
            transforms["sd90"],
            transforms["sd180"],
            transforms["sd270"],
        ]
        table_count = 1 if share_directions else 4
        base = _identity_table(4, 1, bins, (0,)).squeeze(0)
        self.tables = nn.ParameterList(
            [nn.Parameter(base.clone()) for _ in range(table_count)]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[0] != 1:
            raise ValueError("官方SDLUT CUDA内核固定读取batch_size=1")
        output = inputs
        for index, transform in enumerate(self.transforms):
            table = self.tables[0] if self.share_directions else self.tables[index]
            # 官方算子以LUT第一维表示被独立处理的特征通道。
            channel_tables = table.unsqueeze(0).expand(self.channels, -1, -1, -1, -1, -1)
            output = transform(output.contiguous(), channel_tables.contiguous())
        return output

    def smoothness_loss(self, reduction: str = "sum") -> Tensor:
        return torch.stack(
            [_table_regularization(table, 4, "smoothness", reduction) for table in self.tables]
        ).sum()

    def monotonicity_loss(self, reduction: str = "sum") -> Tensor:
        return torch.stack(
            [_table_regularization(table, 4, "monotonicity", reduction) for table in self.tables]
        ).sum()


def _table_regularization(
    table: Tensor,
    input_dims: int,
    kind: str,
    reduction: str,
) -> Tensor:
    if reduction not in {"sum", "mean"}:
        raise ValueError("LUT正则reduction必须为sum或mean")
    values = []
    for axis in range(input_dims):
        difference = torch.diff(table, dim=table.ndim - input_dims + axis)
        penalty = difference.square() if kind == "smoothness" else F.relu(-difference)
        values.append(penalty.sum() if reduction == "sum" else penalty.mean())
    return torch.stack(values).sum()
