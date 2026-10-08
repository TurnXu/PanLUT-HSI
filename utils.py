"""训练、生成、评价和可视化入口共享的工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """相对路径统一以工程根目录为基准，避免受启动目录影响。"""
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_json_config(project_root: Path, value: str | Path) -> dict[str, Any]:
    path = resolve_path(project_root, value)
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_model_config(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """把官方算子目录转换为稳定的绝对路径。"""
    prepared = dict(config)
    prepared["official_root"] = str(
        resolve_path(project_root, prepared.get("official_root", "../Pan-LUT-main"))
    )
    return prepared


def resolve_device(value: str | None, force_cpu: bool = False) -> torch.device:
    """兼容 README 中的数字 GPU 编号和 PyTorch 标准设备写法。"""
    if force_cpu or value == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if value not in (None, "cpu"):
            print("警告：CUDA不可用，已回退到CPU。")
        return torch.device("cpu")
    if value is None or value == "cuda":
        return torch.device("cuda:0")
    if value.isdigit():
        index = int(value)
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"请求cuda:{index}，但当前只有{torch.cuda.device_count()}张可用GPU"
            )
        return torch.device(f"cuda:{index}")
    return torch.device(value)


def dataset_root_from_readme(
    project_root: Path,
    data_path: str | None,
    dataset_name: str,
    configured_root: str,
) -> Path:
    """README 的 data_path 指向 Dataset 根目录，配置文件则直接指向具体数据集。"""
    if data_path:
        return resolve_path(project_root, data_path) / dataset_name
    return resolve_path(project_root, configured_root)


def training_output_dir(
    project_root: Path,
    index: int | None,
    configured_directory: str,
) -> Path:
    """传入 idx 时使用原工程的 <idx>/model 目录，否则沿用 JSON 配置。"""
    if index is not None:
        return project_root / str(index) / "model"
    return resolve_path(project_root, configured_directory)


def result_output_dir(
    project_root: Path,
    dataset_name: str,
    index: int,
    explicit_output: str | None,
) -> Path:
    if explicit_output:
        return resolve_path(project_root, explicit_output)
    return project_root / dataset_name / str(index) / "result"


def checkpoint_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """同时兼容本工程的新检查点和原流程的 net 字段。"""
    if "model" in checkpoint:
        return checkpoint["model"]
    if "net" in checkpoint:
        return checkpoint["net"]
    raise KeyError("检查点中既没有model字段，也没有net字段")


def pixel_shuffle(data: np.ndarray, upscale_factor: int) -> np.ndarray:
    """NumPy版像素重排，输入为[N,C,H,W]。"""
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    return F.pixel_shuffle(tensor, upscale_factor).numpy()


def pixel_shuffle_inv(data: np.ndarray, downscale_factor: int) -> np.ndarray:
    """NumPy版逆像素重排，输入为[N,C,H,W]。"""
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    return F.pixel_unshuffle(tensor, downscale_factor).numpy()


def MSFA_filter(data: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """按照MSFA排列把[H,W,C]光谱立方采样成单通道马赛克。"""
    height, width, bands = data.shape
    rows, columns = pattern.shape
    if bands != rows * columns or height % rows or width % columns:
        raise ValueError("数据尺寸或通道数与MSFA排列不匹配")
    mosaic = np.zeros((height, width, 1), dtype=data.dtype)
    for row in range(rows):
        for column in range(columns):
            mosaic[row::rows, column::columns, 0] = data[
                row::rows, column::columns, int(pattern[row, column])
            ]
    return mosaic


def MSFA_filter_inv(data: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """把单通道马赛克解包为低分辨率多通道光谱立方。"""
    if data.ndim == 2:
        data = data[:, :, None]
    rows, columns = pattern.shape
    height, width = data.shape[:2]
    if height % rows or width % columns:
        raise ValueError("马赛克尺寸与MSFA排列不匹配")
    unpacked = np.zeros((height // rows, width // columns, rows * columns), dtype=data.dtype)
    for row in range(rows):
        for column in range(columns):
            unpacked[:, :, int(pattern[row, column])] = data[row::rows, column::columns, 0]
    return unpacked


def region_detach(
    data: np.ndarray,
    size: list[int] | tuple[int, ...],
    coordinate: list[int] | tuple[int, ...],
    box_width: int = 5,
    rgb: str = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """截取两个局部区域，并在原图上绘制红蓝边框。"""
    if len(size) == 1:
        size = (size[0], size[0])
    if len(size) != 2:
        raise ValueError("size需要1或2个整数")
    if len(coordinate) == 2:
        coordinate = (coordinate[0], coordinate[1], coordinate[0], coordinate[1])
    if len(coordinate) != 4:
        raise ValueError("coordinate需要2或4个整数")
    if rgb not in {"r", "b"}:
        raise ValueError("rgb必须为r或b")
    height, width = int(size[0]), int(size[1])
    top1, left1, top2, left2 = map(int, coordinate)
    for top, left in ((top1, left1), (top2, left2)):
        if top < 0 or left < 0 or top + height > data.shape[0] or left + width > data.shape[1]:
            raise ValueError("局部区域超出图像边界")
    first = np.array([255, 0, 0] if rgb == "r" else [0, 0, 255], dtype=data.dtype)
    second = np.array([0, 0, 255] if rgb == "r" else [255, 0, 0], dtype=data.dtype)
    boxed = data.copy()
    for top, left, color in ((top1, left1, first), (top2, left2, second)):
        bottom, right = top + height, left + width
        boxed[top : top + box_width, left:right] = color
        boxed[bottom - box_width : bottom, left:right] = color
        boxed[top:bottom, left : left + box_width] = color
        boxed[top:bottom, right - box_width : right] = color
    return (
        data[top1 : top1 + height, left1 : left1 + width].copy(),
        data[top2 : top2 + height, left2 : left2 + width].copy(),
        boxed,
        (height, width),
        (top1, left1, top2, left2),
    )
