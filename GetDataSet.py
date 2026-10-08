"""CAVE、Chikusei及真实RAW数据的统一读取与在线模拟。"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import numpy as np
import scipy.io
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


MSFA_PATTERN = np.arange(16, dtype=np.int64).reshape(4, 4)
PAN_SPECTRAL_RESPONSE = np.asarray(
    [1, 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1],
    dtype=np.float32,
)
PAN_SPECTRAL_RESPONSE /= PAN_SPECTRAL_RESPONSE.sum()


def _natural_index(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def _prepare_hsi(image: np.ndarray) -> np.ndarray:
    """保持原数据代码的数值尺度，只处理NaN和数据类型。"""
    image = np.nan_to_num(image.astype(np.float32), copy=False)
    minimum = float(image.min())
    if minimum < 0:
        raise ValueError("高光谱数据包含负值，需先明确辐射定标方式")
    return image


def load_hsi_mat(path: Path, dataset_name: str) -> np.ndarray:
    """读取MAT并整理为[H,W,16]。"""
    content = scipy.io.loadmat(path)
    dataset_name = dataset_name.lower()
    if dataset_name == "cave":
        if "b" not in content:
            raise KeyError(f"{path}中没有变量b")
        image = content["b"][:, :, 12:28]
    elif dataset_name == "chikusei":
        if "chikusei" not in content:
            raise KeyError(f"{path}中没有变量chikusei")
        image = content["chikusei"]
    else:
        raise ValueError(f"不支持的模拟数据集：{dataset_name}")

    if image.ndim != 3 or image.shape[2] != 16:
        raise ValueError(f"{path}应为16波段数据，实际形状为{image.shape}")
    return _prepare_hsi(image)


def _aligned_crop_origin(
    height: int,
    width: int,
    patch_size: int,
    alignment: int,
    random_crop: bool,
) -> tuple[int, int]:
    if patch_size % alignment:
        raise ValueError(f"patch_size必须是{alignment}的整数倍")
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size超过原始图像尺寸")
    maximum_top = (height - patch_size) // alignment
    maximum_left = (width - patch_size) // alignment
    if random_crop:
        top = random.randint(0, maximum_top) * alignment
        left = random.randint(0, maximum_left) * alignment
    else:
        top = maximum_top // 2 * alignment
        left = maximum_left // 2 * alignment
    return top, left


def _grid_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """生成完整落在图像内的滑窗起点，与原数据流程保持一致。"""
    if patch_size > length:
        raise ValueError("patch_size超过原始图像尺寸")
    return list(range(0, length - patch_size + 1, stride))


def _matches_data_id(path: Path, data_ids: set[str]) -> bool:
    return not data_ids or path.name in data_ids or path.stem in data_ids


def simulate_observations(hr_hsi: Tensor) -> tuple[Tensor, Tensor]:
    """由高分辨率16波段真值生成单通道MSFA马赛克和PAN。"""
    if hr_hsi.ndim != 3 or hr_hsi.shape[0] != 16:
        raise ValueError("hr_hsi必须为[16,H,W]")

    low_hsi = F.avg_pool2d(hr_hsi.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
    _, height, width = low_hsi.shape
    mosaic = low_hsi.new_zeros((1, height, width))
    for row in range(4):
        for column in range(4):
            band = int(MSFA_PATTERN[row, column])
            mosaic[0, row::4, column::4] = low_hsi[band, row::4, column::4]

    response = hr_hsi.new_tensor(PAN_SPECTRAL_RESPONSE).view(16, 1, 1)
    pan = (hr_hsi * response).sum(dim=0, keepdim=True)
    return mosaic, pan


class SimulatedMosaicDataset(Dataset):
    """从CAVE或Chikusei高光谱真值在线生成马赛克与PAN。"""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        patch_size: int | None = None,
        random_crop: bool = False,
        stride: int | None = None,
        data_ids: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.stride = stride
        selected_ids = set(data_ids or [])
        self.files = [
            path
            for path in sorted((self.root / split).glob("*.mat"))
            if _matches_data_id(path, selected_ids)
        ]
        if not self.files:
            raise FileNotFoundError(f"在{self.root / split}下没有找到MAT文件")
        if stride is not None and patch_size is None:
            raise ValueError("设置stride时必须同时设置patch_size")
        if stride is not None and (stride % 8 or patch_size % 8):
            raise ValueError("模拟数据的train_size和stride必须是8的整数倍")

        self.samples: list[tuple[int, int, int]] = []
        self._target_cache: dict[int, Tensor] = {}
        if stride is not None and patch_size is not None:
            for file_index, path in enumerate(self.files):
                image = load_hsi_mat(path, self.dataset_name)
                height = image.shape[0] // 8 * 8
                width = image.shape[1] // 8 * 8
                for top in _grid_positions(height, patch_size, stride):
                    for left in _grid_positions(width, patch_size, stride):
                        self.samples.append((file_index, top, left))

    def __len__(self) -> int:
        return len(self.samples) if self.samples else len(self.files)

    def _load_target(self, file_index: int) -> Tensor:
        if file_index not in self._target_cache:
            image = load_hsi_mat(self.files[file_index], self.dataset_name)
            height = image.shape[0] // 8 * 8
            width = image.shape[1] // 8 * 8
            target = torch.from_numpy(np.ascontiguousarray(image[:height, :width])).permute(2, 0, 1)
            self._target_cache[file_index] = target
        return self._target_cache[file_index]

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if self.samples:
            file_index, fixed_top, fixed_left = self.samples[index]
        else:
            file_index, fixed_top, fixed_left = index, None, None
        full_target = self._load_target(file_index)
        height, width = full_target.shape[-2:]
        full_mosaic, full_pan = simulate_observations(full_target)

        if self.patch_size is None:
            ground_truth = full_target
            mosaic = full_mosaic
            pan = full_pan
        else:
            # PAN/GT坐标以8像素对齐，对应马赛克坐标以4像素对齐，保持MSFA相位。
            if fixed_top is None or fixed_left is None:
                top, left = _aligned_crop_origin(
                    height,
                    width,
                    self.patch_size,
                    alignment=8,
                    random_crop=self.random_crop,
                )
            else:
                top, left = fixed_top, fixed_left
            mosaic_top, mosaic_left = top // 2, left // 2
            mosaic_size = self.patch_size // 2
            ground_truth = full_target[:, top : top + self.patch_size, left : left + self.patch_size]
            pan = full_pan[:, top : top + self.patch_size, left : left + self.patch_size]
            mosaic = full_mosaic[
                :,
                mosaic_top : mosaic_top + mosaic_size,
                mosaic_left : mosaic_left + mosaic_size,
            ]
        return {
            "mosaic": mosaic,
            "pan": pan,
            "target": ground_truth,
            "name": self.files[file_index].stem,
        }


class RealWorldMosaicDataset(Dataset):
    """读取现有12 bit RAW马赛克与PAN配对数据。"""

    def __init__(
        self,
        root: str | Path,
        split: str,
        base_height: int = 255,
        base_width: int = 276,
        msfa_size: int = 4,
        pan_scale_from_mosaic: int = 2,
        patch_size: int | None = None,
        random_crop: bool = False,
        stride: int | None = None,
        data_ids: Sequence[str] | None = None,
    ) -> None:
        split_root = Path(root) / split
        selected_ids = set(data_ids or [])
        self.mosaic_files = [
            path
            for path in sorted((split_root / "mosaic").glob("*.raw"), key=_natural_index)
            if _matches_data_id(path, selected_ids)
        ]
        self.pan_files = [
            path
            for path in sorted((split_root / "pan").glob("*.raw"), key=_natural_index)
            if _matches_data_id(path, selected_ids)
        ]
        if not self.mosaic_files or len(self.mosaic_files) != len(self.pan_files):
            raise ValueError("真实数据的mosaic与pan文件缺失或数量不一致")
        if [item.stem for item in self.mosaic_files] != [item.stem for item in self.pan_files]:
            raise ValueError("真实数据的mosaic与pan编号不一致")
        self.base_height = base_height
        self.base_width = base_width
        self.msfa_size = msfa_size
        self.pan_scale_from_mosaic = pan_scale_from_mosaic
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.stride = stride
        if stride is not None and patch_size is None:
            raise ValueError("设置stride时必须同时设置patch_size")
        alignment = self.msfa_size * self.pan_scale_from_mosaic
        if stride is not None and (stride % alignment or patch_size % alignment):
            raise ValueError(f"真实数据的train_size和stride必须是{alignment}的整数倍")

        self.samples: list[tuple[int, int, int]] = []
        self._pair_cache: dict[int, tuple[Tensor, Tensor]] = {}
        if stride is not None and patch_size is not None:
            pan_height = self.base_height * self.msfa_size * self.pan_scale_from_mosaic
            pan_width = self.base_width * self.msfa_size * self.pan_scale_from_mosaic
            for file_index in range(len(self.mosaic_files)):
                for top in _grid_positions(pan_height, patch_size, stride):
                    for left in _grid_positions(pan_width, patch_size, stride):
                        self.samples.append((file_index, top, left))

    def __len__(self) -> int:
        return len(self.samples) if self.samples else len(self.mosaic_files)

    def _load_pair(self, file_index: int) -> tuple[Tensor, Tensor]:
        if file_index in self._pair_cache:
            return self._pair_cache[file_index]

        mosaic_raw = np.fromfile(self.mosaic_files[file_index], dtype=np.int16)
        expected_mosaic_values = self.base_height * self.base_width * self.msfa_size**2
        if mosaic_raw.size != expected_mosaic_values:
            raise ValueError(f"{self.mosaic_files[file_index]}的元素数量不符合配置")

        mosaic = np.zeros(
            (self.base_height * self.msfa_size, self.base_width * self.msfa_size),
            dtype=np.float32,
        )
        plane_size = self.base_height * self.base_width
        for row in range(self.msfa_size):
            for column in range(self.msfa_size):
                band = row * self.msfa_size + column
                plane = mosaic_raw[band * plane_size : (band + 1) * plane_size]
                mosaic[row::self.msfa_size, column::self.msfa_size] = plane.reshape(
                    self.base_height, self.base_width
                )

        pan_height = mosaic.shape[0] * self.pan_scale_from_mosaic
        pan_width = mosaic.shape[1] * self.pan_scale_from_mosaic
        pan_raw = np.fromfile(self.pan_files[file_index], dtype=np.int16)
        if pan_raw.size != pan_height * pan_width:
            raise ValueError(f"{self.pan_files[file_index]}的元素数量不符合配置")
        pan = pan_raw.reshape(pan_height, pan_width).astype(np.float32)

        mosaic_tensor = torch.from_numpy(mosaic / 4096.0).unsqueeze(0)
        pan_tensor = torch.from_numpy(pan / 4096.0).unsqueeze(0)
        self._pair_cache[file_index] = mosaic_tensor, pan_tensor
        return mosaic_tensor, pan_tensor

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if self.samples:
            file_index, fixed_top, fixed_left = self.samples[index]
        else:
            file_index, fixed_top, fixed_left = index, None, None
        mosaic_tensor, pan_tensor = self._load_pair(file_index)
        # 缓存中的张量只读；切片不会修改底层数据。
        if self.patch_size is not None:
            pan_height, pan_width = pan_tensor.shape[-2:]
            if fixed_top is None or fixed_left is None:
                top, left = _aligned_crop_origin(
                    pan_height,
                    pan_width,
                    self.patch_size,
                    alignment=self.msfa_size * self.pan_scale_from_mosaic,
                    random_crop=self.random_crop,
                )
            else:
                top, left = fixed_top, fixed_left
            mosaic_top = top // self.pan_scale_from_mosaic
            mosaic_left = left // self.pan_scale_from_mosaic
            mosaic_patch_size = self.patch_size // self.pan_scale_from_mosaic
            pan_tensor = pan_tensor[:, top : top + self.patch_size, left : left + self.patch_size]
            mosaic_tensor = mosaic_tensor[
                :,
                mosaic_top : mosaic_top + mosaic_patch_size,
                mosaic_left : mosaic_left + mosaic_patch_size,
            ]

        return {
            "mosaic": mosaic_tensor,
            "pan": pan_tensor,
            "name": self.mosaic_files[file_index].stem,
        }
