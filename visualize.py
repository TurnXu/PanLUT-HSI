"""把生成的MAT结果可视化为RGB、PAN或MAE/SAM误差图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


RGB_BANDS = (2, 7, 12)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化Pan-LUT-HSI的MAT结果")
    parser.add_argument("--visual_task", choices=("rgb", "diffmap"), default=None)
    parser.add_argument("--spatial_ratio", type=int, default=8)
    parser.add_argument("--mat_path", default=None)
    parser.add_argument("--mat_path_for_diff", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--data_type", default="fused")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--detach_size", type=int, nargs="+", default=[32])
    parser.add_argument("--detach_coordinate", type=int, nargs="+", default=[0, 0])
    parser.add_argument("--boxwidth", type=int, default=5)
    parser.add_argument("--boxcolor", choices=("r", "b"), default="r")
    parser.add_argument("--mae_level", type=int, default=16)
    parser.add_argument("--sam_level", type=int, default=16)
    # 保留本工程原有简化入口。
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bands", type=int, nargs=3, default=list(RGB_BANDS))
    return parser


def _selected_files(directory: Path, data_ids: list[str]) -> list[Path]:
    selected = set(data_ids)
    files = [
        path
        for path in sorted(directory.glob("*.mat"))
        if not selected or path.name in selected or path.stem in selected
    ]
    if not files:
        raise FileNotFoundError(f"{directory}中没有匹配的MAT文件")
    return files


def _load_with_fallback(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    content = scipy.io.loadmat(path)
    for key in keys:
        if key in content:
            return content[key]
    raise KeyError(f"{path}中不存在变量{keys}")


def _normalize_for_display(array: np.ndarray, brighten: bool = False) -> np.ndarray:
    image = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) or float(np.nanmax(image)) > 2.0:
        image /= 4096.0
        brighten = True
    if brighten:
        image *= 2.0
    return np.clip(image, 0.0, 1.0)


def _resize_cube(array: np.ndarray, scale: int) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, scale_factor=scale, mode="bicubic", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy()


def _region_parameters(size: list[int], coordinate: list[int]) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    if len(size) == 1:
        size = [size[0], size[0]]
    if len(size) != 2:
        raise ValueError("detach_size需要1或2个整数")
    if len(coordinate) == 2:
        coordinate = [coordinate[0], coordinate[1], coordinate[0], coordinate[1]]
    if len(coordinate) != 4:
        raise ValueError("detach_coordinate需要2或4个整数")
    return (size[0], size[1]), tuple(coordinate)


def _draw_rectangle(image: np.ndarray, top: int, left: int, height: int, width: int, line_width: int, color: np.ndarray) -> None:
    bottom = top + height
    right = left + width
    if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
        raise ValueError("局部放大区域超出图像边界")
    image[top : top + line_width, left:right] = color
    image[bottom - line_width : bottom, left:right] = color
    image[top:bottom, left : left + line_width] = color
    image[top:bottom, right - line_width : right] = color


def _save_detached(
    image: np.ndarray,
    output: Path,
    stem: str,
    size_value: list[int],
    coordinate_value: list[int],
    line_width: int,
    first_color: str,
) -> None:
    size, coordinates = _region_parameters(size_value, coordinate_value)
    height, width = size
    top1, left1, top2, left2 = coordinates
    first = np.array([255, 0, 0] if first_color == "r" else [0, 0, 255], dtype=np.uint8)
    second = np.array([0, 0, 255] if first_color == "r" else [255, 0, 0], dtype=np.uint8)
    boxed = image.copy()
    _draw_rectangle(boxed, top1, left1, height, width, line_width, first)
    _draw_rectangle(boxed, top2, left2, height, width, line_width, second)
    plt.imsave(output / f"{stem}_box.png", boxed)
    plt.imsave(output / f"detach_{stem}_{top1}_{left1}.png", image[top1 : top1 + height, left1 : left1 + width])
    plt.imsave(output / f"detach_{stem}_{top2}_{left2}.png", image[top2 : top2 + height, left2 : left2 + width])


def _rgb_image(path: Path, data_type: str, spatial_ratio: int, bands: list[int]) -> np.ndarray:
    if data_type == "fused":
        array = _load_with_fallback(path, ("fused",))
    elif data_type in {"mosaic", "gt"}:
        array = _load_with_fallback(path, (data_type, "gt", "mosaic"))
    elif data_type == "upmosaic":
        array = _resize_cube(_load_with_fallback(path, ("mosaic",)), spatial_ratio)
    elif data_type == "pan":
        array = _load_with_fallback(path, ("pan",))
    else:
        raise ValueError(f"不支持的data_type：{data_type}")

    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] == 1:
        rgb = np.repeat(array, 3, axis=2)
    else:
        if max(bands) >= array.shape[2]:
            raise ValueError(f"RGB波段{bands}超出数据的{array.shape[2]}个通道")
        rgb = array[:, :, bands]
    return (_normalize_for_display(rgb, "chikusei" in str(path).lower()) * 255).round().astype(np.uint8)


def visualize_rgb(args: argparse.Namespace, source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in _selected_files(source, args.data_id):
        image = _rgb_image(path, args.data_type, args.spatial_ratio, args.bands)
        stem = f"{args.data_type}_{path.stem}"
        destination = output / f"{stem}.png"
        plt.imsave(destination, image)
        if args.detach:
            _save_detached(
                image,
                output,
                stem,
                args.detach_size,
                args.detach_coordinate,
                args.boxwidth,
                args.boxcolor,
            )
        print(f"saved: {destination}")


def _colored_error(error: np.ndarray, upper_boundary: float) -> np.ndarray:
    normalized = np.clip(error / max(upper_boundary, 1e-8), 0.0, 1.0)
    return (plt.get_cmap("jet")(normalized)[..., :3] * 255).round().astype(np.uint8)


def visualize_diffmap(args: argparse.Namespace, fused_root: Path, target_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for fused_path in _selected_files(fused_root, args.data_id):
        target_path = target_root / fused_path.name
        fused = _load_with_fallback(fused_path, ("fused",)).astype(np.float32)
        target = _load_with_fallback(target_path, ("gt",)).astype(np.float32)
        if fused.shape != target.shape:
            raise ValueError(f"{fused_path.name}的fused与gt形状不一致")

        mae = np.mean(np.abs(target - fused), axis=2) * 255.0
        dot = np.sum(target * fused, axis=2)
        denominator = np.linalg.norm(target, axis=2) * np.linalg.norm(fused, axis=2)
        cosine = np.clip(dot / np.maximum(denominator, 1e-8), -1.0, 1.0)
        sam = np.arccos(cosine) / np.pi * 255.0
        for kind, values, level in (
            ("mae", mae, args.mae_level),
            ("sam", sam, args.sam_level),
        ):
            image = _colored_error(values, level)
            stem = f"{kind}_{fused_path.stem}"
            destination = output / f"{stem}.png"
            plt.imsave(destination, image)
            if args.detach:
                _save_detached(
                    image,
                    output,
                    stem,
                    args.detach_size,
                    args.detach_coordinate,
                    args.boxwidth,
                    args.boxcolor,
                )
            print(f"saved: {destination}")


def main() -> None:
    args = build_parser().parse_args()
    task = args.visual_task or ("diffmap" if args.data_type in {"mae", "diffmap"} else "rgb")
    if args.mat_path:
        source = Path(args.mat_path).resolve()
    elif args.result_root:
        source_type = "fused" if task == "diffmap" else args.data_type
        source = Path(args.result_root).resolve() / "mat" / source_type
    else:
        raise ValueError("必须通过--mat_path或--result-root指定MAT数据")

    if args.save_path:
        output = Path(args.save_path).resolve()
    elif args.output:
        output = Path(args.output).resolve()
    elif args.result_root:
        output = Path(args.result_root).resolve() / "rgb" / args.data_type
    else:
        output = source.parent.parent / "rgb" / args.data_type

    if task == "rgb":
        visualize_rgb(args, source, output)
    else:
        if args.mat_path_for_diff:
            target = Path(args.mat_path_for_diff).resolve()
        elif args.result_root:
            target = Path(args.result_root).resolve() / "mat" / "gt"
        else:
            raise ValueError("diffmap必须指定--mat_path_for_diff")
        visualize_diffmap(args, source, target, output)


if __name__ == "__main__":
    main()
