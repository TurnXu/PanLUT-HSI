"""可视化融合结果退化前后的MSFA与PAN观测。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from GetDataSet import PAN_SPECTRAL_RESPONSE
from degradation import LearnableSpectralResponse, build_msfa_kernel, degrade_to_mosaic
from utils import resolve_device, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化PanLUT-HSI退化一致性")
    parser.add_argument("--data_path", required=True, help="generate结果的mat目录")
    parser.add_argument("--save_path", default="")
    parser.add_argument("--dataset", choices=("CAVE", "chikusei", "real_world"), default="CAVE")
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--load_model", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="0")
    return parser


def _load(path: Path, key: str) -> np.ndarray:
    content = scipy.io.loadmat(path)
    if key not in content:
        raise KeyError(f"{path}中不存在变量{key}")
    value = content[key]
    if value.ndim == 2:
        value = value[:, :, None]
    return value


def _normalized(array: np.ndarray) -> np.ndarray:
    value = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) or float(value.max()) > 2.0:
        value /= 4096.0
    return np.clip(value, 0.0, 1.0)


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    result = F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False)
    return result[0].permute(1, 2, 0).numpy()


def _rgb(cube: np.ndarray, height: int, width: int) -> np.ndarray:
    image = _resize(cube[:, :, [2, 7, 12]], height, width)
    return np.clip(image * 2.0, 0.0, 1.0)


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    mat_root = resolve_path(project_root, args.data_path)
    output = resolve_path(
        project_root,
        args.save_path or str(mat_root.parent / "rgb" / "degradation"),
    )
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device, force_cpu=args.cpu)

    spectral_response = LearnableSpectralResponse(16).to(device)
    if args.load_model:
        checkpoint = torch.load(resolve_path(project_root, args.load_model), map_location="cpu")
        if "spectral_response" in checkpoint:
            spectral_response.load_state_dict(checkpoint["spectral_response"])
        elif args.dataset == "real_world":
            raise KeyError("真实数据检查点缺少spectral_response")
    elif args.dataset != "real_world":
        with torch.no_grad():
            weights = torch.tensor(PAN_SPECTRAL_RESPONSE).view(1, 16, 1, 1)
            spectral_response.projection.weight.copy_(weights)
    spectral_response.eval()
    kernel = build_msfa_kernel().to(device)

    selected = set(args.data_id)
    fused_files = [
        path
        for path in sorted((mat_root / "fused").glob("*.mat"))
        if not selected or path.name in selected or path.stem in selected
    ]
    if not fused_files:
        raise FileNotFoundError("没有匹配的fused MAT文件")

    with torch.no_grad():
        for fused_path in fused_files:
            name = fused_path.stem
            fused = _normalized(_load(fused_path, "fused"))
            observed_mosaic = _normalized(_load(mat_root / "mosaic" / fused_path.name, "mosaic"))
            observed_pan = _normalized(_load(mat_root / "pan" / fused_path.name, "pan"))
            fused_tensor = torch.from_numpy(fused).permute(2, 0, 1).unsqueeze(0).to(device)
            raw_mosaic = degrade_to_mosaic(fused_tensor, kernel)
            reconstructed_mosaic = F.pixel_unshuffle(raw_mosaic, 4)[0].permute(1, 2, 0).cpu().numpy()
            reconstructed_pan = spectral_response(fused_tensor)[0, 0].cpu().numpy()
            height, width = fused.shape[:2]

            images = {
                "fused": _rgb(fused, height, width),
                "mosaic": _rgb(observed_mosaic, height, width),
                "mosaic_degrade": _rgb(reconstructed_mosaic, height, width),
                "pan": np.repeat(observed_pan[:, :, :1], 3, axis=2),
                "pan_degrade": np.repeat(reconstructed_pan[:, :, None], 3, axis=2),
            }
            for category, image in images.items():
                destination = output / f"{category}_{name}.png"
                plt.imsave(destination, np.clip(image, 0.0, 1.0))
                print(f"saved: {destination}")


if __name__ == "__main__":
    main()
