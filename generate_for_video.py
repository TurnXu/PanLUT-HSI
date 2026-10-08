"""从包含多帧的真实RAW文件生成逐帧PanLUT-HSI结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from model import PanLUTHSI
from utils import (
    checkpoint_model_state,
    prepare_model_config,
    resolve_device,
    resolve_path,
    result_output_dir,
)


BASE_HEIGHT = 255
BASE_WIDTH = 276
MSFA_SIZE = 4
PAN_SCALE_FROM_MOSAIC = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成真实多帧RAW的融合结果")
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--real_world_video", action="store_true")
    parser.add_argument("--frames", type=int, nargs="+", default=[])
    parser.add_argument("--mosaic_save", action="store_true")
    parser.add_argument("--pan_save", action="store_true")
    parser.add_argument("--demosaic_save", action="store_true")
    parser.add_argument("--dataset", default="real_world")
    parser.add_argument("--data_path", default="../Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--load_model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="0")
    return parser


def _save(root: Path, category: str, name: str, value: np.ndarray) -> None:
    directory = root / "mat" / category
    directory.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(directory / f"{name}.mat", {category: value})


def _decode_mosaic_frame(raw: np.ndarray, frame_index: int) -> np.ndarray:
    plane_size = BASE_HEIGHT * BASE_WIDTH
    frame_size = plane_size * MSFA_SIZE**2
    start = frame_index * frame_size
    frame = raw[start : start + frame_size]
    mosaic = np.zeros((BASE_HEIGHT * MSFA_SIZE, BASE_WIDTH * MSFA_SIZE), dtype=np.int16)
    for row in range(MSFA_SIZE):
        for column in range(MSFA_SIZE):
            band = row * MSFA_SIZE + column
            plane = frame[band * plane_size : (band + 1) * plane_size]
            mosaic[row::MSFA_SIZE, column::MSFA_SIZE] = plane.reshape(BASE_HEIGHT, BASE_WIDTH)
    return mosaic


def _decode_pan_frame(raw: np.ndarray, frame_index: int) -> np.ndarray:
    height = BASE_HEIGHT * MSFA_SIZE * PAN_SCALE_FROM_MOSAIC
    width = BASE_WIDTH * MSFA_SIZE * PAN_SCALE_FROM_MOSAIC
    frame_size = height * width
    start = frame_index * frame_size
    return raw[start : start + frame_size].reshape(height, width)


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset != "real_world":
        raise ValueError("多帧RAW入口当前只支持real_world")
    project_root = Path(__file__).resolve().parent
    device = resolve_device(args.device, force_cpu=args.cpu)
    checkpoint = torch.load(resolve_path(project_root, args.load_model), map_location="cpu")
    model = PanLUTHSI(
        **prepare_model_config(project_root, checkpoint["config"]["model"])
    ).to(device)
    model.load_state_dict(checkpoint_model_state(checkpoint))
    model.eval()

    split_root = resolve_path(project_root, args.data_path) / args.dataset / args.split
    selected = set(args.data_id)
    mosaic_files = [
        path
        for path in sorted((split_root / "mosaic").glob("*.raw"))
        if not selected or path.name in selected or path.stem in selected
    ]
    pan_files = [
        path
        for path in sorted((split_root / "pan").glob("*.raw"))
        if not selected or path.name in selected or path.stem in selected
    ]
    if not mosaic_files or [path.stem for path in mosaic_files] != [path.stem for path in pan_files]:
        raise ValueError("多帧mosaic与pan文件缺失或编号不一致")

    output = result_output_dir(project_root, args.dataset, args.idx, args.output)
    mosaic_frame_size = BASE_HEIGHT * BASE_WIDTH * MSFA_SIZE**2
    pan_frame_size = (
        BASE_HEIGHT
        * BASE_WIDTH
        * (MSFA_SIZE * PAN_SCALE_FROM_MOSAIC) ** 2
    )
    with torch.no_grad():
        for mosaic_path, pan_path in zip(mosaic_files, pan_files):
            mosaic_raw = np.fromfile(mosaic_path, dtype=np.int16)
            pan_raw = np.fromfile(pan_path, dtype=np.int16)
            if mosaic_raw.size % mosaic_frame_size or pan_raw.size % pan_frame_size:
                raise ValueError(f"{mosaic_path.name}的RAW长度不是完整帧倍数")
            mosaic_frames = mosaic_raw.size // mosaic_frame_size
            pan_frames = pan_raw.size // pan_frame_size
            if mosaic_frames != pan_frames:
                raise ValueError(f"{mosaic_path.name}的mosaic与pan帧数不一致")
            frame_indices = args.frames or list(range(mosaic_frames))
            for frame_index in frame_indices:
                if frame_index < 0 or frame_index >= mosaic_frames:
                    raise IndexError(
                        f"帧{frame_index}越界，{mosaic_path.name}共有{mosaic_frames}帧"
                    )
                mosaic = _decode_mosaic_frame(mosaic_raw, frame_index)
                pan = _decode_pan_frame(pan_raw, frame_index)
                mosaic_tensor = torch.from_numpy(mosaic.astype(np.float32) / 4096.0)[None, None].to(device)
                pan_tensor = torch.from_numpy(pan.astype(np.float32) / 4096.0)[None, None].to(device)
                fused = model(mosaic_tensor, pan_tensor)[0].permute(1, 2, 0)
                fused = torch.clamp(torch.round(fused * 4096.0), 0, 4096).short().cpu().numpy()
                name = f"{mosaic_path.stem}_{frame_index}"
                _save(output, "fused", name, fused)
                if args.mosaic_save:
                    low_hsi = F.pixel_unshuffle(mosaic_tensor.cpu(), 4)[0].permute(1, 2, 0)
                    low_hsi = torch.round(low_hsi * 4096.0).short().numpy()
                    _save(output, "mosaic", name, low_hsi)
                if args.pan_save:
                    _save(output, "pan", name, pan[:, :, None])
                print(f"saved: {name}")


if __name__ == "__main__":
    main()
