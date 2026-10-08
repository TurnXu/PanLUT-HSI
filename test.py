"""评价generate.py生成的模拟或真实数据结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io
import torch

from quality_index import calc_q2n, evaluate_full_reference, qnr_mosaic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评价Pan-LUT-HSI生成结果")
    parser.add_argument("--result-root", default=None, help="包含mat子目录的结果目录")
    parser.add_argument("--mode", choices=("simulated", "real"), default=None)
    parser.add_argument("--data_path", default=None, help="README兼容：直接指向mat目录")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--idx", type=int, default=1, help="兼容参数，不参与路径拼接")
    parser.add_argument("--device", default="0", help="兼容参数；当前指标可在CPU计算")
    parser.add_argument("--cpu", action="store_true")
    return parser


def _load(path: Path, key: str) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    content = scipy.io.loadmat(path)
    if key not in content:
        raise KeyError(f"{path}中不存在变量{key}")
    array = content[key]
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()


def _selected_files(directory: Path, data_ids: list[str]) -> list[Path]:
    selected = set(data_ids)
    files = [
        path
        for path in sorted(directory.glob("*.mat"))
        if not selected or path.name in selected or path.stem in selected
    ]
    if not files:
        raise FileNotFoundError(f"{directory}中没有匹配的MAT结果")
    return files


def main() -> None:
    args = build_parser().parse_args()
    if args.simulate and args.real_world:
        raise ValueError("--simulate和--real_world不能同时使用")
    if args.data_path:
        mat_root = Path(args.data_path).resolve()
    elif args.result_root:
        mat_root = Path(args.result_root).resolve() / "mat"
    else:
        raise ValueError("必须通过--data_path或--result-root指定结果目录")

    if args.simulate:
        mode = "simulated"
    elif args.real_world:
        mode = "real"
    else:
        mode = args.mode or "simulated"

    fused_files = _selected_files(mat_root / "fused", args.data_id)
    totals: dict[str, float] = {}
    for fused_path in fused_files:
        name = fused_path.stem
        fused = _load(fused_path, "fused")
        if mode == "simulated":
            target = _load(mat_root / "gt" / f"{name}.mat", "gt")
            values = evaluate_full_reference(fused, target)
            block_size = 256 if min(target.shape[-2:]) > 1000 else 64
            q2n_value, _ = calc_q2n(
                target[0], fused[0], Q_blocks_size=block_size, Q_shift=block_size // 2
            )
            values["q2n"] = float(q2n_value)
        else:
            values = qnr_mosaic(
                fused,
                _load(mat_root / "mosaic" / f"{name}.mat", "mosaic"),
                _load(mat_root / "pan" / f"{name}.mat", "pan"),
                patch_size=args.patch_size,
            )
        print(name, values)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value

    averages = {key: value / len(fused_files) for key, value in totals.items()}
    print("average", averages)


if __name__ == "__main__":
    main()
