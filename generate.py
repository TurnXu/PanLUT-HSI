"""模拟数据或真实RAW数据的统一生成入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GetDataSet import RealWorldMosaicDataset, SimulatedMosaicDataset
from model import PanLUTHSI
from utils import (
    checkpoint_model_state,
    prepare_model_config,
    resolve_device,
    resolve_path,
    result_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成Pan-LUT-HSI融合结果")
    parser.add_argument("--checkpoint", default=None, help="新式检查点参数")
    parser.add_argument("--load_model", default=None, help="README兼容检查点参数")
    parser.add_argument("--data-root", default=None, help="具体数据集目录")
    parser.add_argument("--data_path", default=None, help="Dataset根目录")
    parser.add_argument("--dataset", choices=("CAVE", "chikusei", "real_world"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--mosaic_save", action="store_true")
    parser.add_argument("--pan_save", action="store_true")
    parser.add_argument("--demosaic_save", action="store_true")
    parser.add_argument("--gt_save", action="store_true")
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--msfa_size", type=int, default=4)
    parser.add_argument("--spatial_ratio", type=int, default=2)
    parser.add_argument("--num_bands", type=int, default=16)
    return parser


def _save_mat(root: Path, category: str, name: str, value: np.ndarray) -> None:
    directory = root / "mat" / category
    directory.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(directory / f"{name}.mat", {category: value})


def _resolve_mode(args: argparse.Namespace) -> bool:
    if args.real_world and args.simulate:
        raise ValueError("--simulate和--real_world不能同时使用")
    real_world = args.real_world or args.dataset == "real_world"
    if args.simulate and args.dataset == "real_world":
        raise ValueError("real_world数据集不能使用--simulate")
    if real_world and args.dataset != "real_world":
        raise ValueError("--real_world只支持real_world数据集")
    return real_world


def main() -> None:
    args = build_parser().parse_args()
    if (args.msfa_size, args.spatial_ratio, args.num_bands) != (4, 2, 16):
        raise ValueError("本工程固定支持4×4 MSFA、2倍马赛克到PAN倍率和16波段")
    real_world = _resolve_mode(args)
    project_root = Path(__file__).resolve().parent
    checkpoint_value = args.load_model or args.checkpoint
    if not checkpoint_value:
        raise ValueError("必须通过--load_model或--checkpoint指定模型")
    checkpoint_path = resolve_path(project_root, checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint or "model" not in checkpoint["config"]:
        raise KeyError("检查点缺少本工程所需的config.model配置")

    device = resolve_device(args.device, force_cpu=args.cpu)
    model_config = prepare_model_config(project_root, checkpoint["config"]["model"])
    model = PanLUTHSI(**model_config)
    model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    model.to(device).eval()

    if args.data_path:
        data_root = resolve_path(project_root, args.data_path) / args.dataset
    elif args.data_root:
        data_root = resolve_path(project_root, args.data_root)
    else:
        raise ValueError("必须通过--data_path或--data-root指定数据目录")

    if real_world:
        dataset = RealWorldMosaicDataset(
            data_root, args.split, data_ids=args.data_id
        )
    else:
        dataset = SimulatedMosaicDataset(
            root=data_root,
            dataset_name=args.dataset,
            split=args.split,
            patch_size=None,
            random_crop=False,
            data_ids=args.data_id,
        )

    output_root = result_output_dir(
        project_root, args.dataset, args.idx, args.output
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(checkpoint["config"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    legacy_mode = args.load_model is not None or args.data_path is not None
    save_all_observations = not legacy_mode
    save_mosaic = args.mosaic_save or save_all_observations
    save_pan = args.pan_save or save_all_observations
    save_gt = args.gt_save or save_all_observations
    if args.demosaic_save:
        print("提示：当前模型直接输出融合结果，--demosaic_save作为兼容参数保留。")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            mosaic_tensor = batch["mosaic"].to(device)
            pan_tensor = batch["pan"].to(device)
            prediction = model(mosaic_tensor, pan_tensor)
            name = batch["name"][0]

            fused = prediction[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
            low_hsi = F.pixel_unshuffle(batch["mosaic"], 4)[0].permute(1, 2, 0).numpy()
            pan = batch["pan"][0].permute(1, 2, 0).numpy()
            if real_world:
                fused = np.clip(np.rint(fused * 4096.0), 0, 4096).astype(np.int16)
                low_hsi = np.clip(np.rint(low_hsi * 4096.0), 0, 4096).astype(np.int16)
                pan = np.clip(np.rint(pan * 4096.0), 0, 4096).astype(np.int16)

            _save_mat(output_root, "fused", name, fused)
            if save_mosaic:
                _save_mat(output_root, "mosaic", name, low_hsi)
            if save_pan:
                _save_mat(output_root, "pan", name, pan)
            if save_gt and "target" in batch:
                target = batch["target"][0].permute(1, 2, 0).numpy()
                _save_mat(output_root, "gt", name, target)
            print(f"saved: {name}")


if __name__ == "__main__":
    main()
