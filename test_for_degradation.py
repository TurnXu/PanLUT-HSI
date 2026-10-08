"""验证融合结果退化回MSFA马赛克和PAN后的观测一致性。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GetDataSet import (
    PAN_SPECTRAL_RESPONSE,
    RealWorldMosaicDataset,
    SimulatedMosaicDataset,
)
from degradation import LearnableSpectralResponse, build_msfa_kernel, degrade_to_mosaic
from model import PanLUTHSI
from utils import checkpoint_model_state, prepare_model_config, resolve_device, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试PanLUT-HSI退化一致性")
    parser.add_argument("--dataset", choices=("CAVE", "chikusei", "real_world"), default="CAVE")
    parser.add_argument("--data_path", default="../Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_id", nargs="+", default=[])
    parser.add_argument("--load_model", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="0")
    return parser


def _fixed_pan_response(high_resolution_hsi: torch.Tensor) -> torch.Tensor:
    response = high_resolution_hsi.new_tensor(PAN_SPECTRAL_RESPONSE).view(1, 16, 1, 1)
    return (high_resolution_hsi * response).sum(dim=1, keepdim=True)


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = resolve_path(project_root, args.data_path) / args.dataset
    real_world = args.real_world or args.dataset == "real_world"
    if real_world:
        dataset = RealWorldMosaicDataset(data_root, args.split, data_ids=args.data_id)
    else:
        dataset = SimulatedMosaicDataset(
            data_root, args.dataset, args.split, data_ids=args.data_id
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = resolve_device(args.device, force_cpu=args.cpu)
    checkpoint_value = args.load_model or args.checkpoint
    model = None
    spectral_response = None
    if checkpoint_value:
        checkpoint = torch.load(resolve_path(project_root, checkpoint_value), map_location="cpu")
        model = PanLUTHSI(
            **prepare_model_config(project_root, checkpoint["config"]["model"])
        ).to(device)
        model.load_state_dict(checkpoint_model_state(checkpoint))
        model.eval()
        if real_world:
            if "spectral_response" not in checkpoint:
                raise KeyError("真实数据退化测试需要检查点中的spectral_response")
            spectral_response = LearnableSpectralResponse(16).to(device)
            spectral_response.load_state_dict(checkpoint["spectral_response"])
            spectral_response.eval()
    elif real_world:
        raise ValueError("真实数据退化测试必须指定--load_model")

    kernel = build_msfa_kernel().to(device)
    total_mosaic = 0.0
    total_pan = 0.0
    with torch.no_grad():
        for batch in loader:
            observed_mosaic = batch["mosaic"].to(device)
            observed_pan = batch["pan"].to(device)
            if model is None:
                fused = batch["target"].to(device)
            else:
                fused = model(observed_mosaic, observed_pan)
            reconstructed_mosaic = degrade_to_mosaic(fused, kernel)
            reconstructed_pan = (
                spectral_response(fused)
                if spectral_response is not None
                else _fixed_pan_response(fused)
            )
            mosaic_error = F.mse_loss(reconstructed_mosaic, observed_mosaic).item()
            pan_error = F.mse_loss(reconstructed_pan, observed_pan).item()
            total_mosaic += mosaic_error
            total_pan += pan_error
            print(batch["name"][0], {"mosaic_mse": mosaic_error, "pan_mse": pan_error})
    print(
        "average",
        {
            "mosaic_mse": total_mosaic / len(loader),
            "pan_mse": total_pan / len(loader),
        },
    )


if __name__ == "__main__":
    main()
