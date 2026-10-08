"""真实RAW数据退化一致性训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GetDataSet import RealWorldMosaicDataset
from degradation import LearnableSpectralResponse, build_msfa_kernel, degrade_to_mosaic
from model import PanLUTHSI
from utils import (
    dataset_root_from_readme,
    load_json_config,
    prepare_model_config,
    resolve_device,
    training_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练真实RAW数据的16波段Pan-LUT模型")
    parser.add_argument("--config", default="configs/realworld.json")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--data_path", default=None, help="Dataset根目录")
    parser.add_argument("--dataset", default="real_world")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--visual_freq", type=int, default=1)
    parser.add_argument("--msfa_size", type=int, default=4)
    parser.add_argument("--spatial_ratio", type=int, default=2)
    parser.add_argument("--num_bands", type=int, default=16)
    return parser


@torch.no_grad()
def validate(
    model: PanLUTHSI,
    spectral_response: LearnableSpectralResponse,
    loader: DataLoader,
    msfa_kernel: torch.Tensor,
    device: torch.device,
    preview_path: Path | None = None,
) -> float:
    """以观测重建误差作为真实数据验证指标，数值越小越好。"""
    model.eval()
    spectral_response.eval()
    total = 0.0
    preview = None
    for batch_index, batch in enumerate(loader):
        mosaic = batch["mosaic"].to(device)
        pan = batch["pan"].to(device)
        fused = model(mosaic, pan)
        total += float(
            F.mse_loss(degrade_to_mosaic(fused, msfa_kernel), mosaic)
            + F.mse_loss(spectral_response(fused), pan)
        )
        if batch_index == 0 and preview_path is not None:
            preview = fused[0, [2, 7, 12]].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    if preview is not None:
        import matplotlib.pyplot as plt

        preview_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(preview_path, preview)
    return total / len(loader)


def _checkpoint_payload(
    model: PanLUTHSI,
    spectral_response: LearnableSpectralResponse,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    epoch: int,
    validation_loss: float,
) -> dict:
    state = model.state_dict()
    return {
        "model": state,
        "net": state,
        "spectral_response": spectral_response.state_dict(),
        "degrade_r": spectral_response.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "validation_loss": validation_loss,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset != "real_world":
        raise ValueError("train_realworld.py只支持real_world数据集")
    if (args.msfa_size, args.spatial_ratio, args.num_bands) != (4, 2, 16):
        raise ValueError("本工程固定支持4×4 MSFA、2倍马赛克到PAN倍率和16波段")

    project_root = Path(__file__).resolve().parent
    config = load_json_config(project_root, args.config)
    dataset_config = config["dataset"]
    training = config["training"]
    dataset_root = dataset_root_from_readme(
        project_root, args.data_path, args.dataset, dataset_config["root"]
    )
    train_size = args.train_size or int(dataset_config["train_patch_size"])
    stride = args.stride or train_size
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["learning_rate"] = args.lr
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers
    save_frequency = args.save_freq or int(training.get("save_frequency", 50))

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = resolve_device(args.device)

    train_dataset = RealWorldMosaicDataset(
        dataset_root,
        dataset_config["train_split"],
        patch_size=train_size,
        stride=stride,
    )
    validation_dataset = RealWorldMosaicDataset(
        dataset_root,
        dataset_config["validation_split"],
        patch_size=int(dataset_config["validation_patch_size"]),
        random_crop=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = PanLUTHSI(**prepare_model_config(project_root, config["model"])).to(device)
    spectral_response = LearnableSpectralResponse(config["model"]["spectral_bands"]).to(device)
    msfa_kernel = build_msfa_kernel().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(spectral_response.parameters()),
        lr=float(training["learning_rate"]),
    )
    use_scheduler = args.lr_decay or bool(training.get("lr_decay", True))
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training["scheduler_step"]),
            gamma=float(training["scheduler_gamma"]),
        )
        if use_scheduler
        else None
    )

    checkpoint_dir = training_output_dir(project_root, args.idx, training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_dir = checkpoint_dir.parent / "record"
    start_epoch = 1
    best_validation = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = checkpoint_dir / f"{args.resume}.pth"
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        spectral_response.load_state_dict(checkpoint["spectral_response"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("validation_loss", best_validation))

    consistency_weight = float(training["consistency_weight"])
    smoothness_weight = float(config["loss"]["smoothness_weight"])
    monotonicity_weight = float(config["loss"]["monotonicity_weight"])

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        spectral_response.train()
        total_loss = 0.0
        for batch in train_loader:
            mosaic = batch["mosaic"].to(device)
            pan = batch["pan"].to(device)
            optimizer.zero_grad(set_to_none=True)
            fused_first = model(mosaic, pan)
            reconstructed_mosaic = degrade_to_mosaic(fused_first, msfa_kernel)
            reconstructed_pan = spectral_response(fused_first)
            fused_second = model(reconstructed_mosaic, reconstructed_pan)
            observation_loss = F.mse_loss(reconstructed_mosaic, mosaic) + F.mse_loss(
                reconstructed_pan, pan
            )
            consistency_loss = F.mse_loss(fused_second, fused_first.detach())
            smoothness, monotonicity = model.lut_regularization()
            loss = (
                observation_loss
                + consistency_weight * consistency_loss
                + smoothness_weight * smoothness
                + monotonicity_weight * monotonicity
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        if scheduler is not None:
            scheduler.step()

        validation_loss = float("inf")
        if epoch % int(training["validation_frequency"]) == 0:
            preview_path = None
            if args.visual and epoch % args.visual_freq == 0:
                preview_path = record_dir / f"fused_{epoch}.png"
            validation_loss = validate(
                model,
                spectral_response,
                validation_loader,
                msfa_kernel,
                device,
                preview_path,
            )
            print(
                f"epoch={epoch:04d} loss={total_loss / len(train_loader):.6f} "
                f"validation={validation_loss:.6f}"
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                payload = _checkpoint_payload(
                    model,
                    spectral_response,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    validation_loss,
                )
                torch.save(payload, checkpoint_dir / f"best_{epoch}.pth")
                torch.save(payload, checkpoint_dir / "best.pth")

        if epoch % save_frequency == 0:
            torch.save(
                _checkpoint_payload(
                    model,
                    spectral_response,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    validation_loss,
                ),
                checkpoint_dir / f"{epoch}.pth",
            )


if __name__ == "__main__":
    main()
