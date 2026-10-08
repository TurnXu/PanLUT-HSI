"""CAVE/Chikusei模拟数据监督训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from GetDataSet import SimulatedMosaicDataset
from losses import PanLUTLoss
from quality_index import evaluate_full_reference
from model import PanLUTHSI
from utils import (
    dataset_root_from_readme,
    load_json_config,
    prepare_model_config,
    resolve_device,
    training_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练16波段Pan-LUT模拟数据模型")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--idx", type=int, default=None, help="实验编号；设置后输出到<idx>/model")
    parser.add_argument("--data_path", default=None, help="Dataset根目录")
    parser.add_argument("--dataset", choices=("CAVE", "chikusei"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="GPU编号、cuda:N或cpu")
    parser.add_argument("--resume", default="", help="检查点路径或当前model目录中的epoch编号")
    parser.add_argument("--msfa_size", type=int, default=4)
    parser.add_argument("--spatial_ratio", type=int, default=2)
    parser.add_argument("--num_bands", type=int, default=16)
    parser.add_argument("--noise_level", nargs="+", type=float, default=[0.01, 0.05])
    return parser


def _validate_fixed_task_arguments(args: argparse.Namespace) -> None:
    if (args.msfa_size, args.spatial_ratio, args.num_bands) != (4, 2, 16):
        raise ValueError("本工程固定支持4×4 MSFA、2倍马赛克到PAN倍率和16波段")


@torch.no_grad()
def validate(model: PanLUTHSI, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "sam": 0.0, "ergas": 0.0}
    for batch in loader:
        target = batch["target"].to(device)
        prediction = model(batch["mosaic"].to(device), batch["pan"].to(device))
        for name, value in evaluate_full_reference(prediction, target).items():
            totals[name] += value
    return {name: value / len(loader) for name, value in totals.items()}


def _checkpoint_payload(
    model: PanLUTHSI,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    epoch: int,
    metrics: dict[str, float],
) -> dict:
    state = model.state_dict()
    return {
        "model": state,
        "net": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    _validate_fixed_task_arguments(args)
    project_root = Path(__file__).resolve().parent
    config = load_json_config(project_root, args.config)
    dataset_config = config["dataset"]
    training = config["training"]

    dataset_name = args.dataset or dataset_config["name"]
    dataset_config["name"] = dataset_name
    dataset_root = dataset_root_from_readme(
        project_root, args.data_path, dataset_name, dataset_config["root"]
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

    train_dataset = SimulatedMosaicDataset(
        dataset_root,
        dataset_name,
        dataset_config["train_split"],
        patch_size=train_size,
        stride=stride,
    )
    validation_dataset = SimulatedMosaicDataset(
        dataset_root,
        dataset_name,
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
    criterion = PanLUTLoss(**config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
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
    start_epoch = 1
    best_psnr = -float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            resume_path = checkpoint_dir / f"{args.resume}.pth"
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("metrics", {}).get("psnr", best_psnr))

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        totals: dict[str, float] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            prediction = model(batch["mosaic"].to(device), batch["pan"].to(device))
            loss, components = criterion(prediction, target, model)
            loss.backward()
            optimizer.step()
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value)
        if scheduler is not None:
            scheduler.step()
        totals = {name: value / len(train_loader) for name, value in totals.items()}

        metrics: dict[str, float] = {}
        if epoch % int(training["validation_frequency"]) == 0:
            metrics = validate(model, validation_loader, device)
            print(f"epoch={epoch:04d} loss={totals['total']:.6f} metrics={metrics}")
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                payload = _checkpoint_payload(model, optimizer, scheduler, config, epoch, metrics)
                torch.save(payload, checkpoint_dir / f"best_{epoch}.pth")
                torch.save(payload, checkpoint_dir / "best.pth")
        else:
            print(f"epoch={epoch:04d} loss={totals['total']:.6f}")

        if epoch % save_frequency == 0:
            torch.save(
                _checkpoint_payload(model, optimizer, scheduler, config, epoch, metrics),
                checkpoint_dir / f"{epoch}.pth",
            )


if __name__ == "__main__":
    main()
