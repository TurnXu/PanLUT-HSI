"""统计PanLUT-HSI模型参数量与理论运算规模。"""

from __future__ import annotations

import argparse
from pathlib import Path

from flops_counter import count_parameters, estimate_panlut_operations, format_count
from model import PanLUTHSI
from utils import load_json_config, prepare_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计PanLUT-HSI参数量和查表规模")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512, help="PAN高度")
    parser.add_argument("--width", type=int, default=512, help="PAN宽度")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_json_config(project_root, args.config)
    model = PanLUTHSI(**prepare_model_config(project_root, config["model"]))
    total, trainable = count_parameters(model)
    operations = estimate_panlut_operations(
        model, args.batch_size, args.height, args.width
    )
    print(f"total parameters: {total} ({format_count(total)})")
    print(f"trainable parameters: {trainable} ({format_count(trainable)})")
    for name, value in operations.items():
        print(f"{name}: {value} ({format_count(value)})")
    print("说明：LUT采用多线性插值，顶点访问次数不能等同于卷积FLOPs。")


if __name__ == "__main__":
    main()
