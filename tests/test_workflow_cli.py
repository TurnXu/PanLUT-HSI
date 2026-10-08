"""README兼容命令行的解析测试。"""

import unittest

from generate import build_parser as build_generate_parser
from test import build_parser as build_test_parser
from train_realworld import build_parser as build_realworld_parser
from train_simulate import build_parser as build_simulate_parser
from visualize import build_parser as build_visualize_parser


class ReadmeCliTest(unittest.TestCase):
    def test_simulated_training_command(self) -> None:
        args = build_simulate_parser().parse_args(
            "--idx 1 --data_path ../Dataset --dataset CAVE --epochs 1000 "
            "--train_size 64 --stride 32 --batch_size 16 --lr_decay "
            "--save_freq 50 --device 0".split()
        )
        self.assertEqual(args.dataset, "CAVE")
        self.assertEqual(args.stride, 32)

    def test_realworld_training_command(self) -> None:
        args = build_realworld_parser().parse_args(
            "--idx 3 --data_path ../Dataset --dataset real_world --epochs 10 "
            "--train_size 64 --stride 64 --batch_size 16 --visual --device 0".split()
        )
        self.assertTrue(args.visual)
        self.assertEqual(args.dataset, "real_world")

    def test_generate_single_command(self) -> None:
        args = build_generate_parser().parse_args(
            "--idx 1 --simulate --mosaic_save --pan_save --gt_save "
            "--data_path ../Dataset --dataset CAVE --load_model 1/model/best.pth "
            "--data_id watercolors_ms.mat".split()
        )
        self.assertEqual(args.data_id, ["watercolors_ms.mat"])
        self.assertTrue(args.simulate)

    def test_evaluation_and_visualization_commands(self) -> None:
        test_args = build_test_parser().parse_args(
            "--idx 1 --simulate --data_path CAVE/1/result/mat "
            "--data_id oil_painting_ms.mat".split()
        )
        visual_args = build_visualize_parser().parse_args(
            "--visual_task rgb --spatial_ratio 8 --mat_path CAVE/1/result/mat/fused "
            "--save_path CAVE/1/result/rgb/fused --data_type fused "
            "--data_id watercolors_ms.mat --detach --detach_size 50 50 "
            "--detach_coordinate 160 210 160 210 --boxcolor b --boxwidth 2".split()
        )
        self.assertTrue(test_args.simulate)
        self.assertTrue(visual_args.detach)
        self.assertEqual(visual_args.detach_coordinate, [160, 210, 160, 210])


if __name__ == "__main__":
    unittest.main()
