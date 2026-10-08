"""平铺后辅助入口与统计工具测试。"""

import unittest

from degrade_visualize import build_parser as build_degrade_visualize_parser
from flops_counter import count_parameters, estimate_panlut_operations
from generate_for_video import build_parser as build_video_parser
from get_param_and_flops import build_parser as build_flops_parser
from model import PanLUTHSI
from test_for_degradation import build_parser as build_degradation_test_parser


class AuxiliaryToolsTest(unittest.TestCase):
    def test_parameter_and_lut_operation_counts(self) -> None:
        model = PanLUTHSI(bins=3, chunk_pixels=64)
        total, trainable = count_parameters(model)
        operations = estimate_panlut_operations(model, 1, 32, 40)
        self.assertEqual(total, trainable)
        self.assertGreater(total, 0)
        self.assertGreater(operations["lut_vertex_reads"], operations["convolution_macs"])

    def test_auxiliary_readme_parsers(self) -> None:
        degradation = build_degradation_test_parser().parse_args(
            "--dataset CAVE --data_path ../Dataset --simulate".split()
        )
        visualization = build_degrade_visualize_parser().parse_args(
            "--data_path CAVE/1/result/mat --dataset CAVE".split()
        )
        video = build_video_parser().parse_args(
            "--real_world_video --dataset real_world --data_path ../Dataset "
            "--load_model 3/model/best.pth --frames 0 1".split()
        )
        flops = build_flops_parser().parse_args("--height 512 --width 512".split())
        self.assertTrue(degradation.simulate)
        self.assertEqual(visualization.dataset, "CAVE")
        self.assertEqual(video.frames, [0, 1])
        self.assertEqual(flops.height, 512)


if __name__ == "__main__":
    unittest.main()
