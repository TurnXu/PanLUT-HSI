"""不依赖真实数据的最小模型正确性测试。"""

import unittest

import torch

from lut_layers import MultiLinearLUT
from model import PanLUTHSI


class MultiLinearLUTTest(unittest.TestCase):
    def test_identity_mapping(self) -> None:
        lut = MultiLinearLUT(
            input_dims=2,
            output_channels=2,
            bins=3,
            identity_channels=(0, 1),
            chunk_pixels=7,
        )
        inputs = torch.rand(2, 2, 4, 5)
        outputs = lut(inputs)
        self.assertTrue(torch.allclose(outputs, inputs, atol=1e-6))


class PanLUTHSITest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = PanLUTHSI(bins=3, chunk_pixels=64)

    def test_shape_and_bias_configuration(self) -> None:
        mosaic = torch.rand(1, 1, 8, 8)
        pan = torch.rand(1, 1, 16, 16)
        output = self.model(mosaic, pan)

        self.assertEqual(tuple(output.shape), (1, 16, 16, 16))
        self.assertIsNone(self.model.encoder.bias)
        self.assertIsNone(self.model.decoder.bias)

    def test_backward_reaches_projection_and_luts(self) -> None:
        mosaic = torch.rand(1, 1, 8, 8)
        pan = torch.rand(1, 1, 16, 16)
        loss = self.model(mosaic, pan).square().mean()
        loss.backward()

        self.assertIsNotNone(self.model.encoder.weight.grad)
        self.assertIsNotNone(self.model.decoder.weight.grad)
        self.assertIsNotNone(self.model.pglut.table.grad)
        self.assertIsNotNone(self.model.sdlut.luts[0].table.grad)
        self.assertIsNotNone(self.model.aolut.table.grad)

    def test_shared_spatial_lut(self) -> None:
        self.assertEqual(len(self.model.sdlut.luts), 1)

    def test_pan_spectral_order_is_supported(self) -> None:
        model = PanLUTHSI(bins=3, chunk_pixels=64, input_order="pan_spectral")
        output = model(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
