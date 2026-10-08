"""数据退化、尺寸倍率和指标的最小测试。"""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import scipy.io
import torch

from GetDataSet import SimulatedMosaicDataset, simulate_observations
from degradation import build_msfa_kernel, degrade_to_mosaic
from quality_index import calc_q2n, evaluate_full_reference


class DataPipelineTest(unittest.TestCase):
    def test_simulation_and_degradation_are_consistent(self) -> None:
        target = torch.arange(16 * 32 * 40, dtype=torch.float32).reshape(16, 32, 40)
        target /= target.max()
        mosaic, pan = simulate_observations(target)
        reconstructed = degrade_to_mosaic(target.unsqueeze(0), build_msfa_kernel()).squeeze(0)

        self.assertEqual(tuple(mosaic.shape), (1, 16, 20))
        self.assertEqual(tuple(pan.shape), (1, 32, 40))
        self.assertTrue(torch.allclose(mosaic, reconstructed, atol=1e-7))

    def test_identical_metrics(self) -> None:
        image = torch.rand(1, 16, 32, 32)
        metrics = evaluate_full_reference(image, image)
        self.assertGreater(metrics["psnr"], 70.0)
        self.assertLess(metrics["sam"], 0.1)
        self.assertLess(metrics["ergas"], 1e-5)

        q2n_value, q2n_map = calc_q2n(image[0], image[0], 16, 16)
        self.assertAlmostEqual(float(q2n_value), 1.0, places=4)
        self.assertEqual(tuple(q2n_map.shape), (2, 2))

    def test_simulated_dataset_uses_readme_sliding_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            train_root = Path(temporary_directory) / "train"
            train_root.mkdir()
            cube = np.random.default_rng(42).random((32, 40, 28), dtype=np.float32)
            scipy.io.savemat(train_root / "sample.mat", {"b": cube})

            dataset = SimulatedMosaicDataset(
                temporary_directory,
                "CAVE",
                "train",
                patch_size=16,
                stride=8,
            )

            self.assertEqual(len(dataset), 12)
            sample = dataset[0]
            self.assertEqual(tuple(sample["mosaic"].shape), (1, 8, 8))
            self.assertEqual(tuple(sample["pan"].shape), (1, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (16, 16, 16))


if __name__ == "__main__":
    unittest.main()
