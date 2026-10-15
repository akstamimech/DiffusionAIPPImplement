import importlib.util
import unittest
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")


MODULE_PATH = Path(__file__).with_name("sample_3d_sparse_trans_diffusion.py")
SPEC = importlib.util.spec_from_file_location("sample_sparse_trans_diffusion_3d", MODULE_PATH)
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)


class ThreeDimensionalSamplerTests(unittest.TestCase):
    def test_dynamic_import_loads_three_dimensional_training_module(self):
        self.assertEqual(sampler.diffusion.NUM_COORDS, 3)
        self.assertEqual(sampler.diffusion.TARGET_SHAPE, (3, 8))

    def test_sample_sparse_returns_three_dimensional_waypoints(self):
        model = sampler.diffusion.NoisePredictor(
            token_dim=32,
            base_channels=8,
            num_blocks=1,
        )

        sample = sampler.sample_sparse(
            model,
            condition_index=0,
            seed=123,
            num_steps=2,
            clip_x0=True,
        )

        self.assertEqual(tuple(sample.shape), (1, 3, 8))

    def test_build_sample_figure_has_3d_xy_and_altitude_axes(self):
        sample = sampler.diffusion.trajectories[0:1].clone()

        figure = sampler.build_sample_figure(
            sample,
            truth_index=0,
            condition_index=0,
        )
        try:
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual(figure.axes[0].name, "3d")
            self.assertEqual(figure.axes[1].name, "rectilinear")
            self.assertEqual(figure.axes[2].get_ylabel(), "Altitude")
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
