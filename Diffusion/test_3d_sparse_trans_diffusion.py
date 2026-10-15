import importlib.util
import unittest
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")


MODULE_PATH = Path(__file__).with_name("threeDSparseTransDiffusion.py")
SPEC = importlib.util.spec_from_file_location("sparse_trans_diffusion_3d", MODULE_PATH)
diffusion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diffusion)


class ThreeDimensionalDiffusionTests(unittest.TestCase):
    def test_xyz_normalization_round_trip(self):
        xyz = torch.tensor(
            [
                [0.0, 100.0, diffusion.Z_MIN],
                [100.0, 0.0, diffusion.Z_MAX],
            ]
        )

        normalized = diffusion.normalize_xyz(xyz)
        restored = diffusion.denormalize_xyz(normalized)

        self.assertTrue(torch.allclose(restored, xyz))
        self.assertTrue(torch.allclose(normalized[0], torch.tensor([-1.0, 1.0, -1.0])))
        self.assertTrue(torch.allclose(normalized[1], torch.tensor([1.0, -1.0, 1.0])))

    def test_displacement_normalization_does_not_apply_altitude_offset(self):
        displacement = torch.tensor([[50.0, -50.0, 15.0]])

        normalized = diffusion.normalize_xyz_displacement(displacement)

        self.assertTrue(torch.allclose(normalized, torch.tensor([[1.0, -1.0, 1.0]])))

    def test_cubic_spline_accepts_three_dimensional_waypoints(self):
        controls = torch.tensor(
            [
                [
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                ]
            ]
        )
        current_position = torch.tensor([[0.0, 0.0, 10.0]])

        dense = diffusion.pytorch_cubic_spline(
            controls,
            current_position,
            samples_per_segment=2,
        )

        self.assertEqual(tuple(dense.shape), (1, 3, 7))
        self.assertTrue(torch.allclose(dense[:, :, -1], controls[:, -1, :]))

    def test_noise_predictor_preserves_3d_waypoint_shape(self):
        model = diffusion.NoisePredictor(token_dim=32, base_channels=8, num_blocks=1)
        noisy_waypoints = torch.randn(2, 3, 8)
        diffusion_steps = torch.tensor([1, 2])
        maps = torch.randn(2, 3, 51, 51)
        positions = torch.randn(2, 3)
        headings = torch.randn(2, 3)

        prediction = model(
            noisy_waypoints,
            diffusion_steps,
            maps,
            positions,
            headings,
        )

        self.assertEqual(tuple(prediction.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
