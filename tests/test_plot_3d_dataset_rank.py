import unittest

import matplotlib
import torch

matplotlib.use("Agg")

from plot_3d_dataset_rank import build_figure, select_samples, validate_dataset


def make_dataset():
    return {
        "trajectories": torch.zeros((4, 3, 41)),
        "control_waypoints": torch.zeros((4, 3, 8)),
        "current_position": torch.zeros((4, 3)),
        "current_mean": torch.zeros((4, 51, 51)),
        "map_id": torch.tensor([1, 1, 1, 2]),
        # rows 0 and 1 are the same (map=1, start=0, round=3) node: row 0 is
        # the real winner (parent_beam_index=0, highest variance_correction)
        # but has the *lowest* RMSE_correction, so a test that still sorted by
        # RMSE_correction would rank them the wrong way round.
        "start_index": torch.tensor([0, 0, 1, 0]),
        "timestep": torch.tensor([3, 3, 3, 3]),
        "parent_beam_index": torch.tensor([0, 1, 0, 0]),
        "variance_correction": torch.tensor([0.9, 0.5, 0.7, 0.8]),
        "RMSE_correction": torch.tensor([0.2, 0.9, 0.6, 0.8]),
    }


class DatasetSelectionTests(unittest.TestCase):
    def test_select_samples_filters_sorts_and_limits(self):
        selected = select_samples(make_dataset(), map_id=1, start_index=0, round_idx=3, top_n=1)

        self.assertEqual(selected["indices"].tolist(), [0])
        self.assertAlmostEqual(selected["variance_correction"].item(), 0.9, places=5)

    def test_select_samples_ranks_by_variance_not_rmse(self):
        # Both candidates at this node, ordered by variance_correction
        # descending - NOT by RMSE_correction (which would reverse them).
        selected = select_samples(make_dataset(), map_id=1, start_index=0, round_idx=3, top_n=2)

        self.assertEqual(selected["indices"].tolist(), [0, 1])
        self.assertEqual(selected["parent_beam_index"].tolist(), [0, 1])

    def test_select_samples_filters_by_start_index(self):
        # Same map and round as the node above, but a different chain
        # (start_index=1) - must not be mixed in.
        selected = select_samples(make_dataset(), map_id=1, start_index=1, round_idx=3, top_n=5)

        self.assertEqual(selected["indices"].tolist(), [2])

    def test_select_samples_rejects_missing_map_start_or_round(self):
        with self.assertRaisesRegex(ValueError, "No samples"):
            select_samples(make_dataset(), map_id=9, start_index=0, round_idx=9, top_n=5)

    def test_validate_dataset_rejects_missing_required_key(self):
        dataset = make_dataset()
        del dataset["trajectories"]

        with self.assertRaisesRegex(KeyError, "trajectories"):
            validate_dataset(dataset)

    def test_validate_dataset_rejects_wrong_coordinate_shape(self):
        dataset = make_dataset()
        dataset["control_waypoints"] = torch.zeros((4, 2, 8))

        with self.assertRaisesRegex(ValueError, "control_waypoints"):
            validate_dataset(dataset)

    def test_build_figure_creates_3d_and_top_down_axes(self):
        selected = select_samples(make_dataset(), map_id=1, start_index=0, round_idx=3, top_n=2)

        figure = build_figure(selected, map_id=1, start_index=0, round_idx=3)
        try:
            self.assertEqual(len(figure.axes), 3)
            self.assertEqual(figure.axes[0].name, "3d")
            self.assertEqual(figure.axes[1].name, "rectilinear")
            self.assertEqual(figure.axes[2].get_ylabel(), "Variance correction")

            legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
            self.assertTrue(any("[WINNER]" in label for label in legend_labels))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
