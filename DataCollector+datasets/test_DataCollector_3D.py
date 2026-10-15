import unittest
from unittest.mock import patch

import numpy as np

from DataCollector import DataCollector_3D as collector


class PlannerSeedTests(unittest.TestCase):
    def test_planner_seed_is_forwarded_to_3d_cmaes(self):
        mu = np.zeros(4)
        covariance = np.eye(4)
        xs = np.array([0.0, 2.0])
        ys = np.array([0.0, 2.0])
        initial_plan = [(2.0, 2.0, 10.0)]

        with (
            patch.object(collector, "grid_search_3d", return_value=initial_plan),
            patch.object(
                collector,
                "cma_es_refine_waypoints_3d",
                return_value=initial_plan,
            ) as refine,
        ):
            collector.real_receding_horizon_planner(
                cx=0.0,
                cy=0.0,
                cz=10.0,
                mu=mu,
                P=covariance,
                xs=xs,
                ys=ys,
                utility_threshold=0.3,
                beta=1.0,
                planning_horizon=1,
                planner_seed=9876,
            )

        self.assertEqual(refine.call_args.kwargs["seed"], 9876)

    def test_candidate_seeds_differ_between_repeated_proposals(self):
        base_seed = collector.make_candidate_seed(
            selected_map=3,
            rank=4,
            beam_id=2,
        )
        seeds = [base_seed + proposal_index for proposal_index in range(5)]

        self.assertEqual(len(set(seeds)), 5)


if __name__ == "__main__":
    unittest.main()
