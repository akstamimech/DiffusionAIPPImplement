"""
The numpy-only aipp core must reproduce the original sklearn/scipy implementation exactly (up to float rounding).

Compares, against functions.py and frozen copies of the loop helpers of the retired CMAES_classic_singlemap.py
(the code the training data was made with):
  - prior covariance (sklearn Matern 3/2)
  - footprint + resolution blocks + noise variance at many poses (every altitude band, borders, corners)
  - the simulated reading, including the exact random draws
  - a chain of Kalman updates (mean and covariance)
  - point-mass dynamics and the goal-reached test (dynamics_3d, waypoint_3d)
  - the path follower against a copy of the loop body it was lifted out of

Run:  python tests/test_core_equivalence.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from functions import (  # noqa: E402  (original implementation)
    build_correlated_noise_covariance,
    build_sensor_matrix,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)
from functions import fov_lateral_radius  # noqa: E402

from aipp import Command, GPBelief, GridSpec, KernelSpec, PathFollower, SensorSpec, direction_to, observation  # noqa: E402
from sim import PointMassVehicle, SimSensor  # noqa: E402

_STEP = 2.0


# --- frozen copies of the original simulator helpers (verbatim from CMAES_classic_singlemap.py, which every planner
# --- script used to import them from). They are the oracle the new code is held to; do not "improve" them.
def waypoint_3d(cx, cy, cz, goal_x, goal_y, goal_z, step):
    displacement = np.array([goal_x - cx, goal_y - cy, goal_z - cz])
    distance = np.linalg.norm(displacement)
    if distance <= step:
        return 0.0, 0.0, 0.0, True
    direction = displacement / distance
    return direction[0], direction[1], direction[2], False


def dynamics_3d(cx, cy, cz, grad_x, grad_y, grad_z, samplestep, xmin, xmax, ymin, ymax, zmin, zmax, buffer, step=_STEP):
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cz = np.clip(cz + grad_z * samplestep, zmin, zmax)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    cz = step * np.round(cz / step)
    return cx, cy, cz


def compute_fov(cz, xs, ys, angle_of_view=60, step=_STEP, cx=None, cy=None):
    radius = fov_lateral_radius(cz, angle_of_view)
    visible_xs = xs[(xs >= cx - radius) & (xs <= cx + radius)]
    visible_ys = ys[(ys >= cy - radius) & (ys <= cy + radius)]
    return [(float(x), float(y)) for x in visible_xs for y in visible_ys]

GRID, KERNEL, SENSOR = GridSpec(), KernelSpec(), SensorSpec()

# every altitude band, plus footprints that hang over the field edge
POSES = [
    (4.0, 4.0, 10.0), (50.0, 50.0, 10.0), (96.0, 96.0, 10.0),
    (20.0, 80.0, 18.0), (20.0, 80.0, 20.0), (20.0, 80.0, 21.0),
    (50.0, 50.0, 25.0), (50.0, 50.0, 30.0), (70.0, 30.0, 31.0),
    (4.0, 96.0, 36.0), (50.0, 2.0, 40.0), (99.0, 51.0, 40.0), (51.0, 49.0, 27.5),
]


def original_measurement(cx, cy, cz, truth, xs, ys, rng):
    fov = compute_fov(cz=cz, xs=xs, ys=ys, angle_of_view=SENSOR.angle_of_view_deg, step=GRID.step, cx=cx, cy=cy)
    sensor, block_ids = build_sensor_matrix(fov, cz, xs, ys, return_block_ids=True)
    var = noise_model(cz)
    z = sensor @ truth
    z += sample_correlated_sensor_noise(block_ids, var, rng)
    return sensor, block_ids, var, z


class CoreEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.X_test, _, cls.cov, cls.xs, cls.ys, *_ = initialize_gp()

    def test_grid(self):
        np.testing.assert_array_equal(GRID.xs, self.xs)
        np.testing.assert_array_equal(GRID.ys, self.ys)
        np.testing.assert_array_equal(GRID.cell_coordinates(), self.X_test)

    def test_prior_covariance(self):
        belief = GPBelief(GRID, KERNEL, prior_mean=0.2)
        np.testing.assert_allclose(belief.P, self.cov, rtol=1e-12, atol=1e-15)

    def test_observation_geometry(self):
        for cx, cy, cz in POSES:
            with self.subTest(pose=(cx, cy, cz)):
                fov = compute_fov(cz=cz, xs=self.xs, ys=self.ys, angle_of_view=60.0, step=2.0, cx=cx, cy=cy)
                sensor, block_ids = build_sensor_matrix(fov, cz, self.xs, self.ys, return_block_ids=True)
                obs = observation(cx, cy, cz, GRID, SENSOR)
                # original: one row per visible cell, rows of one block identical; keep first row of each block
                _, first = np.unique(block_ids, return_index=True)
                expected = sensor.toarray()[np.sort(first)]
                np.testing.assert_allclose(obs.H, expected, rtol=0, atol=1e-15)
                np.testing.assert_array_equal(obs.block_of_cell, block_ids)
                self.assertEqual(obs.noise_var, noise_model(cz))

    def test_simulated_reading_and_rng_stream(self):
        truth = np.random.default_rng(0).uniform(0.0, 0.8, GRID.n_cells)
        for seed in (0, 5):
            rng_a, rng_b = np.random.default_rng(seed), np.random.default_rng(seed)
            sim = SimSensor(truth, GRID, SENSOR, rng_b)
            for cx, cy, cz in POSES:
                sensor, block_ids, var, z = original_measurement(cx, cy, cz, truth, self.xs, self.ys, rng_a)
                _, first = np.unique(block_ids, return_index=True)
                H, z_new, var_new = sim.measure((cx, cy, cz))
                np.testing.assert_allclose(z_new, z[np.sort(first)], rtol=0, atol=1e-14)
                self.assertEqual(var_new, var)
            # the two generators consumed the same number of draws
            self.assertEqual(rng_a.normal(), rng_b.normal())

    def test_kalman_chain(self):
        truth = np.random.default_rng(1).uniform(0.0, 0.8, GRID.n_cells)
        rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
        mu_o, P_o = np.full(GRID.n_cells, 0.2), self.cov.copy()
        belief = GPBelief(GRID, KERNEL, prior_mean=0.2)
        sim = SimSensor(truth, GRID, SENSOR, rng_b)
        for cx, cy, cz in POSES:
            sensor, block_ids, var, z = original_measurement(cx, cy, cz, truth, self.xs, self.ys, rng_a)
            mu_o, P_o = kalman_update(mu_o, P_o, sensor, z, build_correlated_noise_covariance(block_ids, var), block_ids=block_ids)
            belief.update(*sim.measure((cx, cy, cz)))
        np.testing.assert_allclose(belief.mu, mu_o, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(belief.P, P_o, rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(belief.total_variance(), float(np.sum(np.diag(P_o))), places=8)

    def test_update_does_not_alias_inputs(self):
        belief = GPBelief(GRID, KERNEL, prior_mean=0.2)
        mu_before, P_before = belief.mu.copy(), belief.P.copy()
        old_mu, old_P = belief.mu, belief.P
        H, z, var = SimSensor(np.full(GRID.n_cells, 0.5), GRID, SENSOR, np.random.default_rng(0)).measure((50, 50, 10))
        belief.update(H, z, var)
        np.testing.assert_array_equal(old_mu, mu_before)     # earlier snapshots stay valid
        np.testing.assert_array_equal(old_P, P_before)
        self.assertFalse(np.array_equal(belief.mu, mu_before))

    def test_empty_footprint_is_a_noop(self):
        belief = GPBelief(GRID, KERNEL, prior_mean=0.2)
        before = belief.P.copy()
        belief.update(np.zeros((0, GRID.n_cells)), np.zeros(0), 0.01)
        np.testing.assert_array_equal(belief.P, before)


Z_RANGE = (10.0, 40.0)


class MotionEquivalence(unittest.TestCase):
    def test_direction_to_matches_waypoint_3d(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            pose, goal = rng.uniform(0, 100, 3), rng.uniform(0, 100, 3)
            if rng.random() < 0.2:
                goal = pose + rng.uniform(-2, 2, 3)               # near-reached cases
            (gx, gy, gz), reached = direction_to(pose, goal, 2.0)
            ox, oy, oz, o_reached = waypoint_3d(*pose, *goal, 2.0)
            self.assertEqual(reached, o_reached)
            self.assertEqual((gx, gy, gz), (ox, oy, oz))

    def test_point_mass_matches_dynamics_3d(self):
        rng = np.random.default_rng(1)
        for margin in (4.0, 1.0):
            for _ in range(300):
                pose = tuple(np.append(2.0 * rng.integers(0, 51, 2), 2.0 * rng.integers(5, 21)))   # on the grid
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                if rng.random() < 0.3:                          # push against the edges
                    pose = (rng.choice([0.0, 100.0]), rng.choice([0.0, 100.0]), rng.choice([10.0, 40.0]))
                vehicle = PointMassVehicle(pose, GRID, Z_RANGE)
                vehicle.fly(Command(tuple(d), margin))
                expected = dynamics_3d(*pose, *d, 2.0, 0.0, 100.0, 0.0, 100.0, *Z_RANGE, buffer=margin)
                self.assertEqual(tuple(vehicle.pose), tuple(expected))

    def test_hold_does_not_move(self):
        vehicle = PointMassVehicle((7.0, 9.0, 11.0), GRID, Z_RANGE)         # deliberately off the grid
        vehicle.fly(Command((0.0, 0.0, 0.0), 1.0, hold=True))
        self.assertEqual(vehicle.pose, (7.0, 9.0, 11.0))


def original_follow(pose, path, idx, step=2.0, zmin=10.0, zmax=40.0, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0):
    """The flight branch of the loop in the pre-refactor script, line for line. Returns (pose, path, idx)."""
    cx, cy, cz = pose
    if idx < len(path):
        goal_x, goal_y, goal_z = path[idx]
        grad_x, grad_y, grad_z, reached = waypoint_3d(cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step)
        if reached:
            idx += 1
            if idx < len(path):
                goal_x, goal_y, goal_z = path[idx]
                grad_x, grad_y, grad_z, _ = waypoint_3d(cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=step)
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
        if idx < len(path):
            x_next, y_next, z_next = path[idx]
            padding = step * 2
            clamped_x = min(max(x_next, xmin + padding), xmax - padding)
            clamped_y = min(max(y_next, ymin + padding), ymax - padding)
            clamped_z = min(max(z_next, zmin), zmax)
            if clamped_x != x_next or clamped_y != y_next or clamped_z != z_next:
                path[idx] = [clamped_x, clamped_y, clamped_z]
        cx, cy, cz = dynamics_3d(cx, cy, cz, grad_x, grad_y, grad_z, step, xmin, xmax, ymin, ymax, zmin, zmax, buffer=step / 2)
    return (cx, cy, cz), path, idx


class FollowerEquivalence(unittest.TestCase):
    def test_follower_matches_original_loop_body(self):
        rng = np.random.default_rng(2)
        clamps = reached_steps = 0
        for trial in range(30):
            n = int(rng.integers(3, 45))
            # random walk, some points outside the padded field so the clamp fires
            path = np.cumsum(rng.normal(0, 3.0, (n, 3)), axis=0) + np.array([50.0, 50.0, 20.0])
            path[:, 2] = np.clip(path[:, 2], 8.0, 42.0)
            start = (float(2 * rng.integers(3, 47)), float(2 * rng.integers(3, 47)), float(2 * rng.integers(5, 20)))

            follower, vehicle = PathFollower(GRID, Z_RANGE, execution_chunk=40), PointMassVehicle(start, GRID, Z_RANGE)
            follower.load(path)
            ref_pose, ref_path, ref_idx = start, path.tolist(), 0
            for k in range(n + 5):
                before = follower.index
                command = follower.command(vehicle.pose)
                clamps += command.clamped
                reached_steps += follower.index > before
                vehicle.fly(command)
                ref_pose, ref_path, ref_idx = original_follow(ref_pose, ref_path, ref_idx)
                with self.subTest(trial=trial, step=k):
                    self.assertEqual(tuple(vehicle.pose), tuple(ref_pose))
                    self.assertEqual(follower.index, ref_idx)
                    self.assertEqual(follower.path, ref_path)
        self.assertGreater(clamps, 0)             # the clamp branch and the advance branch were really exercised
        self.assertGreater(reached_steps, 0)

    def test_needs_new_plan_rules(self):
        follower = PathFollower(GRID, Z_RANGE, execution_chunk=3)
        self.assertTrue(follower.needs_new_plan())                     # nothing loaded
        follower.load(np.tile([[20.0, 20.0, 20.0]], (10, 1)))
        self.assertFalse(follower.needs_new_plan())
        follower.index = 3
        self.assertTrue(follower.needs_new_plan())                       # chunk used up
        follower.load(np.tile([[20.0, 20.0, 20.0]], (2, 1)))
        follower.index = 2
        self.assertTrue(follower.needs_new_plan())                       # plan used up


if __name__ == "__main__":
    unittest.main(verbosity=2)
