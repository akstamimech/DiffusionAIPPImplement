"""
The mission loop and the planner wrapper, tested without the simulator's ground truth.

Mission tests use stub collaborators and a fake clock, so they cover what the golden regression cannot (it runs with no
wall-clock budget and no pacing): warm-up, when replans happen, the wall-clock budget, step pacing, observers.
The planner test checks the wrapper against the original composition of the reference network calls.

Run:  python ONNX/tests/test_mission.py
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONNX_DIR = HERE.parent
if str(ONNX_DIR) not in sys.path:
    sys.path.insert(0, str(ONNX_DIR))

from aipp import FlightPlan, GridSpec, Mission, MissionConfig, PathFollower  # noqa: E402
from aipp.sim import PointMassVehicle  # noqa: E402

GRID = GridSpec()
Z_RANGE = (10.0, 40.0)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class StubBelief:
    """Just the surface the mission uses; avoids building a 2601 x 2601 covariance for every test."""
    def __init__(self):
        self.updates = 0

    def mean_map(self):
        return np.zeros((GRID.ny, GRID.nx))

    def var_map(self):
        return np.ones((GRID.ny, GRID.nx))

    def update(self, H, z, noise_var):
        self.updates += 1

    def total_variance(self):
        return 100.0 - self.updates


class StubSensor:
    def __init__(self, clock=None, seconds_per_measurement=0.0):
        self.clock, self.seconds = clock, seconds_per_measurement
        self.poses = []

    def measure(self, pose):
        self.poses.append(tuple(pose))
        if self.clock is not None:
            self.clock.now += self.seconds
        return np.zeros((0, GRID.n_cells)), np.zeros(0), 0.01


class LinePlanner:
    """Plans a straight line of `points` points, 2 m apart in x, starting one step ahead of the current pose."""
    def __init__(self, points=20):
        self.points, self.calls = points, []

    def plan(self, mean_map, var_map, pose):
        self.calls.append(tuple(pose))
        path = np.array([[pose[0] + 2.0 * (i + 1), pose[1], pose[2]] for i in range(self.points)])
        return FlightPlan(waypoints=path[::5][:8].copy(), path=path, seconds=0.01)


def make_mission(planner, chunk=5, clock=None, sensor=None, **config):
    clock = clock or FakeClock()
    vehicle = PointMassVehicle((4.0, 4.0, 10.0), GRID, Z_RANGE)
    mission = Mission(StubBelief(), sensor or StubSensor(), planner, PathFollower(GRID, Z_RANGE, chunk), vehicle,
                      MissionConfig(**config), clock=clock, sleep=clock.sleep)
    return mission, vehicle, clock


class MissionLoop(unittest.TestCase):
    def test_warmup_flies_toward_goal_then_first_plan_at_step_two(self):
        planner = LinePlanner()
        mission, vehicle, _ = make_mission(planner)
        result = mission.run(5)
        phases = [r.phase for r in result.records]
        self.assertEqual(phases, ["warmup", "warmup", "flight", "flight", "flight"])
        self.assertEqual([r.replanned for r in result.records], [False, False, True, False, False])
        self.assertEqual(len(planner.calls), 1)
        # the two warm-up steps move toward (80, 80, 10): up and to the right, altitude unchanged
        first, second = result.records[0].pose, result.records[1].pose
        self.assertTrue(4.0 < first[0] < second[0] and 4.0 < first[1] < second[1])
        self.assertEqual((first[2], second[2]), (10.0, 10.0))
        # the plan was requested from where the warm-up ended
        self.assertEqual(planner.calls[0], tuple(second))

    def test_replans_every_execution_chunk(self):
        planner = LinePlanner(points=20)
        mission, _, _ = make_mission(planner, chunk=5)
        result = mission.run(15)
        self.assertEqual([r.ts for r in result.records if r.replanned], [2, 7, 12])

    def test_replans_when_plan_runs_out_before_chunk(self):
        planner = LinePlanner(points=5)
        mission, _, _ = make_mission(planner, chunk=40)
        result = mission.run(14)
        self.assertEqual([r.ts for r in result.records if r.replanned], [2, 7, 12])

    def test_every_step_measures_at_the_new_pose_and_updates_the_belief(self):
        sensor = StubSensor()
        mission, _, _ = make_mission(LinePlanner(), sensor=sensor)
        result = mission.run(8)
        self.assertEqual(sensor.poses, [tuple(r.pose) for r in result.records])
        self.assertEqual(mission.belief.updates, 8)
        self.assertEqual([r.total_variance for r in result.records], [99.0 - i for i in range(8)])

    def test_records_describe_the_plan_still_ahead(self):
        mission, _, _ = make_mission(LinePlanner(points=20), chunk=40)
        result = mission.run(6)
        warmup, flight = result.records[:2], result.records[2:]
        for r in warmup:
            self.assertEqual((r.planned_path_xy, r.control_waypoints_xy), ([], []))
        self.assertTrue(result.records[2].replanned and result.records[2].plan is not None)
        counts = [len(r.planned_path_xy) for r in flight]
        self.assertEqual(counts, sorted(counts, reverse=True))        # shrinks as points are consumed
        self.assertTrue(all(len(r.control_waypoints_xy) > 0 for r in flight))

    def test_empty_plan_holds_position(self):
        class EmptyPlanner:
            def plan(self, mean_map, var_map, pose):
                return FlightPlan(waypoints=np.zeros((0, 3)), path=np.zeros((0, 3)), seconds=0.0)
        mission, _, _ = make_mission(EmptyPlanner())
        result = mission.run(6)
        flight_poses = {tuple(r.pose) for r in result.records[2:]}
        self.assertEqual(len(flight_poses), 1)                         # not moving
        self.assertTrue(all(r.planned_path_xy == [] for r in result.records))

    def test_observers_see_each_step_before_pacing_and_wall_time_is_set_after(self):
        seen = []
        mission, _, _ = make_mission(LinePlanner(), min_step_seconds=0.5)
        result = mission.run(4, observers=[lambda r: seen.append((r.ts, math.isnan(r.wall_time)))])
        self.assertEqual(seen, [(0, True), (1, True), (2, True), (3, True)])
        self.assertTrue(all(not math.isnan(r.wall_time) for r in result.records))


class Pacing(unittest.TestCase):
    def test_fast_steps_are_padded_to_the_floor(self):
        mission, _, clock = make_mission(LinePlanner(), min_step_seconds=1 / 3)
        result = mission.run(6)
        self.assertEqual(len(clock.sleeps), 6)
        np.testing.assert_allclose(clock.sleeps, 1 / 3)
        np.testing.assert_allclose([r.wall_time for r in result.records], [(i + 1) / 3 for i in range(6)])

    def test_slow_steps_are_left_alone(self):
        clock = FakeClock()
        mission, _, _ = make_mission(LinePlanner(), clock=clock, sensor=StubSensor(clock, 1.0), min_step_seconds=1 / 3)
        mission.run(4)
        self.assertEqual(clock.sleeps, [])

    def test_no_pacing_by_default(self):
        mission, _, clock = make_mission(LinePlanner())
        mission.run(4)
        self.assertEqual(clock.sleeps, [])


class WallClockBudget(unittest.TestCase):
    def test_stops_starting_steps_once_the_budget_is_spent(self):
        clock = FakeClock()
        mission, _, _ = make_mission(LinePlanner(), clock=clock, sensor=StubSensor(clock, 1.0), wallclock_seconds=3.5)
        result = mission.run(10)
        self.assertEqual(len(result.records), 4)                       # steps start at t = 0, 1, 2, 3; t = 4 is over budget
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.flight_seconds, 4.0)

    def test_zero_budget_means_unconstrained(self):
        clock = FakeClock()
        mission, _, _ = make_mission(LinePlanner(), clock=clock, sensor=StubSensor(clock, 1000.0), wallclock_seconds=0.0)
        result = mission.run(5)
        self.assertEqual(len(result.records), 5)
        self.assertFalse(result.stopped_early)


class PlannerWrapper(unittest.TestCase):
    MODEL = ONNX_DIR / "new_1200_fp32.onnx"
    DEPLOY = ONNX_DIR / "new_1200_deploy.json"

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")
        if not cls.MODEL.exists():
            raise unittest.SkipTest(f"{cls.MODEL.name} not found")

    def maps(self):
        rng = np.random.default_rng(5)
        mean = np.clip(0.2 + 0.1 * rng.standard_normal((51, 51)), 0.0, 1.0)
        var = np.clip(0.004 * rng.random((51, 51)), 1e-6, None)
        return mean, var

    def test_matches_the_original_composition(self):
        from aipp.planner import DiffusionPlanner
        from onnx_sample_reference import Planner

        mean, var = self.maps()
        pose = (30.0, 44.0, 16.0)
        plan = DiffusionPlanner(self.MODEL, self.DEPLOY, GRID, threads=1, seed=3).plan(mean, var, pose)

        # what the pre-refactor script did, step by step
        net = Planner(self.MODEL, self.DEPLOY, threads=1, seed=3)
        position = np.asarray(pose, dtype=np.float64).reshape(3)
        maps, pos_n, tv = net.build_inputs(mean, var, position)
        waypoints = net.denormalize(net.sample_normalized(maps, pos_n, tv, eta=0.0, clip_x0=True, num_steps=None))
        dense = net.cubic_spline(position, waypoints).T.copy()
        controls = waypoints.T.copy()
        pad, (zmin, zmax) = GRID.step * 2, (10.0, 40.0)
        for arr in (dense, controls):
            arr[0] = np.clip(arr[0], GRID.xmin + pad, GRID.xmax - pad)
            arr[1] = np.clip(arr[1], GRID.ymin + pad, GRID.ymax - pad)
            arr[2] = np.clip(arr[2], zmin, zmax)

        np.testing.assert_array_equal(plan.path, dense.T)
        np.testing.assert_array_equal(plan.waypoints, controls.T)
        self.assertEqual(plan.path.shape, (41, 3))
        self.assertEqual(plan.waypoints.shape, (8, 3))
        self.assertGreater(plan.seconds, 0.0)

    def test_plan_stays_inside_the_padded_field_and_altitude_band(self):
        from aipp.planner import DiffusionPlanner

        mean, var = self.maps()
        planner = DiffusionPlanner(self.MODEL, self.DEPLOY, GRID, threads=1, seed=0)
        self.assertEqual(planner.z_range, (10.0, 40.0))
        for pose in [(4.0, 4.0, 10.0), (96.0, 96.0, 40.0), (50.0, 2.0, 25.0)]:
            plan = planner.plan(mean, var, pose)
            for pts in (plan.path, plan.waypoints):
                self.assertTrue(np.all(pts[:, :2] >= 4.0) and np.all(pts[:, :2] <= 96.0))
                self.assertTrue(np.all(pts[:, 2] >= 10.0) and np.all(pts[:, 2] <= 40.0))

    def test_same_seed_same_plan_different_seed_different_plan(self):
        from aipp.planner import DiffusionPlanner

        mean, var = self.maps()
        make = lambda seed: DiffusionPlanner(self.MODEL, self.DEPLOY, GRID, threads=1, seed=seed).plan(mean, var, (30.0, 44.0, 16.0))
        np.testing.assert_array_equal(make(1).path, make(1).path)
        self.assertFalse(np.array_equal(make(1).path, make(2).path))      # the initial noise comes from the seeded generator


if __name__ == "__main__":
    unittest.main(verbosity=2)
