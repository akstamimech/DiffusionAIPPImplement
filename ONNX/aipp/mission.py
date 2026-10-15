"""
The mission loop: one step = move, measure, update the belief; replan when the current plan is used up.

    warm-up steps    fly toward a fixed point so the belief has a few measurements before the first plan
    flight steps     follow the plan; when it runs out (or after execution_chunk points) ask the planner for a new one

It knows nothing about ground truth, metrics, files or plots. Whatever wants to watch a run (the simulator's RMSE
evaluation, a flight logger) registers an observer, which is called with a StepRecord after every belief update.
Every collaborator comes in through aipp.interfaces, so the same loop drives the simulator and the real drone.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .belief import GPBelief
from .follower import PathFollower, direction_to
from .interfaces import Command, FlightPlan, MeasurementSource, PathPlanner, Pose, Vehicle


@dataclass(frozen=True)
class MissionConfig:
    warmup_steps: int = 2
    warmup_goal: Pose = (80.0, 80.0, 10.0)
    warmup_tolerance: float = 2.0       # counts as arrived within this distance of the goal (one grid step)
    warmup_margin: float = 4.0          # keep-out margin from the field edge during warm-up (two grid steps)
    wallclock_seconds: float = 0.0      # <= 0: no budget, run all steps; otherwise stop starting new steps after this
    min_step_seconds: float = 0.0       # > 0: pad a step that finished sooner (emulates flying at a fixed speed)


@dataclass
class StepRecord:
    ts: int                                             # 0-based step index
    step_number: int                                    # ts + 1: how many steps have been flown
    phase: str                                          # "warmup" | "flight"
    pose: Pose                                          # where the vehicle is now (and where the measurement was taken)
    replanned: bool
    plan: Optional[FlightPlan]                          # the plan made in this step, None if none was made
    planned_path_xy: List[Tuple[float, float]]          # points of the plan still ahead
    control_waypoints_xy: List[Tuple[float, float]]
    clamped: bool                                       # the follower had to pull a waypoint back into the field
    total_variance: float                               # belief variance summed over the field after this update
    wall_time: float = math.nan                         # seconds since the flight started; set after the step is paced


@dataclass
class MissionResult:
    records: List[StepRecord] = field(default_factory=list)
    stopped_early: bool = False                         # the wall-clock budget ran out before all steps were flown
    flight_seconds: float = 0.0


class Mission:
    def __init__(self, belief: GPBelief, sensor: MeasurementSource, planner: PathPlanner, follower: PathFollower,
                 vehicle: Vehicle, config: MissionConfig = MissionConfig(),
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.belief, self.sensor, self.planner = belief, sensor, planner
        self.follower, self.vehicle, self.config = follower, vehicle, config
        self.clock, self.sleep = clock, sleep
        self._flight_start = None
        self._waypoints = None

    def step(self, ts: int, observers: Sequence[Callable[[StepRecord], None]] = ()) -> StepRecord:
        cfg = self.config
        step_start = self.clock()
        if self._flight_start is None:
            self._flight_start = step_start

        plan = None
        if ts < cfg.warmup_steps:
            phase = "warmup"
            direction, _ = direction_to(self.vehicle.pose, cfg.warmup_goal, cfg.warmup_tolerance)
            command = Command(direction, cfg.warmup_margin)
        else:
            phase = "flight"
            if self.follower.needs_new_plan:
                plan = self.planner.plan(self.belief.mean_map(), self.belief.var_map(), self.vehicle.pose)
                self.follower.load(plan.path)
                self._waypoints = plan.waypoints
            command = self.follower.command(self.vehicle.pose)
        self.vehicle.fly(command)
        pose = self.vehicle.pose

        flying_a_plan = phase == "flight" and not command.hold
        planned_xy = self.follower.remaining_xy() if flying_a_plan else []
        waypoints_xy = [(float(x), float(y)) for x, y, _ in self._waypoints] if flying_a_plan else []

        self.belief.update(*self.sensor.measure(pose))

        record = StepRecord(
            ts=ts, step_number=ts + 1, phase=phase, pose=pose, replanned=plan is not None, plan=plan,
            planned_path_xy=planned_xy, control_waypoints_xy=waypoints_xy, clamped=command.clamped,
            total_variance=self.belief.total_variance(),
        )
        for observer in observers:
            observer(record)

        if cfg.min_step_seconds > 0:
            elapsed = self.clock() - step_start
            if elapsed < cfg.min_step_seconds:
                self.sleep(cfg.min_step_seconds - elapsed)
        record.wall_time = self.clock() - self._flight_start
        return record

    def run(self, n_steps: int, observers: Sequence[Callable[[StepRecord], None]] = ()) -> MissionResult:
        result = MissionResult()
        self._flight_start = self.clock()
        budgeted = self.config.wallclock_seconds > 0
        for ts in range(n_steps):
            if budgeted and (self.clock() - self._flight_start) >= self.config.wallclock_seconds:
                result.stopped_early = True
                break
            result.records.append(self.step(ts, observers))
        result.flight_seconds = self.clock() - self._flight_start
        return result
