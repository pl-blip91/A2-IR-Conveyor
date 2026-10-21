"""Loading station: a Yaskawa MH5 on a TurtleBot3 loads toy cars onto the conveyor.

Techniques used, and why:
    * Joint-space quintic trajectories (jtraj) for the large, unconstrained moves.
    * Inverse kinematics (ikine_LM) for the pre-grasp and pre-place poses.
    * Resolved-rate motion control (RMRC) for the straight vertical approach/retreat,
      with singularity damping and joint-limit protection.
    * Image-based visual servoing (IBVS) with a simulated eye-in-hand camera to find a
      car that was dropped slightly off its slot on the supply table.
    * Collision prediction: the planned path is checked ahead of the arm; if a
      controlled obstacle appears in it the station re-plans around it, and stops and
      waits if no way round exists.
    * Safety states: emergency stop, light-curtain / unsafe-zone protective stop, with a
      deliberate reset and a separate resume (never a restart on release).
    * A teach/jog interface (joint jog, Cartesian jog, go to a joint state) for the GUI.

Everything advances one dt per tick() using generators, so nothing blocks.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import Callable, Iterator, Optional

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3
from spatialmath import base as smb

from robots.parallel_gripper import OPEN_WIDTH, TCP_OFFSET, ParallelGripper
from robots.turtlebot_base import TOP_HEIGHT, TurtleBotBase
from robots.yaskawa_mh5 import YaskawaMH5
from safety.collision import ArmEnvelope, Box, CollisionChecker, Obstacle, plan_around
from safety.equipment import SafetyEquipment
from safety.zones import SafetyInputs
from workcell.vision import (
    Camera,
    CarServo,
    camera_twist_to_tool_twist,
    limit_twist,
)

from . import config as cfg
from . import loading_config as lc


class StationFault(RuntimeError):
    """Raised inside the sequence when it must stop (unreachable pose, singularity, etc)."""


class RunState(Enum):
    IDLE = "IDLE"  # not started
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"  # holding still: an obstacle is in the planned path
    ESTOPPED = "ESTOPPED"  # emergency stop latched; needs reset() then resume()
    PROTECTIVE = "PROTECTIVE"  # light curtain / unsafe zone stop; needs reset() then resume()
    STOPPED = "STOPPED"  # reset done; waiting for resume()
    DONE = "DONE"
    FAULT = "FAULT"  # the run was aborted; see ``fault``; rearm() to recover


# States the station can be in when a stop arrives; resume() returns to the saved one.
_STOPPABLE = {RunState.IDLE, RunState.RUNNING, RunState.BLOCKED, RunState.DONE, RunState.FAULT}
_BUSY = {RunState.RUNNING, RunState.BLOCKED, RunState.ESTOPPED, RunState.PROTECTIVE, RunState.STOPPED}
# States in which the operator may jog the arm or send it to a joint state.
_MANUAL_OK = {RunState.IDLE, RunState.DONE, RunState.FAULT}


# --------------------------------------------------------------------------------------
# World additions
# --------------------------------------------------------------------------------------
def car_pose(x: float, height: float) -> SE3:
    """Pose of a car centre on a surface at ``height`` (long axis along world x)."""
    return SE3(x, 0.0, height + cfg.CAR_SIZE[2] / 2)


def build_loading_area(env, n_cars: int = lc.MAX_CARS) -> dict:
    """Add the supply table, its cars and the floor markings to ``env``.

    Returns the created shapes: ``{"table": [...], "cars": [...], "markings": [...]}``.
    """
    if not 1 <= n_cars <= lc.MAX_CARS:
        raise ValueError(f"n_cars must be 1..{lc.MAX_CARS}")

    def box(size, centre, colour):
        shape = sg.Cuboid(scale=list(size), pose=SE3(*centre), color=list(colour))
        env.add(shape)
        return shape

    top_z = lc.TABLE_HEIGHT - lc.TABLE_TOP_THICKNESS / 2
    table = [box((lc.TABLE_SIZE[0], lc.TABLE_SIZE[1], lc.TABLE_TOP_THICKNESS),
                 (lc.TABLE_CENTRE_X, 0.0, top_z), lc.TABLE_COLOUR)]
    leg_h = lc.TABLE_HEIGHT - lc.TABLE_TOP_THICKNESS
    inset = lc.LEG_SIZE / 2 + 0.02
    for dx in (-1, 1):
        for dy in (-1, 1):
            table.append(box(
                (lc.LEG_SIZE, lc.LEG_SIZE, leg_h),
                (lc.TABLE_CENTRE_X + dx * (lc.TABLE_SIZE[0] / 2 - inset),
                 dy * (lc.TABLE_SIZE[1] / 2 - inset), leg_h / 2),
                lc.TABLE_COLOUR))

    cars = [box(cfg.CAR_SIZE, car_pose(lc.TABLE_SLOT_XS[k], lc.TABLE_HEIGHT).t, cfg.UNPAINTED)
            for k in range(n_cars)]

    # Floor marking where the base stops to pick (the belt-side dock is drawn by build_scene).
    pick_dock = box(cfg.DOCK_SIZE, (lc.TABLE_CENTRE_X, lc.WORK_Y, cfg.DOCK_SIZE[2] / 2), cfg.DOCK_COLOUR)
    return {"table": table, "cars": cars, "markings": [pick_dock]}


def _static_keepouts() -> list[Box]:
    """Workcell structures the arm links must not pass through (belt and table top)."""
    half_belt = cfg.BELT_WIDTH / 2
    table_half = lc.TABLE_SIZE[0] / 2
    return [
        Box((0.0, -half_belt, 0.0), (cfg.BELT_LENGTH, half_belt, cfg.BELT_HEIGHT)),
        Box((lc.TABLE_CENTRE_X - table_half, -half_belt, lc.TABLE_HEIGHT - lc.TABLE_TOP_THICKNESS),
            (lc.TABLE_CENTRE_X + table_half, half_belt, lc.TABLE_HEIGHT)),
    ]


# --------------------------------------------------------------------------------------
# The station
# --------------------------------------------------------------------------------------
class LoadingStation:
    """MH5-on-TurtleBot loading task with safety, vision and a manual (jog) mode."""

    def __init__(
        self,
        env,
        n_cars: int = lc.MAX_CARS,
        dt: float = 0.05,
        speed_scale: float = 1.0,
        on_car_placed: Optional[Callable[[int, sg.Cuboid], None]] = None,
        build_world: bool = True,
        jitter: bool = True,
    ):
        self.env = env
        self.n_cars = n_cars
        self.dt = dt
        self.speed_scale = speed_scale
        self.on_car_placed = on_car_placed
        self.jitter = jitter

        self.area = build_loading_area(env, n_cars) if build_world else {"cars": []}
        self.cars = self.area["cars"]

        park = lc.STATION.robot_base
        self._park = (park.t[0], park.t[1], lc.WORK_YAW)
        self.base = TurtleBotBase(*self._park)
        self.arm = YaskawaMH5(tool_offset=TCP_OFFSET)
        self.gripper = ParallelGripper()
        self.plate = sg.Cuboid(scale=list(lc.MOUNT_PLATE_SIZE), color=list(lc.MOUNT_PLATE_COLOUR))
        self._kin = self.arm.kinematics_only()

        self.base.add_to_env(env)
        self.arm.add_to_env(env)
        self.gripper.add_to_env(env)
        env.add(self.plate)

        # Safety: simulated sensors, equipment models and the collision checker.
        self.inputs = SafetyInputs()
        self.inputs.person.add_to_env(env)
        self.equipment = SafetyEquipment(env) if build_world else None
        self.obstacle = Obstacle(radius=lc.OBSTACLE_RADIUS)
        self._obstacle_shape = sg.Sphere(radius=lc.OBSTACLE_RADIUS, color=[1.0, 0.15, 0.15, 0.75])
        env.add(self._obstacle_shape)
        self._obstacle_last: Optional[tuple] = None
        self.checker = CollisionChecker(
            ArmEnvelope(self.arm.links, np.asarray(self.arm.tool.A)),
            lambda: np.asarray(self.arm.base.A),
            [self.obstacle], _static_keepouts(), lc.COLLISION_MARGIN,
        )

        # Simulated eye-in-hand camera and the visual-servo controller.
        self._cam_mount = SE3(0, 0, -TCP_OFFSET)
        T_car_tcp_goal = SE3(0, 0, lc.GRASP_LIFT + lc.PRE_HEIGHT) * SE3.Rx(math.pi)
        self._servo = CarServo(Camera(), T_car_tcp_goal * self._cam_mount)

        self.arm.q = lc.Q_TRAVEL
        self.gripper.opening = OPEN_WIDTH
        self._held: Optional[tuple[sg.Cuboid, SE3]] = None
        self._last_base: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(lc.JITTER_SEED)
        self._place_cars_on_table()
        self._sync()

        self.run_state = RunState.IDLE
        self.phase = "idle"
        self.fault: Optional[str] = None
        self.placed = 0
        self.manipulability = float("nan")
        self.sim_time = 0.0
        self.vision_error_px = float("nan")
        self.replans = 0
        self.block_reason: Optional[str] = None
        self.message = ""
        self.events: deque = deque(maxlen=60)
        self._estop_active = False
        self._resume_state = RunState.IDLE
        self._phase_before_stop = "idle"
        self._gen: Optional[Iterator[None]] = None
        self._manual: Optional[Iterator[None]] = None
        self._jog: Optional[tuple] = None
        self._motion: Optional[dict] = None  # what is moving now, for drop_obstacle_in_path()

    # ---------------------------------------------------------------- poses
    def pick_pose(self, k: int) -> SE3:
        """Nominal pose of car ``k`` on the supply table."""
        return car_pose(lc.TABLE_SLOT_XS[k], lc.TABLE_HEIGHT)

    def place_pose(self, k: int) -> SE3:
        """Pose of car ``k`` once it is on the belt."""
        return car_pose(lc.BELT_SLOT_XS[k], cfg.BELT_HEIGHT)

    @staticmethod
    def grasp_tcp(car: SE3) -> SE3:
        """Grip-centre pose that grasps ``car``: gripper pointing down, fingers across the car."""
        return SE3(car.t + np.array([0.0, 0.0, lc.GRASP_LIFT])) * SE3.Rx(math.pi)

    def _place_cars_on_table(self) -> None:
        """Put every car on the supply table, offset a little if ``jitter`` is on."""
        for k, car in enumerate(self.cars):
            pose = self.pick_pose(k)
            if self.jitter:
                dx, dy = self._rng.uniform(-lc.PICK_JITTER_XY, lc.PICK_JITTER_XY, size=2)
                yaw = self._rng.uniform(-lc.PICK_JITTER_YAW, lc.PICK_JITTER_YAW)
                pose = SE3(dx, dy, 0.0) * pose * SE3.Rz(yaw)
            car.T = pose.A

    # --------------------------------------------------------- run control
    @property
    def inputs_clear(self) -> bool:
        """True when no stop input (e-stop, light curtain, unsafe zone) is active."""
        return not self._estop_active and not self.inputs.tripped

    def start(self) -> None:
        """Begin the loading sequence (from IDLE)."""
        if self._estop_active:
            raise RuntimeError("cannot start while the emergency stop is pressed")
        if self.run_state is not RunState.IDLE:
            raise RuntimeError(f"cannot start from {self.run_state.value}")
        if self.inputs.tripped:
            raise RuntimeError("cannot start while a person is in the guarded zone")
        self._manual, self._jog = None, None
        self._gen = self.sequence()
        self.run_state = RunState.RUNNING

    def _enter_stop(self, state: RunState, text: str) -> None:
        """Latch a stop state, remembering where to go back to on resume."""
        if self.run_state in _STOPPABLE:
            self._resume_state = self.run_state
            self._phase_before_stop = self.phase
        self.run_state = state
        self.phase = text
        self._manual, self._jog = None, None  # a stop cancels any manual command
        self.base.cmd_v = self.base.cmd_w = 0.0

    def emergency_stop(self) -> None:
        """Stop everything now and latch ``ESTOPPED`` (simulated GUI e-stop)."""
        self._estop_active = True
        if self.run_state is not RunState.ESTOPPED:
            self._enter_stop(RunState.ESTOPPED, "EMERGENCY STOP")
            self._log("EMERGENCY STOP pressed")

    def release_estop(self) -> None:
        """The stop button was released. This alone never restarts motion."""
        if self._estop_active:
            self._log("e-stop released (reset required)")
        self._estop_active = False

    def reset(self) -> bool:
        """Acknowledge a stop. Only works once every stop input is clear."""
        if self.run_state in (RunState.ESTOPPED, RunState.PROTECTIVE) and self.inputs_clear:
            self.run_state = RunState.STOPPED
            self.phase = "stopped - press resume"
            self._log("stop acknowledged (reset); press resume")
            return True
        return False

    def resume(self) -> bool:
        """Continue after ``reset()``, from exactly where the station stopped."""
        if self.run_state is RunState.STOPPED and self.inputs_clear:
            self.run_state = self._resume_state
            self.phase = self._phase_before_stop
            self._log(f"resumed -> {self.run_state.value}")
            return True
        return False

    def rearm(self) -> bool:
        """Return to a known start state: cars back on the table, arm stowed, base docked.

        Only allowed when idle, done or faulted, and with no stop input active.
        """
        if self.run_state not in _MANUAL_OK or not self.inputs_clear:
            return False
        self._gen, self._manual, self._jog = None, None, None
        self._held = None
        self.run_state = RunState.IDLE
        self.phase = "idle"
        self.fault = None
        self.placed = 0
        self.block_reason = None
        self.base.set_pose(*self._park)
        self.arm.q = lc.Q_TRAVEL
        self.gripper.opening = OPEN_WIDTH
        self.obstacle.remove()
        self._place_cars_on_table()
        self._last_base = None
        self._sync()
        self._log("re-armed: cars on table, arm stowed")
        return True

    # ------------------------------------------------------------ sensors
    def trip_light_curtain(self, tripped: bool = True) -> None:
        """Force the light curtain on/off (an asynchronous simulated sensor signal)."""
        self.inputs.forced_curtain = bool(tripped)

    def person_walk_in(self) -> None:
        self.inputs.person.walk_in()

    def person_walk_out(self) -> None:
        self.inputs.person.walk_out()

    def _safety_update(self) -> None:
        was = self.inputs.tripped
        if self.inputs.update(self.dt) and not was:
            what = "light curtain broken" if self.inputs.curtain_broken else "person in unsafe zone"
            self._log(f"SAFETY: {what}")
            if self.run_state is not RunState.ESTOPPED and self.run_state is not RunState.PROTECTIVE:
                self._enter_stop(RunState.PROTECTIVE, f"SAFETY STOP ({what})")
        elif was and not self.inputs.tripped:
            self._log("safety input cleared (reset required)")

    # ------------------------------------------------------------ obstacle
    def place_obstacle(self, xyz) -> None:
        """Put the controlled obstacle at ``xyz`` (world frame)."""
        self.obstacle.place(xyz)
        self._log(f"obstacle placed at {np.round(self.obstacle.centre, 2).tolist()}")

    def remove_obstacle(self) -> None:
        self.obstacle.remove()
        self._log("obstacle removed")

    def drop_obstacle_in_path(self) -> Optional[np.ndarray]:
        """Place the obstacle on the upcoming part of whatever is moving; None if nothing is."""
        m = self._motion
        if m is None or self.run_state not in (RunState.RUNNING, RunState.BLOCKED) and self._manual is None:
            return None
        if m["kind"] == "joint":
            path, i, goal = m["path"], m["i"], m["goal"]
            base = np.asarray(self.arm.base.A)
            tips = [self.checker.envelope.points(path[j], base)[-1] for j in range(i, len(path))]
            mid = len(tips) // 2
            # Prefer a spot on the upcoming path that leaves the arm and the goal clear, so the
            # obstacle is something to steer round rather than something already touching.
            centre = tips[mid]
            for j in [*range(mid, len(tips)), *range(mid - 1, -1, -1)]:
                self.obstacle.place(tips[j])
                if (self.checker.config_hit(self.arm.q, False) is None
                        and self.checker.config_hit(goal, False) is None):
                    centre = tips[j]
                    break
            self.obstacle.remove()
        elif m["kind"] == "tool":
            centre = 0.5 * (m["p0"] + m["p1"])
        else:  # base
            heading = m["sign"] * np.array([math.cos(self.base.yaw), math.sin(self.base.yaw)])
            xy = np.array([self.base.x, self.base.y]) + heading * 0.45
            centre = np.array([xy[0], xy[1], lc.OBSTACLE_RADIUS])
        self.place_obstacle(centre)
        return centre

    # ---------------------------------------------------------- the tick
    def tick(self) -> bool:
        """Advance the simulation by one ``dt``. Returns True while the job is unfinished."""
        self._safety_update()
        if self.run_state in (RunState.RUNNING, RunState.BLOCKED):
            try:
                next(self._gen)
            except StopIteration:
                self.run_state = RunState.DONE
                self.phase = "done"
                self._motion = None
                self._log("job complete")
            except StationFault as err:
                self.run_state = RunState.FAULT
                self.fault = str(err)
                self.phase = "FAULT"
                self._motion = None
                self._log(f"FAULT: {err}")
        else:
            self.base.cmd_v = self.base.cmd_w = 0.0
            if self.manual_allowed:
                self._manual_step()
            self._sync()  # generators redraw as they move; manual jogs need it here
        self._sync_obstacle()
        if self.equipment is not None:
            self.equipment.update(self.run_state.value, self._estop_active, self.inputs.curtain_broken)
        self.env.step(self.dt)
        self.sim_time += self.dt
        return self.run_state in _BUSY

    def run(self, max_ticks: Optional[int] = None) -> RunState:
        """Start (if idle) and run to completion."""
        if self.run_state is RunState.IDLE:
            self.start()
        ticks = 0
        while self.tick():
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
        return self.run_state

    def status(self) -> dict:
        """Snapshot for a GUI or log: state, phase, base pose, arm joints, tool position."""
        tcp = self.arm.fkine(self.arm.q)
        return {
            "state": self.run_state.value,
            "phase": self.phase,
            "fault": self.fault,
            "cars_placed": self.placed,
            "base_xy_yaw": (self.base.x, self.base.y, self.base.yaw),
            "base_cmd_v_w": (self.base.cmd_v, self.base.cmd_w),
            "arm_q_deg": np.degrees(self.arm.q).round(1).tolist(),
            "tcp_xyz": tcp.t.round(3).tolist(),
            "manipulability": self.manipulability,
            "holding_car": self._held is not None,
            "estop_pressed": self._estop_active,
            "curtain_broken": self.inputs.curtain_broken,
            "zone_occupied": self.inputs.zone_occupied,
            "obstacle_active": self.obstacle.active,
            "block_reason": self.block_reason,
            "replans": self.replans,
            "vision_error_px": self.vision_error_px,
            "manual_allowed": self.manual_allowed,
            "message": self.message,
        }

    def _log(self, text: str) -> None:
        self.message = text
        self.events.append(f"[{self.sim_time:6.1f}s] {text}")

    # ------------------------------------------------------ manual control
    @property
    def manual_allowed(self) -> bool:
        """True when the operator may jog the arm or send it to a joint state."""
        if self._estop_active or self.inputs.tripped:
            return False
        if self.run_state in _MANUAL_OK:
            return True
        return self.run_state is RunState.STOPPED and self._resume_state in _MANUAL_OK

    def set_jog(self, kind: str, index: int, sign: int) -> bool:
        """Start jogging (held button). ``kind`` is 'joint' (index 0-5) or 'cart' (index 0-2 = x, y, z)."""
        if not self.manual_allowed or self._manual is not None:
            return False
        if kind not in ("joint", "cart") or sign not in (-1, 1):
            raise ValueError("kind must be 'joint' or 'cart' and sign +-1")
        self._jog = (kind, int(index), int(sign))
        return True

    def stop_jog(self) -> None:
        self._jog = None

    def _manual_step(self) -> None:
        if self._manual is not None:
            try:
                next(self._manual)
            except StopIteration:
                self._manual = None
                self._motion = None
            except StationFault as err:
                self._manual = None
                self._motion = None
                self._log(f"move stopped: {err}")
        elif self._jog is not None:
            self._apply_jog(*self._jog)

    def _apply_jog(self, kind: str, index: int, sign: int) -> None:
        q = np.array(self.arm.q, dtype=float)
        lo, hi = self.arm.qlim_array
        if kind == "joint":
            q_new = q.copy()
            q_new[index] += sign * lc.JOG_JOINT_SPEED * self.dt
            if q_new[index] < lo[index] or q_new[index] > hi[index]:
                self.message = f"joint {index + 1} at its limit"
                return
        else:
            v = np.zeros(6)
            v[index] = sign * lc.JOG_CART_SPEED
            try:
                q_new = self._resolved_rate(v)
            except StationFault as err:
                self.message = f"jog stopped: {err}"
                return
        hit = self.checker.config_hit(q_new)
        if hit:
            self.message = f"jog blocked: would hit {hit}"
            return
        self.arm.q = q_new
        self.message = ""

    def goto_q(self, q_goal) -> bool:
        """Send the arm to a joint state (radians) by a collision-checked joint trajectory.

        Works when the station is idle, done, faulted or stopped between runs. Returns
        False (and logs why) if the move is refused.
        """
        q_goal = np.asarray(q_goal, dtype=float)
        if q_goal.shape != (6,):
            raise ValueError("q_goal needs 6 joint angles")
        if not self.manual_allowed or self._manual is not None:
            self._log("goto refused: station is not in a state that allows manual moves")
            return False
        lo, hi = self.arm.qlim_array
        if np.any(q_goal < lo - 1e-9) or np.any(q_goal > hi + 1e-9):
            self._log("goto refused: a joint is outside its limit")
            return False
        hit = self.checker.config_hit(q_goal)
        if hit:
            self._log(f"goto refused: target would hit {hit}")
            return False
        self._jog = None
        self._manual = self._arm_joint_move(q_goal)
        self._log(f"moving to joint state {np.degrees(q_goal).round(1).tolist()} deg")
        return True

    def go_home(self) -> bool:
        """Move the arm to the stowed travel pose."""
        return self.goto_q(lc.Q_TRAVEL)

    # ------------------------------------------------------------ the job
    def sequence(self) -> Iterator[None]:
        """Generator for the whole job; yields once per tick."""
        for k in range(self.n_cars):
            self.phase = f"car {k + 1}/{self.n_cars}: driving to supply table"
            yield from self._travel_to(lc.TABLE_SLOT_XS[k])
            self.phase = f"car {k + 1}/{self.n_cars}: picking"
            yield from self._pick(k)
            self.phase = f"car {k + 1}/{self.n_cars}: driving to conveyor"
            yield from self._travel_to(lc.BELT_SLOT_XS[k])
            self.phase = f"car {k + 1}/{self.n_cars}: placing"
            yield from self._place(k)
        self.phase = "returning to dock"
        yield from self._travel_to(lc.STATION.x)

    # ------------------------------------------------- holding for obstacles
    def _hold_while(self, blocked: Callable[[], bool], reason: str) -> Iterator[None]:
        """Stand still (one yield per tick) for as long as ``blocked()`` is true."""
        if not blocked():
            return
        self.block_reason = reason
        self._log(f"BLOCKED: {reason}")
        if self.run_state is RunState.RUNNING:
            self.run_state = RunState.BLOCKED
        while blocked():
            yield
        self.block_reason = None
        if self.run_state is RunState.BLOCKED:
            self.run_state = RunState.RUNNING
        self._log("path clear, continuing")

    def _guarded(self, motion: Iterator[None], blocked: Callable[[], bool], reason: str) -> Iterator[None]:
        """Run a base motion, holding still whenever ``blocked()`` becomes true."""
        while True:
            if blocked():
                yield from self._hold_while(blocked, reason)
            try:
                next(motion)
            except StopIteration:
                return
            self._sync()
            yield

    def _base_blocked(self, sign: Optional[int], remaining: float) -> bool:
        """True if the obstacle is in the footprint swept by the next stretch of driving."""
        ob = self.obstacle
        if not ob.active or ob.centre[2] - ob.radius > 0.9:
            return False
        p = np.array([self.base.x, self.base.y])
        a, b = p, p
        if sign is not None and remaining > 0:
            heading = np.array([math.cos(self.base.yaw), math.sin(self.base.yaw)])
            b = p + sign * heading * min(remaining, lc.BASE_LOOKAHEAD)
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(ob.centre[:2] - a, ab) / np.dot(ab, ab), 0, 1))
        dist = float(np.linalg.norm(a + ab * t - ob.centre[:2]))
        return dist < lc.BASE_CLEARANCE_RADIUS + ob.radius + lc.COLLISION_MARGIN

    # ---------------------------------------------------------- base moves
    def _travel_to(self, x_goal: float) -> Iterator[None]:
        """Move the base to the work position at x_goal, ending facing the belt/table."""
        dx = x_goal - self.base.x
        if abs(dx) < 1e-6:
            return
        out = abs(lc.STAGING_Y) - abs(lc.WORK_Y)
        yield from self._drive(-out)
        yield from self._turn(0.0 if dx > 0 else math.pi)
        yield from self._drive(abs(dx))
        yield from self._turn(lc.WORK_YAW)
        yield from self._drive(out)

    def _drive(self, distance: float, detour: bool = True) -> Iterator[None]:
        """Drive straight; hold still for an obstacle, or steer round it when driving along the lane."""
        sign = 1 if distance >= 0 else -1
        self._motion = {"kind": "base", "sign": sign}
        remaining = abs(distance)
        motion = self.base.drive(sign * remaining, self.dt, self.speed_scale)
        x0, y0 = self.base.x, self.base.y

        def left() -> float:
            return remaining - math.hypot(self.base.x - x0, self.base.y - y0)

        def blocked() -> bool:
            return self._base_blocked(sign, left())

        while True:
            if blocked():
                plan = self._detour_plan(sign, left()) if detour else None
                if plan is None:
                    yield from self._hold_while(blocked, "obstacle in the base's path")
                else:
                    y_d, forward = plan
                    rest = left() - forward
                    yield from self._detour(sign, y_d, forward)
                    if rest <= 1e-6:
                        break
                    remaining = rest
                    motion = self.base.drive(sign * remaining, self.dt, self.speed_scale)
                    x0, y0 = self.base.x, self.base.y
                    self._motion = {"kind": "base", "sign": sign}
                    continue
            try:
                next(motion)
            except StopIteration:
                break
            self._sync()
            yield
        self._motion = None

    def _detour_plan(self, sign: int, remaining: float) -> Optional[tuple]:
        """Lane to shift to and distance to drive along it, or None if the obstacle can't be passed.

        Only used while driving along the belt (heading along x). The base must still be
        well short of the obstacle, and have room beside it between the belt and the fence.
        """
        yaw = self.base.yaw
        if abs(math.sin(yaw)) > 0.1:
            return None
        ob = self.obstacle
        r = lc.BASE_CLEARANCE_RADIUS + ob.radius + lc.COLLISION_MARGIN + 0.03  # a little more than the stop distance
        ahead = sign * math.cos(yaw) * (ob.centre[0] - self.base.x)
        forward = ahead + r
        if ahead < r or forward > remaining:
            return None  # too close to turn aside, or the destination itself is blocked
        lo_y = cfg.BELT_WIDTH / 2 + lc.BASE_CLEARANCE_RADIUS + 0.01
        hi_y = lc.ZONE_Y[1] - lc.BASE_CLEARANCE_RADIUS - 0.01
        options = [y for y in (ob.centre[1] + r, ob.centre[1] - r) if lo_y <= y <= hi_y]
        if not options:
            return None
        return min(options, key=lambda y: abs(y - self.base.y)), forward

    def _detour(self, sign: int, y_d: float, forward: float) -> Iterator[None]:
        """Shift to lane ``y_d``, drive ``forward`` along it, then shift back to the original lane."""
        yaw, y_lane = self.base.yaw, self.base.y
        self.replans += 1
        self._log(f"obstacle ahead: steering round it via y={y_d:.2f} m")
        for target_y in (y_d, y_lane):
            dy = target_y - self.base.y
            if abs(dy) > 1e-6:
                yield from self._turn(math.pi / 2 if dy > 0 else -math.pi / 2)
                yield from self._drive(abs(dy), detour=False)
                yield from self._turn(yaw)
            if target_y == y_d:
                yield from self._drive(sign * forward, detour=False)

    def _turn(self, yaw_goal: float) -> Iterator[None]:
        yield from self._guarded(self.base.turn_to(yaw_goal, self.dt, self.speed_scale),
                                 lambda: self._base_blocked(None, 0.0), "obstacle beside the base")

    # ------------------------------------------------------------ pick/place
    def _pick(self, k: int) -> Iterator[None]:
        car = self.cars[k]
        nominal = self.grasp_tcp(self.pick_pose(k))
        pre = SE3(0, 0, lc.PRE_HEIGHT) * nominal
        yield from self._arm_joint_move(self._ik(pre))
        # The car may not be exactly in its slot: let the camera line the gripper up.
        self.phase = f"car {k + 1}/{self.n_cars}: visual servoing on car"
        yield from self._visual_servo(car)
        self.phase = f"car {k + 1}/{self.n_cars}: picking"
        aligned = self.arm.fkine(self.arm.q)
        grasp = SE3(0, 0, -lc.PRE_HEIGHT) * aligned
        yield from self._rmrc(grasp)
        yield from self._gripper_to(cfg.CAR_SIZE[1])  # close on the car's width
        tcp = self.arm.fkine(self.arm.q)
        car_now = SE3(np.asarray(car.T), check=False)
        if np.linalg.norm(car_now.t - tcp.t) > lc.GRASP_LIFT + 5e-3:
            raise StationFault(f"car {k + 1} is not where the gripper closed")
        self._held = (car, tcp.inv() * car_now)
        yield from self._rmrc(SE3(0, 0, lc.PRE_HEIGHT) * tcp)
        yield from self._arm_joint_move(lc.Q_CARRY)  # carry the car high so it clears the belt edge

    def _place(self, k: int) -> Iterator[None]:
        target = self.place_pose(k)
        grasp = self.grasp_tcp(target)
        pre = SE3(0, 0, lc.PLACE_PRE_HEIGHT) * grasp
        yield from self._arm_joint_move(self._ik(pre))
        yield from self._rmrc(grasp)
        yield from self._gripper_to(OPEN_WIDTH)
        car, _ = self._held
        self._held = None
        car.T = target.A  # settle exactly on the slot
        self.placed += 1
        if self.on_car_placed:
            self.on_car_placed(k, car)
        yield from self._rmrc(pre)
        yield from self._arm_joint_move(lc.Q_TRAVEL)

    # --------------------------------------------------------------- vision
    def _visual_servo(self, car: sg.Cuboid) -> Iterator[None]:
        """Image-based visual servoing: move the camera until the car looks like the goal image."""
        s = self.speed_scale
        for _ in range(lc.IBVS_MAX_TICKS):
            tcp = self.arm.fkine(self.arm.q)
            T_cam = tcp * self._cam_mount
            T_car = SE3(np.asarray(car.T), check=False)
            v_cam, err = self._servo.twist(T_cam, T_car, scale=s)
            self.vision_error_px = err
            if err < lc.IBVS_TOL_PX:
                self._log(f"visual servoing converged ({err:.2f} px)")
                return
            v = camera_twist_to_tool_twist(v_cam, T_cam, tcp.t)
            v = limit_twist(v, lc.IBVS_MAX_LIN * s, lc.IBVS_MAX_ANG * s)
            q_new = self._resolved_rate(v)
            if self.checker.config_hit(q_new, with_statics=False):
                yield from self._hold_while(lambda: self.checker.config_hit(q_new, with_statics=False) is not None,
                                            "obstacle at the car")
            self.arm.q = q_new
            self._sync()
            yield
        raise StationFault("visual servoing did not converge")

    # --------------------------------------------------------------- arm moves
    def _ik(self, goal_world: SE3) -> np.ndarray:
        """Joint solution for a world-frame tool pose, near the usual reach configuration."""
        local = self.arm.base.inv() * goal_world
        sol = self._kin.ikine_LM(local, q0=lc.Q_IK_SEED, ilimit=100, slimit=20)
        if not sol.success:
            raise StationFault("target is out of reach for the arm")
        q = np.asarray(sol.q, dtype=float)
        lo, hi = self.arm.qlim_array
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            raise StationFault("target needs a joint outside its limits")
        return q

    def _joint_path(self, waypoints: list) -> np.ndarray:
        """Quintic joint trajectory through the waypoints, one sample per tick (start excluded)."""
        segments = []
        for q0, q1 in zip(waypoints[:-1], waypoints[1:]):
            duration = max(lc.MIN_MOVE_TIME, 1.875 * np.abs(q1 - q0).max() / lc.ARM_JOINT_SPEED)
            n = max(2, math.ceil(duration / self.speed_scale / self.dt))
            segments.append(rtb.jtraj(q0, q1, n).q[1:])
        return np.vstack(segments)

    def _straight_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(8, int(np.abs(b - a).max() / 0.03))
        return self.checker.path_hit(np.linspace(a, b, n), stride=2) is None

    def _replan(self, q0: np.ndarray, q_goal: np.ndarray) -> Optional[np.ndarray]:
        lo, hi = self.arm.qlim_array
        vias = plan_around(q0, q_goal, self._straight_free, lo, hi, lc.REPLAN_TRIES, lc.REPLAN_SEED)
        if vias is None:
            return None
        return self._joint_path([q0, *vias, q_goal])

    def _arm_joint_move(self, q_goal: np.ndarray) -> Iterator[None]:
        """Quintic joint-space move to ``q_goal`` that watches the path ahead.

        If an obstacle shows up in the next stretch of the path the arm stops and plans
        a new path round it (via-point search). If none exists it holds still until the
        obstacle is removed.
        """
        q_goal = np.asarray(q_goal, dtype=float)
        path = self._joint_path([np.array(self.arm.q, dtype=float), q_goal])
        info = {"kind": "joint", "path": path, "i": 0, "goal": q_goal}
        self._motion = info
        i = 0
        while i < len(path):
            if self.obstacle.active:
                ahead = path[i:i + lc.LOOKAHEAD_TICKS]
                if self.checker.path_hit(ahead, stride=4, with_statics=False) is not None:
                    new_path = self._replan(np.array(self.arm.q, dtype=float), q_goal)
                    if new_path is not None:
                        path, i = new_path, 0
                        info["path"], info["i"] = path, 0
                        self.replans += 1
                        self._log(f"obstacle ahead: re-planned around it ({len(path)} steps)")
                        continue
                    yield from self._hold_while(
                        lambda: self.checker.path_hit(path[i:i + lc.LOOKAHEAD_TICKS], 4, False) is not None,
                        "obstacle in the planned path (no way round)")
                    continue
            self.arm.q = path[i]
            self._sync()
            i += 1
            info["i"] = i
            yield
        self._motion = None

    def _resolved_rate(self, v: np.ndarray) -> np.ndarray:
        """Joint solution one ``dt`` ahead for a world-frame tool twist ``v`` (damped least squares)."""
        q = np.array(self.arm.q, dtype=float)
        J = self.arm.jacob0(q)
        self.manipulability = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        if self.manipulability < lc.MANIPULABILITY_MIN:
            raise StationFault(f"near a singularity (manipulability {self.manipulability:.4f})")
        damping = 0.0
        if self.manipulability < lc.MANIPULABILITY_DAMP:
            damping = lc.RMRC_DAMPING**2 * (1 - (self.manipulability / lc.MANIPULABILITY_DAMP) ** 2)
        q_new = q + (J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), v)) * self.dt
        lo, hi = self.arm.qlim_array
        if np.any(q_new < lo) or np.any(q_new > hi):
            raise StationFault("RMRC move would exceed a joint limit")
        return q_new

    def _rmrc(self, goal_world: SE3) -> Iterator[None]:
        """Straight-line Cartesian move of the tool to goal_world by resolved-rate control."""
        start = self.arm.fkine(self.arm.q)
        length = float(np.linalg.norm(goal_world.t - start.t))
        duration = max(lc.MIN_MOVE_TIME, length / lc.RMRC_SPEED)
        n = max(2, math.ceil(duration / self.speed_scale / self.dt))
        path = rtb.ctraj(start, goal_world, n)
        self._motion = {"kind": "tool", "p0": np.array(start.t), "p1": np.array(goal_world.t)}

        def blocked() -> bool:
            if not self.obstacle.active:
                return False
            tcp = self.arm.fkine(self.arm.q).t
            return (self.checker.tool_line_hit(tcp, goal_world.t)
                    or self.checker.config_hit(self.arm.q, with_statics=False) is not None)

        for k in range(1, n):
            if blocked():
                yield from self._hold_while(blocked, "obstacle on the tool's straight-line path")
            q = np.array(self.arm.q, dtype=float)
            now = self.arm.fkine(q)
            want = path[k]
            err_rot = smb.vex(smb.trlog(want.R @ now.R.T, check=False))
            v = np.r_[(want.t - now.t) / self.dt, err_rot / self.dt]
            self.arm.q = self._resolved_rate(v)
            self._sync()
            yield
        self._motion = None
        if np.linalg.norm(self.arm.fkine(self.arm.q).t - goal_world.t) > lc.POSITION_TOLERANCE:
            raise StationFault("RMRC move ended off target")

    def _gripper_to(self, width: float) -> Iterator[None]:
        w0 = self.gripper.opening
        n = max(1, math.ceil(lc.GRIPPER_TIME / self.speed_scale / self.dt))
        for i in range(1, n + 1):
            self.gripper.opening = w0 + (width - w0) * i / n
            self._sync()
            yield

    # -------------------------------------------------------------- drawing
    def _sync(self) -> None:
        """Place every drawn part from the current base pose, arm joints and gripper."""
        pose = self.base.pose
        if self._last_base is None or not np.array_equal(self._last_base, pose.A):
            self._last_base = pose.A.copy()
            self.plate.T = (pose * SE3(0, 0, TOP_HEIGHT + lc.MOUNT_PLATE_SIZE[2] / 2)).A
            self.arm.base = pose * SE3(0, 0, lc.MOUNT_HEIGHT)
        tcp = self.arm.fkine(self.arm.q)
        self.gripper.update(tcp, self.gripper.opening)
        if self._held is not None:
            car, rel = self._held
            car.T = (tcp * rel).A
        self._sync_obstacle()

    def _sync_obstacle(self) -> None:
        """Move the obstacle's drawing to the obstacle (below the floor when there is none)."""
        key = (tuple(np.round(self.obstacle.centre, 6)), self.obstacle.active)
        if key != self._obstacle_last:
            self._obstacle_last = key
            self._obstacle_shape.T = SE3(*self.obstacle.centre).A
