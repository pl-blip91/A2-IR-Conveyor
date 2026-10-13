#Loading station: a Yaskawa MH5 on a TurtleBot3 loads toy cars onto the conveyor.
from __future__ import annotations

import math
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

from . import config as cfg
from . import loading_config as lc


class StationFault(RuntimeError):
    """Raised inside the sequence when it must stop (unreachable pose, singularity, etc)."""


class RunState(Enum):
    IDLE = "IDLE"  # not started
    RUNNING = "RUNNING"
    ESTOPPED = "ESTOPPED"  # stop latched; needs reset() then resume()
    STOPPED = "STOPPED"  # reset done; waiting for resume()
    DONE = "DONE"
    FAULT = "FAULT"  # the run was aborted; see ``fault``


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


# --------------------------------------------------------------------------------------
# The station
# --------------------------------------------------------------------------------------
class LoadingStation:
    
    #MH5-on-TurtleBot loading task.
    def __init__(
        self,
        env,
        n_cars: int = lc.MAX_CARS,
        dt: float = 0.05,
        speed_scale: float = 1.0,
        on_car_placed: Optional[Callable[[int, sg.Cuboid], None]] = None,
        build_world: bool = True,
    ):
        self.env = env
        self.n_cars = n_cars
        self.dt = dt
        self.speed_scale = speed_scale
        self.on_car_placed = on_car_placed

        self.area = build_loading_area(env, n_cars) if build_world else {"cars": []}
        self.cars = self.area["cars"]

        park = lc.STATION.robot_base
        self.base = TurtleBotBase(park.t[0], park.t[1], lc.WORK_YAW)
        self.arm = YaskawaMH5(tool_offset=TCP_OFFSET)
        self.gripper = ParallelGripper()
        self.plate = sg.Cuboid(scale=list(lc.MOUNT_PLATE_SIZE), color=list(lc.MOUNT_PLATE_COLOUR))
        self._kin = self.arm.kinematics_only()

        self.base.add_to_env(env)
        self.arm.add_to_env(env)
        self.gripper.add_to_env(env)
        env.add(self.plate)

        self.arm.q = lc.Q_TRAVEL
        self.gripper.opening = OPEN_WIDTH
        self._held: Optional[tuple[sg.Cuboid, SE3]] = None
        self._last_base: Optional[np.ndarray] = None
        self._sync()

        self.run_state = RunState.IDLE
        self.phase = "idle"
        self.fault: Optional[str] = None
        self.placed = 0
        self.manipulability = float("nan")
        self.sim_time = 0.0
        self._estop_active = False
        self._phase_before_stop = "idle"
        self._gen: Optional[Iterator[None]] = None

    # ---------------------------------------------------------------- poses
    def pick_pose(self, k: int) -> SE3:
        """Pose of car ``k`` on the supply table."""
        return car_pose(lc.TABLE_SLOT_XS[k], lc.TABLE_HEIGHT)

    def place_pose(self, k: int) -> SE3:
        """Pose of car ``k`` once it is on the belt."""
        return car_pose(lc.BELT_SLOT_XS[k], cfg.BELT_HEIGHT)

    @staticmethod
    def grasp_tcp(car: SE3) -> SE3:
        """Grip-centre pose that grasps ``car``: gripper pointing down, fingers across the car."""
        return SE3(car.t + np.array([0.0, 0.0, lc.GRASP_LIFT])) * SE3.Rx(math.pi)

    # --------------------------------------------------------- run control
    def start(self) -> None:
        """Begin the loading sequence (from IDLE)."""
        if self.run_state is not RunState.IDLE:
            raise RuntimeError(f"cannot start from {self.run_state.value}")
        if self._estop_active:
            raise RuntimeError("cannot start while the emergency stop is pressed")
        self._gen = self.sequence()
        self.run_state = RunState.RUNNING

    def emergency_stop(self) -> None:
        """Stop everything now and latch ``ESTOPPED`` (simulated GUI or physical e-stop)."""
        self._estop_active = True
        if self.run_state is RunState.RUNNING:
            self._phase_before_stop = self.phase
        if self.run_state in (RunState.RUNNING, RunState.STOPPED):
            self.run_state = RunState.ESTOPPED
            self.phase = "EMERGENCY STOP"

    def release_estop(self) -> None:
        """The stop button was released. This alone never restarts motion."""
        self._estop_active = False

    def reset(self) -> bool:
        """Acknowledge a stop. Only works once the stop input is released."""
        if self.run_state is RunState.ESTOPPED and not self._estop_active:
            self.run_state = RunState.STOPPED
            self.phase = "stopped - press resume"
            return True
        return False

    def resume(self) -> bool:
        """Continue the sequence after ``reset()``."""
        if self.run_state is RunState.STOPPED and not self._estop_active:
            self.run_state = RunState.RUNNING
            self.phase = self._phase_before_stop
            return True
        return False

    def tick(self) -> bool:
        """Advance the simulation by one ``dt``. Returns True while the job is unfinished."""
        if self.run_state is RunState.RUNNING:
            try:
                next(self._gen)
            except StopIteration:
                self.run_state = RunState.DONE
                self.phase = "done"
            except StationFault as err:
                self.run_state = RunState.FAULT
                self.fault = str(err)
                self.phase = "FAULT"
        else:
            self.base.cmd_v = self.base.cmd_w = 0.0
        self.env.step(self.dt)
        self.sim_time += self.dt
        return self.run_state in (RunState.RUNNING, RunState.ESTOPPED, RunState.STOPPED)

    def run(self, max_ticks: Optional[int] = None) -> RunState:
        #Start and run to completion
        if self.run_state is RunState.IDLE:
            self.start()
        ticks = 0
        while self.tick():
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
        return self.run_state

    def status(self) -> dict:
        #Snapshot for a GUI or log: state, phase, base pose, arm joints, tool position.
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
        }

    # ------------------------------------------------------------ the job
    def sequence(self) -> Iterator[None]:
        #Generator for the whole job; yields once per tick.
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

    # ---------------------------------------------------------- base moves
    def _travel_to(self, x_goal: float) -> Iterator[None]:
        #Move the base to the work position at x_goal, ending facing the belt/table.

        dx = x_goal - self.base.x
        if abs(dx) < 1e-6:
            return
        out = abs(lc.STAGING_Y) - abs(lc.WORK_Y)
        s = self.speed_scale
        yield from self._with_sync(self.base.drive(-out, self.dt, s))
        yield from self._with_sync(self.base.turn_to(0.0 if dx > 0 else math.pi, self.dt, s))
        yield from self._with_sync(self.base.drive(abs(dx), self.dt, s))
        yield from self._with_sync(self.base.turn_to(lc.WORK_YAW, self.dt, s))
        yield from self._with_sync(self.base.drive(out, self.dt, s))

    def _with_sync(self, motion: Iterator[None]) -> Iterator[None]:
        for _ in motion:
            self._sync()
            yield

    # ------------------------------------------------------------ pick/place
    def _pick(self, k: int) -> Iterator[None]:
        car = self.cars[k]
        grasp = self.grasp_tcp(self.pick_pose(k))
        pre = SE3(0, 0, lc.PRE_HEIGHT) * grasp
        yield from self._arm_joint_move(self._ik(pre))
        yield from self._rmrc(grasp)
        yield from self._gripper_to(cfg.CAR_SIZE[1])  # close on the car's width
        tcp = self.arm.fkine(self.arm.q)
        car_now = SE3(np.asarray(car.T), check=False)
        if np.linalg.norm(car_now.t - tcp.t) > lc.GRASP_LIFT + 5e-3:
            raise StationFault(f"car {k + 1} is not where the gripper closed")
        self._held = (car, tcp.inv() * car_now)
        yield from self._rmrc(pre)
        yield from self._arm_joint_move(lc.Q_TRAVEL)

    def _place(self, k: int) -> Iterator[None]:
        target = self.place_pose(k)
        grasp = self.grasp_tcp(target)
        pre = SE3(0, 0, lc.PRE_HEIGHT) * grasp
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

    def _arm_joint_move(self, q_goal: np.ndarray) -> Iterator[None]:
        """Quintic joint-space move to ``q_goal``."""
        q0 = np.array(self.arm.q, dtype=float)
        duration = max(lc.MIN_MOVE_TIME, 1.875 * np.abs(q_goal - q0).max() / lc.ARM_JOINT_SPEED)
        n = max(2, math.ceil(duration / self.speed_scale / self.dt))
        for q in rtb.jtraj(q0, q_goal, n).q[1:]:
            self.arm.q = q
            self._sync()
            yield

    def _rmrc(self, goal_world: SE3) -> Iterator[None]:
        
        #Straight-line Cartesian move of the tool to goal_world by resolved-rate control.
        start = self.arm.fkine(self.arm.q)
        length = float(np.linalg.norm(goal_world.t - start.t))
        duration = max(lc.MIN_MOVE_TIME, length / lc.RMRC_SPEED)
        n = max(2, math.ceil(duration / self.speed_scale / self.dt))
        path = rtb.ctraj(start, goal_world, n)
        lo, hi = self.arm.qlim_array
        for k in range(1, n):
            q = np.array(self.arm.q, dtype=float)
            now = self.arm.fkine(q)
            want = path[k]
            err_rot = smb.vex(smb.trlog(want.R @ now.R.T, check=False))
            v = np.r_[(want.t - now.t) / self.dt, err_rot / self.dt]

            J = self.arm.jacob0(q)
            self.manipulability = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
            if self.manipulability < lc.MANIPULABILITY_MIN:
                raise StationFault(f"near a singularity (manipulability {self.manipulability:.4f})")
            damping = 0.0
            if self.manipulability < lc.MANIPULABILITY_DAMP:
                damping = lc.RMRC_DAMPING**2 * (1 - (self.manipulability / lc.MANIPULABILITY_DAMP) ** 2)
            q_new = q + (J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), v)) * self.dt
            if np.any(q_new < lo) or np.any(q_new > hi):
                raise StationFault("RMRC move would exceed a joint limit")
            self.arm.q = q_new
            self._sync()
            yield
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
        #Place every drawn part from the current base pose, arm joints and gripper.
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
