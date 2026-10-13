from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from ir_support_extra_robots.robots import Turtlebot3Waffle
from spatialmath import SE3

# TurtleBot3 Waffle datasheet values
WHEEL_RADIUS = 0.033  # m
WHEEL_SEPARATION = 0.287  # m
TOP_HEIGHT = 0.138  # m, top of the mesh above the floor
FOOTPRINT_RADIUS = 0.215  # m, circle around the centre that encloses the body


@dataclass(frozen=True)
class DriveLimits:

    max_lin: float = 0.26  # m/s
    max_ang: float = 1.0  # rad/s
    acc_lin: float = 0.25  # m/s^2
    acc_ang: float = 1.5  # rad/s^2


def wrap_angle(angle: float) -> float:
    #Wrap an angle to (-pi, pi]
    return math.atan2(math.sin(angle), math.cos(angle))


def trapezoid_profile(distance: float, v_max: float, a_max: float, dt: float):
    
    #Sample a rest-to-rest trapezoidal speed profile
    if distance <= 1e-9:
        return np.array([0.0]), np.array([0.0])
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if distance >= 2 * d_acc:
        t_cruise = (distance - 2 * d_acc) / v_max
        v_peak = v_max
    else:  # triangular: never reaches v_max
        t_acc = math.sqrt(distance / a_max)
        t_cruise = 0.0
        v_peak = a_max * t_acc
    t_total = 2 * t_acc + t_cruise
    n = max(1, math.ceil(t_total / dt))
    t = np.minimum(np.arange(1, n + 1) * dt, t_total)

    s = np.empty_like(t)
    v = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < t_acc:
            s[i], v[i] = 0.5 * a_max * ti**2, a_max * ti
        elif ti < t_acc + t_cruise:
            s[i], v[i] = d_acc + v_peak * (ti - t_acc), v_peak
        else:
            td = t_total - ti
            s[i], v[i] = distance - 0.5 * a_max * td**2, a_max * td
    s[-1], v[-1] = distance, 0.0
    return s, v


class TurtleBotBase:
    
    #Planar differential-drive base with the TurtleBot3 Waffle mesh.
    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, limits: DriveLimits | None = None):
        self.limits = limits or DriveLimits()
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.cmd_v = 0.0  # last commanded linear speed (m/s)
        self.cmd_w = 0.0  # last commanded angular speed (rad/s)
        self.model = Turtlebot3Waffle(base=self.pose)

    # --- pose ---------------------------------------------------------------
    @property
    def pose(self) -> SE3:
        """Base pose in the world frame (z = 0: it drives on the floor)."""
        return SE3(self.x, self.y, 0.0) * SE3.Rz(self.yaw)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = float(x), float(y), wrap_angle(float(yaw))
        self.model.base = self.pose

    def add_to_env(self, env) -> None:
        self.model.add_to_env(env)

    @staticmethod
    def wheel_speeds(v: float, w: float) -> tuple[float, float]:
        """Left and right wheel angular speeds (rad/s) for body speeds ``v``, ``w``."""
        left = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        right = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        return left, right

    # --- motion primitives (generators; one yield per tick) -------------------
    def turn_to(self, yaw_goal: float, dt: float, speed_scale: float = 1.0) -> Iterator[None]:
        """Rotate on the spot to ``yaw_goal`` by the shortest way."""
        delta = wrap_angle(yaw_goal - self.yaw)
        yield from self._rotate(delta, dt, speed_scale)

    def _rotate(self, delta: float, dt: float, speed_scale: float) -> Iterator[None]:
        yaw0 = self.yaw
        sign = 1.0 if delta >= 0 else -1.0
        s, v = trapezoid_profile(
            abs(delta), self.limits.max_ang * speed_scale, self.limits.acc_ang * speed_scale**2, dt
        )
        for si, vi in zip(s, v):
            self.cmd_v, self.cmd_w = 0.0, sign * vi
            self.set_pose(self.x, self.y, yaw0 + sign * si)
            yield
        self.cmd_v = self.cmd_w = 0.0

    def drive(self, distance: float, dt: float, speed_scale: float = 1.0) -> Iterator[None]:
        """Drive straight along the current heading; negative ``distance`` reverses."""
        x0, y0 = self.x, self.y
        sign = 1.0 if distance >= 0 else -1.0
        s, v = trapezoid_profile(
            abs(distance), self.limits.max_lin * speed_scale, self.limits.acc_lin * speed_scale**2, dt
        )
        for si, vi in zip(s, v):
            self.cmd_v, self.cmd_w = sign * vi, 0.0
            self.set_pose(x0 + sign * si * math.cos(self.yaw), y0 + sign * si * math.sin(self.yaw), self.yaw)
            yield
        self.cmd_v = self.cmd_w = 0.0

    def drive_to_point(self, x: float, y: float, dt: float, speed_scale: float = 1.0) -> Iterator[None]:
        """Turn towards ``(x, y)`` and drive there in a straight line (forwards)."""
        dx, dy = x - self.x, y - self.y
        if math.hypot(dx, dy) < 1e-6:
            return
        yield from self.turn_to(math.atan2(dy, dx), dt, speed_scale)
        yield from self.drive(math.hypot(dx, dy), dt, speed_scale)
