"""Simulated eye-in-hand camera and image-based visual servoing (IBVS).

The camera looks along the tool z axis from the flange. It sees the four corners of
the top face of a toy car. IBVS drives the tool so those corners appear where they
would if the gripper were exactly above the car, which corrects for the car being
slightly off its nominal slot on the supply table.

Camera frame: x right, y down, z forward (along the optical axis). The controller only
uses what the camera measures; the car's true pose is used solely to render the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from spatialmath import SE3

from . import config as cfg
from . import loading_config as lc


@dataclass(frozen=True)
class Camera:
    """Pinhole camera with a square sensor and the principal point at the centre."""

    focal: float = lc.CAM_FOCAL
    pixel: float = lc.CAM_PIXEL
    res: int = lc.CAM_RES

    @property
    def f_px(self) -> float:
        return self.focal / self.pixel

    def project(self, T_world_cam: SE3, points_world: np.ndarray):
        """Pixel coordinates (n x 2) and depths (n,) of world points."""
        Tinv = np.linalg.inv(np.asarray(T_world_cam.A))
        P = (Tinv @ np.c_[points_world, np.ones(len(points_world))].T).T[:, :3]
        z = P[:, 2]
        uv = self.f_px * P[:, :2] / z[:, None] + self.res / 2
        return uv, z

    def normalised(self, uv: np.ndarray) -> np.ndarray:
        return (uv - self.res / 2) / self.f_px


def car_top_corners() -> np.ndarray:
    """Corners of the car's top face in the car frame (4 x 3)."""
    lx, ly, lz = (s / 2 for s in cfg.CAR_SIZE)
    return np.array([[lx, ly, lz], [-lx, ly, lz], [-lx, -ly, lz], [lx, -ly, lz]])


def interaction_matrix(xy: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Image Jacobian for normalised point features (2n x 6), camera-frame twist."""
    rows = []
    for (x, y), z in zip(xy, Z):
        rows.append([-1 / z, 0, x / z, x * y, -(1 + x * x), y])
        rows.append([0, -1 / z, y / z, 1 + y * y, -x * y, -x])
    return np.array(rows)


class CarServo:
    """IBVS controller that aligns a camera above a car.

    T_car_cam_goal is the desired camera pose in the car frame. The desired image
    is rendered once from the car model, so the loop never looks at the car's true pose.
    """

    def __init__(self, camera: Camera, T_car_cam_goal: SE3, gain: float = lc.IBVS_GAIN):
        self.camera = camera
        self.gain = gain
        self.corners = car_top_corners()
        uv_goal, _ = camera.project(SE3(), (np.linalg.inv(T_car_cam_goal.A) @ np.c_[self.corners, np.ones(4)].T).T[:, :3])
        self.s_goal = camera.normalised(uv_goal)

    def measure(self, T_world_cam: SE3, T_world_car: SE3):
        """What the camera sees: normalised features and depths."""
        pts = (np.asarray(T_world_car.A) @ np.c_[self.corners, np.ones(4)].T).T[:, :3]
        uv, z = self.camera.project(T_world_cam, pts)
        return self.camera.normalised(uv), z

    def twist(self, T_world_cam: SE3, T_world_car: SE3, scale: float = 1.0):
        """Camera-frame velocity command and the rms feature error in pixels."""
        s, z = self.measure(T_world_cam, T_world_car)
        err = (s - self.s_goal).reshape(-1)
        L = interaction_matrix(s, z)
        v = -self.gain * scale * np.linalg.pinv(L) @ err
        err_px = float(np.sqrt(np.mean(err**2)) * self.camera.f_px)
        return v, err_px


def limit_twist(v: np.ndarray, max_lin: float = lc.IBVS_MAX_LIN, max_ang: float = lc.IBVS_MAX_ANG) -> np.ndarray:
    """Scale a 6-vector twist so its linear and angular parts respect their limits."""
    v = np.asarray(v, dtype=float).copy()
    k = max(1.0, np.linalg.norm(v[:3]) / max_lin, np.linalg.norm(v[3:]) / max_ang)
    return v / k


def camera_twist_to_tool_twist(v_cam: np.ndarray, T_world_cam: SE3, tool_point_world: np.ndarray) -> np.ndarray:
    """Convert a camera-frame twist to a world-frame twist of the tool point.

    The camera and tool are one rigid body, so the tool point moves with
    v + w x r where r runs from the camera origin to the tool point.
    """
    R = np.asarray(T_world_cam.R)
    v_w, w_w = R @ v_cam[:3], R @ v_cam[3:]
    r = np.asarray(tool_point_world) - np.asarray(T_world_cam.t)
    return np.r_[v_w + np.cross(w_w, r), w_w]
