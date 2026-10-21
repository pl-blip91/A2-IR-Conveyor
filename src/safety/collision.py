"""Collision envelope, obstacle and path prediction for the MH5 loading station.

The arm is approximated by capsules (a line segment with a radius) laid along its
links. A simulated obstacle is a sphere. Planned joint paths are sampled and checked
before the arm reaches them, so the station can stop or re-plan ahead of a collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# Radius of each capsule: base column, 5 arm links, then the gripper (flange to tool tip).
LINK_RADII = np.array([0.09, 0.075, 0.065, 0.06, 0.055, 0.055, 0.075])
_HIDDEN = np.array([0.0, 0.0, -10.0])


@dataclass
class Obstacle:
    """A controlled simulated object that can be placed in a planned path."""

    radius: float = 0.05
    centre: np.ndarray = field(default_factory=lambda: _HIDDEN.copy())
    active: bool = False

    def place(self, xyz: Sequence[float]) -> None:
        self.centre = np.asarray(xyz, dtype=float)
        self.active = True

    def remove(self) -> None:
        self.centre = _HIDDEN.copy()
        self.active = False


@dataclass(frozen=True)
class Box:
    """Axis-aligned static keep-out volume (world frame)."""

    lo: tuple
    hi: tuple


class ArmEnvelope:
    """Forward kinematics of a standard-DH arm and its capsule collision envelope."""

    def __init__(self, links, tool: np.ndarray, radii: np.ndarray = LINK_RADII):
        self._dh = [(float(link.d), float(link.a), float(link.alpha), float(link.offset)) for link in links]
        self._tool = np.asarray(tool, dtype=float)
        self.radii = np.asarray(radii, dtype=float)

    def points(self, q: Sequence[float], base: np.ndarray) -> np.ndarray:
        """World positions of: arm base, joint 1..6 origins, tool tip (8 x 3)."""
        T = np.asarray(base, dtype=float)
        pts = [T[:3, 3].copy()]
        for (d, a, alpha, offset), qi in zip(self._dh, q):
            th = qi + offset
            ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(alpha), np.sin(alpha)
            A = np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d],
                          [0.0, 0.0, 0.0, 1.0]])
            T = T @ A
            pts.append(T[:3, 3].copy())
        pts.append((T @ self._tool)[:3, 3])
        return np.array(pts)


def segment_point_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from point c to each segment a[i]-b[i] (arrays of shape n x 3)."""
    ab = b - a
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", c - a, ab) / denom, 0.0, 1.0)
    return np.linalg.norm(a + ab * t[:, None] - c, axis=1)


def _box_distance(points: np.ndarray, box: Box) -> np.ndarray:
    lo, hi = np.asarray(box.lo), np.asarray(box.hi)
    d = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.linalg.norm(d, axis=-1)


class CollisionChecker:
    """Checks arm configurations and joint paths against obstacles and static boxes."""

    def __init__(self, envelope: ArmEnvelope, base_getter: Callable[[], np.ndarray],
                 obstacles: Sequence[Obstacle], statics: Sequence[Box] = (), margin: float = 0.025):
        self.envelope = envelope
        self._base = base_getter
        self.obstacles = list(obstacles)
        self.statics = list(statics)
        self.margin = margin

    def config_hit(self, q: Sequence[float], with_statics: bool = True) -> Optional[str]:
        """Return a description of what q hits, or None if it is clear."""
        pts = self.envelope.points(q, self._base())
        a, b, r = pts[:-1], pts[1:], self.envelope.radii
        for ob in self.obstacles:
            if ob.active:
                d = segment_point_distance(a, b, ob.centre) - r - ob.radius - self.margin
                if np.any(d < 0):
                    return "obstacle"
        if with_statics and self.statics:
            # Skip the base column and the gripper: they legitimately touch the mounting
            # plate and the pick/place surfaces.
            for i in range(1, len(r) - 1):
                samples = a[i] + (b[i] - a[i]) * np.linspace(0, 1, 5)[:, None]
                for box in self.statics:
                    if np.any(_box_distance(samples, box) < r[i]):
                        return "workcell structure"
        return None

    def path_hit(self, qs: np.ndarray, stride: int = 4, with_statics: bool = True) -> Optional[int]:
        """Index of the first sampled configuration in qs that collides, else None."""
        if len(qs) == 0:
            return None
        idx = list(range(0, len(qs), stride))
        if idx[-1] != len(qs) - 1:
            idx.append(len(qs) - 1)
        for i in idx:
            if self.config_hit(qs[i], with_statics):
                return i
        return None

    def tool_line_hit(self, p0: np.ndarray, p1: np.ndarray, gripper_radius: float = 0.075) -> bool:
        """True if the straight tool path p0 -> p1 passes through an obstacle."""
        for ob in self.obstacles:
            if ob.active:
                d = segment_point_distance(p0[None], p1[None], ob.centre)[0]
                if d < gripper_radius + ob.radius + self.margin:
                    return True
        return False


def plan_around(q0: np.ndarray, q_goal: np.ndarray, free: Callable[[np.ndarray, np.ndarray], bool],
                lo: np.ndarray, hi: np.ndarray, tries: int = 300, seed: int = 7) -> Optional[list]:
    """Find via-points so that q0 -> via -> q_goal is free. Returns [] if direct is free.

    Tries structured sideways/up swings first (joints 1-3), then random via-points
    around the midpoint. Returns None if nothing works.
    """
    if free(q0, q_goal):
        return []
    mid = 0.5 * (q0 + q_goal)
    candidates = []
    for d1 in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
        for d2 in (0.0, -0.4, 0.4, -0.8):
            for d3 in (0.0, -0.4, 0.4):
                v = mid.copy()
                v[0] += d1
                v[1] += d2
                v[2] += d3
                candidates.append(v)
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        candidates.append(mid + rng.normal(0.0, 0.7, size=mid.shape))
    for v in candidates:
        v = np.clip(v, lo, hi)
        if free(q0, v) and free(v, q_goal):
            return [v]
    return None
