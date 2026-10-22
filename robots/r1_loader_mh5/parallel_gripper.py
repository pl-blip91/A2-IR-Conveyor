from __future__ import annotations

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

TCP_OFFSET = 0.10  # flange to grip centre (m)
OPEN_WIDTH = 0.16  # gap between the fingers when fully open (m)
MAX_FINGER_SPEED = 0.10  # m/s of total opening/closing travel

_PALM_SIZE = (0.06, 0.14, 0.05)
_FINGER_SIZE = (0.04, 0.015, 0.07)
_FINGER_Z = 0.05 + _FINGER_SIZE[2] / 2  # fingers start where the palm ends
_COLOUR = (0.15, 0.15, 0.17, 1.0)


class ParallelGripper:
    """Palm plus two fingers that follow the arm's tool centre point."""

    def __init__(self):
        self.opening = OPEN_WIDTH
        self.palm = sg.Cuboid(scale=list(_PALM_SIZE), color=list(_COLOUR))
        self.fingers = [
            sg.Cuboid(scale=list(_FINGER_SIZE), color=[0.85, 0.55, 0.1, 1.0]) for _ in range(2)
        ]

    @property
    def shapes(self) -> list:
        return [self.palm, *self.fingers]

    def add_to_env(self, env) -> None:
        for shape in self.shapes:
            env.add(shape)

    def update(self, tcp: SE3, opening: float | None = None) -> None:
        """Place the gripper so its grip centre is at ``tcp`` (world frame)."""
        if opening is not None:
            self.opening = float(np.clip(opening, 0.0, OPEN_WIDTH))
        flange = tcp.A @ SE3(0, 0, -TCP_OFFSET).A  # tool0 frame
        self.palm.T = flange @ SE3(0, 0, _PALM_SIZE[2] / 2).A
        offset = self.opening / 2 + _FINGER_SIZE[1] / 2
        for finger, side in zip(self.fingers, (+1, -1)):
            finger.T = flange @ SE3(0, side * offset, _FINGER_Z).A
