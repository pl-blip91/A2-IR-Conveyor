#Yaskawa Motoman MH5 - parameter-based (standard DH) model with STL meshes

from __future__ import annotations

import os
from math import pi, radians

import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialmath import SE3
from spatialmath.base import transl

MESH_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes", "mh5")

# Joint limits in degrees, from the URDF.
JOINT_LIMITS_DEG = (
    (-170, 170),
    (-65, 150),
    (-70, 190),
    (-190, 190),
    (-135, 135),
    (-360, 360),
)

# Joint 1 .. joint 6 origins when every joint is zero, expressed in the world
# (robot base) frame. Each STL mesh is defined in its own URDF link frame, which
# at q = 0 is a pure translation of the base frame.
_LINK_ORIGINS_AT_ZERO = (
    (0.0, 0.0, 0.0),      # base_link
    (0.0, 0.0, 0.330),    # link_1_s
    (0.088, 0.0, 0.330),  # link_2_l
    (0.088, 0.0, 0.640),  # link_3_u
    (0.088, 0.0, 0.680),  # link_4_r
    (0.393, 0.0, 0.680),  # link_5_b
    (0.473, 0.0, 0.680),  # link_6_t
)

_MESH_FILES = ("base_link", "link_1_s", "link_2_l", "link_3_u", "link_4_r", "link_5_b", "link_6_t")

_BASE_COLOUR = (0.12, 0.12, 0.14, 1.0)
_ARM_COLOUR = (0.05, 0.32, 0.72, 1.0)  # Yaskawa-style blue
_WRIST_COLOUR = (0.75, 0.76, 0.78, 1.0)


class YaskawaMH5(DHRobot3D):

    def __init__(self, base: SE3 | np.ndarray | None = None, tool_offset: float = 0.0):
        links = [
            rtb.RevoluteDH(d=0.330, a=0.088, alpha=-pi / 2, qlim=self._qlim(0)),
            rtb.RevoluteDH(d=0.0, a=0.310, alpha=pi, offset=-pi / 2, qlim=self._qlim(1)),
            rtb.RevoluteDH(d=0.0, a=0.040, alpha=-pi / 2, qlim=self._qlim(2)),
            rtb.RevoluteDH(d=-0.305, a=0.0, alpha=pi / 2, qlim=self._qlim(3)),
            rtb.RevoluteDH(d=0.0, a=0.0, alpha=-pi / 2, qlim=self._qlim(4)),
            rtb.RevoluteDH(d=-0.080, a=0.0, alpha=0.0, qlim=self._qlim(5)),
        ]

        link3d_names = {f"link{i}": stem for i, stem in enumerate(_MESH_FILES)}
        colours = [_BASE_COLOUR] + [_ARM_COLOUR] * 4 + [_WRIST_COLOUR, _WRIST_COLOUR]
        for i, colour in enumerate(colours):
            link3d_names[f"color{i}"] = colour

        qtest = [0.0] * 6
        qtest_transforms = [transl(*origin) for origin in _LINK_ORIGINS_AT_ZERO]

        super().__init__(
            links,
            link3d_names,
            link3d_dir=MESH_DIR,
            name="YaskawaMH5",
            qtest=qtest,
            qtest_transforms=qtest_transforms,
        )

        self.tool = SE3.Rx(pi) * SE3(0, 0, tool_offset)
        self.q = np.zeros(6)
        if base is not None:
            self.base = base if isinstance(base, SE3) else SE3(np.asarray(base, float), check=False)

    @staticmethod
    def _qlim(joint: int) -> list[float]:
        lo, hi = JOINT_LIMITS_DEG[joint]
        return [radians(lo), radians(hi)]

    def __setattr__(self, name, value):
        # DHRobot3D refreshes the meshes before the new q/base is stored, which
        # leaves the picture one assignment behind. Refresh again afterwards.
        super().__setattr__(name, value)
        if name in ("q", "base") and hasattr(self, "links_3d"):
            self._update_3dmodel()

    def kinematics_only(self) -> rtb.DHRobot:
  
        return rtb.DHRobot(self.links, tool=self.tool, name="YaskawaMH5 (kinematics)")

    @property
    def qlim_array(self) -> np.ndarray:
        #Joint limits as a (2, 6) array in radians (row 0 lower, row 1 upper).
        return np.array([[radians(lo) for lo, _ in JOINT_LIMITS_DEG],
                         [radians(hi) for _, hi in JOINT_LIMITS_DEG]])
