"""
DH (standard: Rz(theta) Tz(d) Tx(a) Rx(alpha)):  d, a, alpha, offset
    1 | 0.450 | 0.150 | -pi/2 | 0         4 | 0.640 | 0 | +pi/2 | 0
    2 | 0     | 0.600 |  0    | -pi/2     5 | 0     | 0 | -pi/2 | 0
    3 | 0     | 0.200 | -pi/2 | 0         6 | 0.100 | 0 |  0    | pi
"""

import os
from math import pi

import numpy as np
from roboticstoolbox import DHLink, DHRobot
from scipy import linalg
from spatialgeometry import Mesh
from spatialmath.base import transl

MESH_NAMES = [f"link_{i}.stl" for i in range(1, 7)]
FANUC_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
FANUC_MESH_SCALE = 1.0  # 0.001 if the STL files are in millimetres
URDF_LIMITS_RAD = [(-3.14, 3.14), (-1.57, 2.79), (-3.14, 4.61), (-3.31, 3.31), (-3.31, 3.31), (-6.28, 6.28)]
URDF_AXIS_SIGN = [1, 1, -1, -1, -1, -1]  # DH q = sign * URDF q
URDF_ZERO_POS = [(0, 0, 0.45), (0.15, 0, 0.45), (0.15, 0, 1.05), (0.15, 0, 1.25), (0.79, 0, 1.25), (0.89, 0, 1.25)]
FANUC_YELLOW = [1.0, 0.84, 0.0, 1.0]
FANUC_DARK = [0.48, 0.47, 0.47, 1.0]
BASE_GREY = [0.30, 0.30, 0.30, 1.0]
LINK_COLOURS = [FANUC_YELLOW] * 5 + [FANUC_DARK]


class FanucM10iA(DHRobot):
    """DHRobot (Lab 2 Q4 / Lab 3 Q2 pattern). Meshes follow the link frames every frame (Lab 4 Q1 pen_mesh.T = fkine...)."""

    def __init__(self):
        lim = [[lo, hi] if s > 0 else [-hi, -lo] for (lo, hi), s in zip(URDF_LIMITS_RAD, URDF_AXIS_SIGN)]
        links = [
            DHLink(d=0.450, a=0.150, alpha=-pi / 2, offset=0, qlim=lim[0]),
            DHLink(d=0.0, a=0.600, alpha=0, offset=-pi / 2, qlim=lim[1]),
            DHLink(d=0.0, a=0.200, alpha=-pi / 2, offset=0, qlim=lim[2]),
            DHLink(d=0.640, a=0.0, alpha=pi / 2, offset=0, qlim=lim[3]),
            DHLink(d=0.0, a=0.0, alpha=-pi / 2, offset=0, qlim=lim[4]),
            DHLink(d=0.100, a=0.0, alpha=0, offset=pi, qlim=lim[5]),
        ]
        super().__init__(links, name="FANUC M-10iA", manufacturer="FANUC")
        self.q = np.zeros(6)
        self.meshes, self.offsets = [], []

    def attach_meshes(self, env, mesh_dir):
        """Create one Mesh per link and remember its offset from the DH frame (computed at q = 0, base = identity)."""
        frames = self.fkine_all(np.zeros(6)).A  # Lab 5 Q2/3: link frames from fkine_all
        for i in range(6):
            self.offsets.append(linalg.inv(frames[i + 1]) @ transl(*URDF_ZERO_POS[i]))
            mesh = Mesh(filename=os.path.join(mesh_dir, MESH_NAMES[i]), scale=[FANUC_MESH_SCALE] * 3, color=LINK_COLOURS[i])
            env.add(mesh)
            self.meshes.append(mesh)

    def draw(self):
        frames = self.fkine_all(self.q).A
        for mesh, offset, T in zip(self.meshes, self.offsets, frames[1:]):
            mesh.T = T @ offset


def make_fanuc(env, base, q0):
    """Build the FANUC with its meshes, stand it at `base` (SE3) in joint state q0."""
    robot = FanucM10iA()
    missing = [n for n in MESH_NAMES + ["base_link.stl"] if not os.path.isfile(os.path.join(FANUC_MESH_DIR, n))]
    if missing:
        raise FileNotFoundError(f"FANUC STL meshes missing from {FANUC_MESH_DIR}: {missing}")
    robot.attach_meshes(env, FANUC_MESH_DIR)  # before the base is set (offsets use base = identity)
    env.add(Mesh(filename=os.path.join(FANUC_MESH_DIR, "base_link.stl"), scale=[FANUC_MESH_SCALE] * 3,
                 color=BASE_GREY, pose=base))
    robot.base = base
    robot.q = np.asarray(q0, float).copy()
    return robot
