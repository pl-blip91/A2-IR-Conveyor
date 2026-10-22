"""
UR3e DH (standard: Rz(theta) Tz(d) Tx(a) Rx(alpha)), from the manufacturer data sheet:  d, a, alpha
    1 | 0.15185 |  0       | +pi/2        4 | 0.13105 | 0 | +pi/2
    2 | 0       | -0.24355 |  0           5 | 0.08535 | 0 | -pi/2
    3 | 0       | -0.21320 |  0           6 | 0.09210 | 0 |  0
"""

import os
from math import pi

import numpy as np
from roboticstoolbox import DHLink, DHRobot
from spatialgeometry import Mesh
from spatialmath import SE3
from spatialmath.base import transl

HERE = os.path.dirname(os.path.abspath(__file__))

UR_D = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
UR_A = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
UR_ALPHA = [pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0]
UR_MESH_NAMES = ["shoulder.stl", "upperarm.stl", "forearm.stl", "wrist1.stl", "wrist2.stl", "wrist3.stl"]
UR_BASE_MESH = "base.stl"
UR_MESH_DIR = os.path.join(HERE, "meshes", "ur3e")
UR_MESH_SCALE = 1.0  # the UR3e STL files are in metres
# Each STL sits in its own link frame (the manufacturer's URDF convention). Pose of the STL inside that link frame:
UR_VISUAL = [SE3.Rz(pi),  # shoulder
             SE3(0, 0, 0.12) * SE3.Rz(-pi / 2) * SE3.Rx(pi / 2),  # upper arm
             SE3(0, 0, 0.027) * SE3.Rz(-pi / 2) * SE3.Rx(pi / 2),  # forearm
             SE3(0, 0, -0.104) * SE3.Rx(pi / 2),  # wrist 1
             SE3(0, 0, -0.08535),  # wrist 2
             SE3(0, 0, -0.0921) * SE3.Rx(pi / 2)]  # wrist 3 (flange end)
UR_BASE_VISUAL = SE3.Rz(pi)
UR_BLUE = [0.62, 0.76, 0.90, 1.0]
UR_DARK = [0.30, 0.32, 0.35, 1.0]
UR_BASE_GREY = [0.30, 0.32, 0.35, 1.0]
UR_LINK_COLOURS = [UR_BLUE, UR_BLUE, UR_BLUE, UR_BLUE, UR_BLUE, UR_DARK]

UR_TOOL_LEN = 0.207  # OnRobot RG2 TCP beyond the UR3e flange along tool z
GRIP_OPEN, GRIP_CLOSED = 0.090, 0.064  # gap between the pads (m)

# OnRobot RG2 meshes
RG2_HAND_FILE = "rg2_hand.stl"
RG2_FINGER_FILES = ("rg2_finger_left.stl", "rg2_finger_right.stl")
RG2_MESH_DIR = os.path.join(HERE, "meshes", "rg2")
RG2_MESH_SCALE = 0.001  # the RG2 STL files are in millimetres
RG2_MESH_OFFSET = (SE3.Rz(pi / 2) * SE3(0, 0.009, 0.055)).A  # pose of the STL frame in the gripper base frame
RG2_MESH_GAP = 0.0472  # gap between the pads as modelled in the finger STLs (m)
RG2_BODY_COLOUR = (0.82, 0.84, 0.87, 1.0)
RG2_FINGER_COLOUR = (0.12, 0.12, 0.13, 1.0)


class UR3eRobot(DHRobot):
    """DHRobot (Lab 2 Q4 / Lab 3 Q2 pattern). Meshes follow the link frames every frame (Lab 4 Q1 pen_mesh.T = fkine...)."""

    def __init__(self):
        links = [DHLink(d=UR_D[i], a=UR_A[i], alpha=UR_ALPHA[i], offset=0, qlim=[-2 * pi, 2 * pi]) for i in range(6)]
        super().__init__(links, name="UR3e", manufacturer="Universal Robots")
        self.q = np.zeros(6)
        self.meshes, self.offsets = [], []

    def attach_meshes(self, env, mesh_dir):
        """One Mesh per link. The STL sits in the manufacturer link frame, which is the DH frame i with its Tx(a) Rx(alpha)
        taken back off, so the fixed offset is  Rx(-alpha) * Tx(-a) * (pose of the STL in its link frame)."""
        for i in range(6):
            offset = SE3.Rx(-UR_ALPHA[i]) * SE3(-UR_A[i], 0, 0) * UR_VISUAL[i]
            self.offsets.append(offset.A)
            mesh = Mesh(filename=os.path.join(mesh_dir, UR_MESH_NAMES[i]), scale=[UR_MESH_SCALE] * 3,
                        color=UR_LINK_COLOURS[i])
            env.add(mesh)
            self.meshes.append(mesh)

    def draw(self):
        frames = self.fkine_all(self.q).A  # Lab 5 Q2/3: link frames from fkine_all (the base is frames[0])
        for mesh, offset, T in zip(self.meshes, self.offsets, frames[1:]):
            mesh.T = T @ offset


def make_ur3e(env, base, q0):
    """Build the UR3e with its meshes at `base` (SE3), tool point at the RG2 grip centre, in joint state q0."""
    robot = UR3eRobot()
    missing = [n for n in [UR_BASE_MESH] + UR_MESH_NAMES if not os.path.isfile(os.path.join(UR_MESH_DIR, n))]
    if missing:
        raise FileNotFoundError(f"UR3e STL meshes missing from {UR_MESH_DIR}: {missing}")
    robot.attach_meshes(env, UR_MESH_DIR)
    env.add(Mesh(filename=os.path.join(UR_MESH_DIR, UR_BASE_MESH), scale=[UR_MESH_SCALE] * 3, color=UR_BASE_GREY,
                 pose=base * UR_BASE_VISUAL))
    robot.base = base
    robot.tool = SE3(0, 0, UR_TOOL_LEN)  # Lab 11: robot.tool - fkine / jacob0 / ikine now work at the tool point
    robot.q = np.asarray(q0, float).copy()
    return robot


class RG2Gripper:
    """OnRobot RG2 meshes following the UR3e TCP frame (Lab 4 Q1: mesh.T = fkine(q).A @ ...). The finger STLs slide."""

    def __init__(self, env, robot):
        self.robot, self.opening = robot, GRIP_OPEN
        self.hand, self.fingers = None, []
        hand_path = os.path.join(RG2_MESH_DIR, RG2_HAND_FILE)
        if not os.path.isfile(hand_path):
            print(f"RG2 meshes not found in {RG2_MESH_DIR} - the UR3e is drawn without its gripper.")
            return
        self.hand = Mesh(filename=hand_path, scale=[RG2_MESH_SCALE] * 3, color=RG2_BODY_COLOUR)
        env.add(self.hand)
        for name in RG2_FINGER_FILES:
            path = os.path.join(RG2_MESH_DIR, name)
            if os.path.isfile(path):
                finger = Mesh(filename=path, scale=[RG2_MESH_SCALE] * 3, color=RG2_FINGER_COLOUR)
                env.add(finger)
                self.fingers.append(finger)

    def update(self):
        if self.hand is None:
            return
        G = self.robot.fkine(self.robot.q).A @ transl(0, 0, -UR_TOOL_LEN)  # gripper base frame = the tool flange
        self.hand.T = G @ RG2_MESH_OFFSET
        for k, finger in enumerate(self.fingers):
            shift = (1 if k == 0 else -1) * (self.opening - RG2_MESH_GAP) / 2
            finger.T = G @ transl(0, shift, 0) @ RG2_MESH_OFFSET
