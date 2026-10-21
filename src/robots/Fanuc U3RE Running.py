"""
Toy Car Production workcell (Swift) - FANUC M-10iA stamping station + UR3e sorting station.
Student: 25556123    


"""
import os
import threading
import time
from itertools import combinations
from math import pi

import keyboard
import numpy as np
import pygame
import swift
from scipy import linalg
from spatialgeometry import Cuboid, Cylinder, Mesh, Sphere
from spatialmath import SE3
from spatialmath.base import transl
from roboticstoolbox import DHLink, DHRobot, jtraj, trapezoidal
from ir_support import RectangularPrism, line_plane_intersection


HERE = os.path.dirname(os.path.abspath(__file__))  # Lab 4 Q1: folder of this script

# =============================================================================
# 1. LAYOUT CONSTANTS (metres; +x along the conveyor, +y across it, +z up, floor z = 0)
# =============================================================================
BELT_LENGTH, BELT_WIDTH, BELT_HEIGHT = 6.4, 0.40, 0.60
BELT_COLOUR = (0.15, 0.15, 0.17, 1.0)
CAR_SLIDE_SPEED = 0.50  # m/s cruise speed when a car is slid along the belt (trapezoidal blend)

PAD_SIZE = (0.5, 0.5, 0.6)  # the top of the pad is where the robot base sits
PAD_HEIGHT = PAD_SIZE[2]
PAD_OFFSET_Y = 0.5
PAD_COLOUR = (0.55, 0.55, 0.58, 1.0)

STATION_X = 2 * BELT_LENGTH / 3  # FANUC (label station) - where the car stops to be stamped
STATION_SIDE = +1  # FANUC pad is on the +y side of the belt
ROBOT_BASE = SE3(STATION_X, STATION_SIDE * PAD_OFFSET_Y, PAD_HEIGHT) * SE3.Rz(-STATION_SIDE * pi / 2)
SORT_X, SORT_SIDE = BELT_LENGTH, -1  # UR3e (sorting station) at the end of the belt, on the -y side

CAR_SIZE = (0.15, 0.075, 0.042)  # length, width, height
UNPAINTED = (0.7, 0.7, 0.7, 1.0)
CAR_COLOURS = {"red": (0.85, 0.12, 0.12, 1.0), "green": (0.12, 0.65, 0.22, 1.0), "blue": (0.15, 0.35, 0.90, 1.0)}
CAR_PITCH = 0.25  # distance between neighbouring cars at the start (m)
# Cars queue at the start of the belt. The one furthest along goes first: blue, then red, then green.
CARS = tuple((CAR_SIZE[0] / 2 + i * CAR_PITCH, colour) for i, colour in enumerate(("green", "red", "blue")))

# Stamping
TOOL_LEN = 0.10  # stamp die length beyond the FANUC flange (m)
STAMP_HOVER = 0.05
STAMP_DWELL = 0.3
PLUNGE_SPEED = 0.10  # m/s
DT = 0.03  # simulation step (s)
FANUC_STEP = np.deg2rad(1.5)  # largest joint step per frame in FANUC joint moves (Lab 5 interpolate_waypoints_radians)

LETTER_H = 0.038
LETTER_W = {"I": 0.024, "R": 0.030}
LETTER_GAP = 0.012
STROKE_W = 0.005
STROKE_COLOUR = (0.04, 0.04, 0.07, 1.0)
TEXT = "IR"
TEXT_W = sum(LETTER_W[c] for c in TEXT) + LETTER_GAP * (len(TEXT) - 1)

# Safety scenery (all built from Cuboid / Cylinder as in Lab 3)
FENCE_COLOUR = (1.0, 0.45, 0.0, 1.0)
GATE_COLOUR = (1.0, 0.85, 0.0, 1.0)
FENCE_POST_GAP = 1.5
FENCE_Y = 1.5  # fence stands this far from the belt centre-line on both sides (m)
FENCE_END = 1.5  # length of fence between the door and the end of the cell (m)
DOOR_W = 1.06
OBSTACLE_Y = 0.12  # FANUC obstacle sits on the belt, to the side of the car

# Collision / singularity (Lab 5 Q2/3 collision check, Lab 9 Q1 DLS)
FANUC_EPS = 0.01  # manipulability below which the FANUC RMRC damping switches on
UR_EPS = 0.003  # same for the UR3e (it is small, so its manipulability values are small)
FANUC_OBS_SIZE, FANUC_OBS_MARGIN = 0.16, 0.05  # cube side and clearance kept around it (m)
UR_OBS_SIZE, UR_OBS_MARGIN = 0.10, 0.04

# =============================================================================
# 2. FANUC M-10iA (DH model from the manufacturer drawing / xacro; meshes are the external STL files)
# =============================================================================
# DH (standard: Rz(theta) Tz(d) Tx(a) Rx(alpha)):  d, a, alpha, offset
#     1 | 0.450 | 0.150 | -pi/2 | 0         4 | 0.640 | 0 | +pi/2 | 0
#     2 | 0     | 0.600 |  0    | -pi/2     5 | 0     | 0 | -pi/2 | 0
#     3 | 0     | 0.200 | -pi/2 | 0         6 | 0.100 | 0 |  0    | pi
MESH_NAMES = [f"link_{i}.stl" for i in range(1, 7)]
FANUC_MESH_DIR = os.path.join(HERE, "meshes")  # <- the folder that contains link_1.stl ... link_6.stl, base_link.stl
FANUC_MESH_SCALE = 1.0  # 0.001 if the STL files are in millimetres
URDF_LIMITS_RAD = [(-3.14, 3.14), (-1.57, 2.79), (-3.14, 4.61), (-3.31, 3.31), (-3.31, 3.31), (-6.28, 6.28)]
URDF_AXIS_SIGN = [1, 1, -1, -1, -1, -1]  # DH q = sign * URDF q
URDF_ZERO_POS = [(0, 0, 0.45), (0.15, 0, 0.45), (0.15, 0, 1.05), (0.15, 0, 1.25), (0.79, 0, 1.25), (0.89, 0, 1.25)]
FANUC_YELLOW = [1.0, 0.84, 0.0, 1.0]
FANUC_DARK = [0.48, 0.47, 0.47, 1.0]
BASE_GREY = [0.30, 0.30, 0.30, 1.0]
LINK_COLOURS = [FANUC_YELLOW] * 5 + [FANUC_DARK]
Q_HOME = np.deg2rad([0, 30, 30, 0, 60, 0])  # q5 = 0 is a wrist singularity, so home uses q5 = 60 deg
FANUC_SEEDS = [np.deg2rad(g) for g in ([0, 30, 30, 0, 60, 0], [0, 45, 0, 0, 45, 0], [0, 20, 60, 0, 40, 0],
                                       [0, 60, -20, 0, 50, 0], [90, 30, 30, 0, 60, 0], [-90, 30, 30, 0, 60, 0],
                                       [60, 30, 30, 0, 60, 0], [-60, 30, 30, 0, 60, 0])]
TOOL_DOWN = SE3.Rx(pi).R  # Lab 4 Q1: a rotation of pi about x turns the tool z axis downward


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


def make_fanuc(env):
    robot = FanucM10iA()
    print(f"Assessed robot model (student {STUDENT_NUMBER}): FANUC M-10iA DH table\n{robot}")
    missing = [n for n in MESH_NAMES + ["base_link.stl"] if not os.path.isfile(os.path.join(FANUC_MESH_DIR, n))]
    if missing:
        raise FileNotFoundError(f"FANUC STL meshes missing from {FANUC_MESH_DIR}: {missing}")
    robot.attach_meshes(env, FANUC_MESH_DIR)  # before the base is set (offsets use base = identity)
    env.add(Mesh(filename=os.path.join(FANUC_MESH_DIR, "base_link.stl"), scale=[FANUC_MESH_SCALE] * 3,
                 color=BASE_GREY, pose=ROBOT_BASE))
    robot.base = ROBOT_BASE
    robot.q = Q_HOME.copy()
    return robot


# =============================================================================
# 3. SCENE: CONVEYOR, CAR, STAMP TOOL, SAFETY EQUIPMENT (Lab 3 Cuboid / Cylinder, Lab 4 .T)
# =============================================================================
def _box(env, size, centre, colour):
    shape = Cuboid(scale=list(size), pose=SE3(*centre), color=list(colour))
    env.add(shape)
    return shape


class ToyCar:
    """Toy car built from simple shapes. `.T` moves the whole car."""

    def __init__(self, env, x, rgba):
        L, W, H = CAR_SIZE
        glass, dark, rubber = (0.08, 0.12, 0.20, 1.0), (0.10, 0.10, 0.12, 1.0), (0.06, 0.06, 0.06, 1.0)
        specs = [
            (Cuboid(scale=[L, 0.064, 0.016], color=list(rgba)), SE3(0, 0, 0.018)),
            (Cuboid(scale=[0.085, 0.054, 0.012], color=list(glass)), SE3(0, 0, 0.032)),
            (Cuboid(scale=[0.080, 0.052, 0.004], color=list(rgba)), SE3(0, 0, 0.040)),
            (Cuboid(scale=[0.016, 0.068, 0.003], color=list(dark)), SE3(-0.066, 0, 0.038)),
            (Cuboid(scale=[0.008, 0.012, 0.012], color=list(dark)), SE3(-0.066, 0, 0.032)),
            (Cuboid(scale=[0.004, 0.014, 0.006], color=[1.0, 0.9, 0.4, 1.0]), SE3(0.074, 0.020, 0.020)),
            (Cuboid(scale=[0.004, 0.014, 0.006], color=[1.0, 0.9, 0.4, 1.0]), SE3(0.074, -0.020, 0.020)),
            (Cuboid(scale=[0.004, 0.050, 0.005], color=[0.9, 0.1, 0.1, 1.0]), SE3(-0.074, 0, 0.022)),
        ]
        wheel_y = W / 2 - 0.006
        for wx in (0.047, -0.047):
            for wy in (wheel_y, -wheel_y):
                specs.append((Cylinder(radius=0.014, length=0.012, color=list(rubber)), SE3(wx, wy, 0.014) * SE3.Rx(pi / 2)))
        self.parts = []
        for shape, rel in specs:
            env.add(shape)
            self.parts.append((shape, rel))
        self._T = np.eye(4)
        self.T = transl(x, 0, BELT_HEIGHT)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = np.array(value, dtype=float)
        base = SE3(self._T)
        for shape, rel in self.parts:
            shape.T = (base * rel).A


class StampTool:
    """Stamp die that follows the FANUC flange (Lab 3 Q3 / Lab 5 Q1: shape.T = fkine(q) * SE3(0, 0, d))."""

    def __init__(self, env, robot):
        self.robot = robot
        self.body = Cylinder(radius=0.02, length=TOOL_LEN, color=(0.25, 0.25, 0.28, 1.0))
        self.plate = Cuboid(scale=[TEXT_W + 0.01, LETTER_H + 0.01, 0.008], color=(0.08, 0.08, 0.10, 1.0))
        env.add(self.body)
        env.add(self.plate)
        self.update()

    def update(self):
        T = self.robot.fkine(self.robot.q).A
        self.body.T = T @ transl(0, 0, TOOL_LEN / 2)
        self.plate.T = T @ transl(0, 0, TOOL_LEN - 0.004)


def letter_strokes(stamp_x):
    """World-space (x, y) polylines that make up TEXT, centred on the car roof."""
    H = LETTER_H
    shapes = {
        "I": lambda w: [[(0, H), (w, H)], [(w / 2, H), (w / 2, 0)], [(0, 0), (w, 0)]],
        "R": lambda w: [[(0, 0), (0, H), (0.6 * w, H), (w, 0.82 * H), (w, 0.68 * H), (0.6 * w, 0.5 * H), (0, 0.5 * H)],
                        [(0.4 * w, 0.5 * H), (w, 0)]],
    }
    x0, y0 = stamp_x - TEXT_W / 2, -H / 2
    strokes = []
    for c in TEXT:
        w = LETTER_W[c]
        for poly in shapes[c](w):
            strokes.append([(x0 + u, y0 + v) for u, v in poly])
        x0 += w + LETTER_GAP
    return strokes


def fence_run(env, a, b):
    """Straight run of orange fence (two rails + posts) from a=(x, y) to b=(x, y); runs are along x or along y."""
    (xa, ya), (xb, yb) = a, b
    length = float(np.hypot(xb - xa, yb - ya))
    along_x = abs(xb - xa) > abs(yb - ya)
    rail = (length, 0.04, 0.05) if along_x else (0.04, length, 0.05)
    for z in (0.35, 0.85):
        _box(env, rail, ((xa + xb) / 2, (ya + yb) / 2, z), FENCE_COLOUR)
    n = max(1, int(np.ceil(length / FENCE_POST_GAP)))
    for k in range(n + 1):
        _box(env, (0.06, 0.06, 1.0), (xa + (xb - xa) * k / n, ya + (yb - ya) * k / n, 0.5), FENCE_COLOUR)


def estop_station(env, x, y):
    """Red mushroom e-stop on a yellow box on a grey post (scenery)."""
    _box(env, (0.06, 0.06, 0.8), (x, y, 0.4), (0.4, 0.4, 0.42, 1.0))
    _box(env, (0.14, 0.14, 0.12), (x, y, 0.86), GATE_COLOUR)
    env.add(Cylinder(radius=0.05, length=0.04, pose=SE3(x, y, 0.94), color=[0.85, 0.05, 0.05, 1.0]))


def build_safety_cell(env):
    """Orange fence round the whole conveyor, ONE yellow door (UR3e side, next to the box), a red e-stop for each robot."""
    x1 = SORT_X + DOOR_W / 2 + FENCE_END  # the fence runs FENCE_END past the door ...
    x0 = BELT_LENGTH - x1  # ... and the same distance past the other end of the belt
    door_l, door_r = SORT_X - DOOR_W / 2, SORT_X + DOOR_W / 2
    for side in (+1, -1):
        y = side * FENCE_Y
        if side == SORT_SIDE:  # the door side: fence, gap for the door, fence
            fence_run(env, (x0, y), (door_l, y))
            fence_run(env, (door_r, y), (x1, y))
        else:
            fence_run(env, (x0, y), (x1, y))
    fence_run(env, (x0, -FENCE_Y), (x0, FENCE_Y))
    fence_run(env, (x1, -FENCE_Y), (x1, FENCE_Y))
    gy = SORT_SIDE * FENCE_Y  # the yellow door: two posts, a top bar and a middle bar
    for gx in (door_l, door_r):
        _box(env, (0.06, 0.06, 1.0), (gx, gy, 0.5), GATE_COLOUR)
    _box(env, (DOOR_W, 0.04, 0.06), (SORT_X, gy, 1.0), GATE_COLOUR)
    _box(env, (DOOR_W, 0.03, 0.04), (SORT_X, gy, 0.55), GATE_COLOUR)
    estop_station(env, STATION_X, STATION_SIDE * (FENCE_Y + 0.6))  # FANUC e-stop
    estop_station(env, door_l - 0.8, SORT_SIDE * (FENCE_Y + 0.6))  # UR3e e-stop (beside the door)


# =============================================================================
# 4. UR3e SORTING STATION CONSTANTS + STORAGE BOX
# =============================================================================
SORT_BASE_INSET = 0.07
SORT_BASE = (SE3(SORT_X, SORT_SIDE * (PAD_OFFSET_Y - PAD_SIZE[1] / 2 + SORT_BASE_INSET), PAD_HEIGHT)
             * SE3.Rz(-SORT_SIDE * pi / 2))  # +x of the robot base faces the belt
PICK_X = BELT_LENGTH - CAR_SIZE[0] / 2  # where the lead car stops at the end of the belt

UR_STEP = np.deg2rad(1.8)  # largest joint step per frame (~1.05 rad/s peak at DT = 0.03: UR default movej speed)
UR_LINE_SPEED = 0.25  # m/s straight-line RMRC (UR default movel speed)
UR_TOOL_LEN = 0.207  # OnRobot RG2 TCP beyond the UR3e flange along tool z
GRIP_OPEN, GRIP_CLOSED = 0.090, 0.064  # gap between the pads (m)
TCP_Z = 0.020
PICK_HOVER = 0.10
TRAVEL_Z = 0.80
PICK_DWELL = 0.5
LOOKAHEAD = 24  # frames of the planned path checked for the obstacle
RRT_TRIES = 40  # random waypoints tried when re-planning (Lab 5 Q2/3 primitive RRT)
RRT_SPREAD = np.deg2rad(60)  # random waypoints are drawn within this of the colliding waypoint
Q_UR_HOME = np.deg2rad([29, -76.2, -91.9, -101.8, 90, 29])
UR_SEEDS = [np.deg2rad(g) for g in ([0, -90, -90, -90, 90, 0], [0, -60, -100, -110, 90, 0], [0, -100, -60, -110, 90, 0],
                                    [90, -90, -90, -90, 90, 0], [-90, -90, -90, -90, 90, 0], [180, -90, -90, -90, 90, 0])]
UR_OBSTACLE_P = (PICK_X - 0.215, -0.10, BELT_HEIGHT + 0.20)  # in the belt-end -> tub transit
# Self-collision clearance radius (m) of each UR3e segment: base column, upper arm, forearm, wrist 1, wrist 2, wrist 3, gripper.
# Two non-neighbouring segments closer than the sum of their radii count as the arm going through itself.
UR_SELF_RADII = (0.07, 0.09, 0.06, 0.04, 0.04, 0.04, 0.05)

# OnRobot RG2 meshes (external STL files, folder `rg2_meshes` next to this script)
RG2_HAND_FILE = "rg2_hand.stl"
RG2_FINGER_FILES = ("rg2_finger_left.stl", "rg2_finger_right.stl")
RG2_MESH_DIR = os.path.join(HERE, "rg2_meshes")
RG2_MESH_SCALE = 0.001  # the RG2 STL files are in millimetres
RG2_MESH_OFFSET = (SE3.Rz(pi / 2) * SE3(0, 0.009, 0.055)).A  # pose of the STL frame in the gripper base frame
RG2_MESH_GAP = 0.0472  # gap between the pads as modelled in the finger STLs (m)
RG2_BODY_COLOUR = (0.82, 0.84, 0.87, 1.0)
RG2_FINGER_COLOUR = (0.12, 0.12, 0.13, 1.0)

# Storage box: ONE plywood box with three open compartments in a row, lettered B | R | G
BOX_CELL_X, BOX_INNER_Y = 0.13, 0.18
BOX_CAR_YAW = pi / 2  # the UR3e turns its wrist by this before dropping so the car's long axis lies along y
BOX_T, BOX_H = 0.006, 0.09
TUB_FLOOR = BOX_T
BOX_PITCH = BOX_CELL_X + BOX_T
BOX_CENTRE = (0.0, 0.52)  # (dx along the belt from the UR3e, distance from the belt centre-line) - the middle compartment
_VIEW_RIGHT = -SORT_SIDE
TUB_LAYOUT = {"blue": (BOX_CENTRE[0] - _VIEW_RIGHT * BOX_PITCH, BOX_CENTRE[1]),
              "red": (BOX_CENTRE[0], BOX_CENTRE[1]),
              "green": (BOX_CENTRE[0] + _VIEW_RIGHT * BOX_PITCH, BOX_CENTRE[1])}
TUB_XY = {k: (SORT_X + dx, SORT_SIDE * dy) for k, (dx, dy) in TUB_LAYOUT.items()}
PLY = (0.88, 0.74, 0.54, 1.0)
LETTER_COLOURS = {"B": (0.08, 0.12, 0.55, 1.0), "R": (0.85, 0.08, 0.08, 1.0), "G": (0.08, 0.50, 0.12, 1.0)}
LETTER_SEGMENTS = {  # block letters as line segments in a unit box (u across, v up)
    "B": [(0, 0, 0, 1), (0, 1, .75, 1), (.75, 1, .85, .9), (.85, .9, .85, .6), (.85, .6, .75, .5), (0, .5, .75, .5),
          (.75, .5, .85, .4), (.85, .4, .85, .1), (.85, .1, .75, 0), (0, 0, .75, 0)],
    "R": [(0, 0, 0, 1), (0, 1, .75, 1), (.75, 1, .85, .9), (.85, .9, .85, .6), (.85, .6, .75, .5), (0, .5, .75, .5),
          (.4, .5, .9, 0)],
    "G": [(.85, .85, .7, 1), (.7, 1, .2, 1), (.2, 1, 0, .8), (0, .8, 0, .2), (0, .2, .2, 0), (.2, 0, .7, 0),
          (.7, 0, .9, .2), (.9, .2, .9, .45), (.5, .45, .9, .45)],
}
BOX_LETTER_H, BOX_LETTER_W, BOX_LETTER_STROKE = 0.050, 0.036, 0.008
STEEL, STEEL_DARK = (0.62, 0.64, 0.67, 1.0), (0.45, 0.47, 0.50, 1.0)


def _segment(env, a, b, thick_y, stroke, colour, y):
    """A thin bar from world-plane point a=(x,z) to b=(x,z) at depth y (a Cuboid, as in Lab 3)."""
    (x0, z0), (x1, z1) = a, b
    length = float(np.hypot(x1 - x0, z1 - z0))
    phi = float(np.arctan2(-(z1 - z0), x1 - x0))
    env.add(Cuboid(scale=[length + stroke, thick_y, stroke], color=list(colour),
                   pose=SE3((x0 + x1) / 2, y, (z0 + z1) / 2) * SE3.Ry(phi)))


def build_storage_box(env):
    """Three-compartment plywood box (Cuboids) with coloured B / R / G letters on the front face."""
    cx, cy = SORT_X + BOX_CENTRE[0], SORT_SIDE * BOX_CENTRE[1]
    z0, t, h = PAD_HEIGHT, BOX_T, BOX_H
    L = 3 * BOX_CELL_X + 4 * t
    W = BOX_INNER_Y + 2 * t
    face_y = cy + SORT_SIDE * W / 2  # outside face that carries the letters (turned away from the robot)
    _box(env, (L, W, t), (cx, cy, z0 + t / 2), PLY)  # floor
    for sy in (-1, 1):  # front and back walls
        _box(env, (L, t, h), (cx, cy + sy * (BOX_INNER_Y / 2 + t / 2), z0 + h / 2), PLY)
    for sx in (-1, 1):  # end walls
        _box(env, (t, BOX_INNER_Y, h), (cx + sx * (L / 2 - t / 2), cy, z0 + h / 2), PLY)
    for x in (cx - BOX_PITCH / 2, cx + BOX_PITCH / 2):  # the two dividers
        _box(env, (t, BOX_INNER_Y, h), (x, cy, z0 + h / 2), PLY)
    for key, letter in (("blue", "B"), ("red", "R"), ("green", "G")):
        lx, zc = TUB_XY[key][0], z0 + h * 0.55
        for (u0, v0, u1, v1) in LETTER_SEGMENTS[letter]:
            pa = (lx + _VIEW_RIGHT * (u0 - 0.5) * BOX_LETTER_W, zc + (v0 - 0.5) * BOX_LETTER_H)
            pb = (lx + _VIEW_RIGHT * (u1 - 0.5) * BOX_LETTER_W, zc + (v1 - 0.5) * BOX_LETTER_H)
            _segment(env, pa, pb, 0.0016, BOX_LETTER_STROKE, LETTER_COLOURS[letter], face_y + SORT_SIDE * 0.0016)


def build_sorting_equipment(env):
    """Stainless-steel bench that carries the UR3e and the plywood box."""
    deck_y = SORT_SIDE * 0.525
    _box(env, (0.95, 0.55, PAD_HEIGHT), (SORT_X, deck_y, PAD_HEIGHT / 2), STEEL)
    _box(env, (0.97, 0.57, 0.012), (SORT_X, deck_y, PAD_HEIGHT - 0.006), STEEL_DARK)
    build_storage_box(env)


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


# =============================================================================
# 4b. UR3e (DH model from the manufacturer data sheet; meshes are the external STL files, same pattern as the FANUC)
# =============================================================================
# DH (standard: Rz(theta) Tz(d) Tx(a) Rx(alpha)):  d, a, alpha
#     1 | 0.15185 |  0       | +pi/2        4 | 0.13105 | 0 | +pi/2
#     2 | 0       | -0.24355 |  0           5 | 0.08535 | 0 | -pi/2
#     3 | 0       | -0.21320 |  0           6 | 0.09210 | 0 |  0
UR_D = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
UR_A = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
UR_ALPHA = [pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0]
UR_MESH_NAMES = ["shoulder.stl", "upperarm.stl", "forearm.stl", "wrist1.stl", "wrist2.stl", "wrist3.stl"]
UR_BASE_MESH = "base.stl"
UR_MESH_DIR = os.path.join(HERE, "ur3e_meshes")
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


def make_ur3e(env):
    robot = UR3eRobot()
    missing = [n for n in [UR_BASE_MESH] + UR_MESH_NAMES if not os.path.isfile(os.path.join(UR_MESH_DIR, n))]
    if missing:
        raise FileNotFoundError(f"UR3e STL meshes missing from {UR_MESH_DIR}: {missing}")
    robot.attach_meshes(env, UR_MESH_DIR)
    env.add(Mesh(filename=os.path.join(UR_MESH_DIR, UR_BASE_MESH), scale=[UR_MESH_SCALE] * 3, color=UR_BASE_GREY,
                 pose=SORT_BASE * UR_BASE_VISUAL))
    robot.base = SORT_BASE
    robot.tool = SE3(0, 0, UR_TOOL_LEN)  # Lab 11: robot.tool - fkine / jacob0 / ikine now work at the tool point
    robot.q = Q_UR_HOME.copy()
    return robot


# =============================================================================
# 5. COLLISION CHECKING - taken from Lab 5 Q2/3 (RectangularPrism + line_plane_intersection + is_collision)
# =============================================================================
class Obstacle:
    """A cube that can be dropped into the planned path (Lab 5 Q2/3: Cuboid + RectangularPrism(...).get_data())."""

    def __init__(self, env, size, margin):
        self.size, self.margin, self.active = size, margin, False
        self.vertices = self.faces = self.face_normals = None
        self.shape = Cuboid(scale=[size, size, size], color=[1.0, 0.45, 0.0, 1.0])
        env.add(self.shape)
        self.remove()

    def place(self, centre):
        big = self.size + 2 * self.margin  # collision prism = cube + safety clearance
        self.vertices, self.faces, self.face_normals = RectangularPrism(big, big, big, center=list(centre)).get_data()
        self.shape.T = transl(centre)
        self.active = True

    def remove(self):
        self.active = False
        self.shape.T = transl(0, 0, -5)  # parked below the floor


def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    w = intersect_p - triangle_verts[0, :]
    wu, wv = np.dot(w, u), np.dot(w, v)
    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False
    return True


def get_link_poses(robot, q=None, tip=0.0):
    """Lab 5 get_link_poses (fkine_all) plus one extra point: the end-effector (tool included) pushed on by `tip`."""
    q = robot.q if q is None else q
    tr = list(robot.fkine_all(q).A)
    tr.append(robot.fkine(q).A @ transl(0, 0, tip))
    return tr


def is_collision(robot, q_matrix, faces, vertex, face_normals, tip=0.0):
    """Lab 5 Q2/3 is_collision: True as soon as any link of any pose in q_matrix crosses a face of the obstacle prism."""
    for q in q_matrix:
        tr = get_link_poses(robot, q, tip)
        for i in range(len(tr) - 1):
            if np.linalg.norm(tr[i + 1][:3, 3] - tr[i][:3, 3]) < 1e-9:  # zero-length link (coincident frames)
                continue
            for j, face in enumerate(faces):
                vert_on_plane = vertex[face][0]
                intersect_p, check = line_plane_intersection(face_normals[j], vert_on_plane, tr[i][:3, 3], tr[i + 1][:3, 3])
                triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                if check == 1:
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertex[triangle]):
                            return True
    return False


def fine_interpolation(q1, q2, max_step_radians):
    """Lab 5 Q2/3: keep calling jtraj until all step sizes are smaller than a given max step size."""
    steps = 2
    while np.any(max_step_radians < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
        steps += 1
    return jtraj(q1, q2, steps).q


def interpolate_waypoints_radians(waypoint_radians, max_step_radians):
    """Lab 5 Q2/3: given a set of waypoints, finely interpolate them."""
    q_matrix = []
    for i in range(np.size(waypoint_radians, 0) - 1):
        for q in fine_interpolation(waypoint_radians[i], waypoint_radians[i + 1], max_step_radians):
            q_matrix.append(q)
    return q_matrix


class Arm:
    """A robot with its own obstacle, tip length (for the collision points), singularity threshold, joint step and status.
    `self_radii` (optional) switches on the self-collision check: one clearance radius per link segment (see below)."""

    def __init__(self, name, robot, obstacle, tip, epsilon, step, self_radii=None):
        self.name, self.robot, self.obstacle = name, robot, obstacle
        self.tip, self.epsilon, self.step = tip, epsilon, step
        self.self_radii = self_radii
        self.safe_q = None  # upright pose used as a via-point when a direct joint move would hit the arm itself
        self.blocked, self.note, self.replans = False, "idle", 0

    def hits(self, q_list):
        o = self.obstacle
        return o.active and len(q_list) > 0 and is_collision(self.robot, q_list, o.faces, o.vertices, o.face_normals, self.tip)

    def self_hits(self, q_list):
        """True if the arm would pass through ITSELF in any pose of q_list (UR3e only; FANUC has no radii so it is False)."""
        return self.self_radii is not None and any(self_collision(self.robot, q, self.tip, self.self_radii) for q in q_list)

    def unsafe(self, q_list):
        """Used by the planner: a path is unsafe if it hits the obstacle OR the arm itself."""
        return self.hits(q_list) or self.self_hits(q_list)


# ---------------- self-collision (Lab 5 get_link_poses + Lab 6 Q2 'distance < radius' test, plain numpy) ----------------
SELF_SAMPLES = 7  # points sampled along each link segment
SELF_WRIST_FROM = 3  # segment index where the wrist starts (base column, upper arm, forearm are segments 0, 1, 2)


def segment_points(p0, p1, n=SELF_SAMPLES):
    """n points evenly spaced along the line from p0 to p1 (the link is a line through consecutive joint frames, Lab 5)."""
    s = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1 - s) * np.asarray(p0, float) + s * np.asarray(p1, float)


def self_collision(robot, q, tip, radii):
    """Lab 5 get_link_poses gives the joint positions, so segment k runs from joint k to joint k+1 (the last segment is the
    tool). Two segments that are NOT neighbours collide when some sampled points are closer than the sum of their radii
    (the same 'distance below a radius' idea as Lab 6 Q2). Neighbouring segments always touch, so they are skipped."""
    tr = get_link_poses(robot, q, tip)
    pts = [t[:3, 3] for t in tr]
    segs = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    samples = [segment_points(a, b) for a, b in segs]
    for i in range(min(len(segs), SELF_WRIST_FROM)):  # segments from SELF_WRIST_FROM on are the wrist + gripper: they sit
        for j in range(i + 2, len(segs)):  # a few cm apart by design, so wrist-vs-wrist pairs are not tested
            if np.linalg.norm(segs[i][1] - segs[i][0]) < 1e-9 or np.linalg.norm(segs[j][1] - segs[j][0]) < 1e-9:
                continue  # zero-length link (coincident frames)
            d = np.linalg.norm(samples[i][:, None, :] - samples[j][None, :, :], axis=2).min()
            if d < radii[i] + radii[j]:
                return True
    return False


# =============================================================================
# 5b. SAFETY + OPERATION: constants, person / light curtain / stack light models, hardware e-stop input, e-stop latch
# =============================================================================
STUDENT_NUMBER = "25556123"
FANUC_FINAL_DEFAULT = np.deg2rad([35, 25, 35, 0, 70, 0])  # default final joint states (replaced at marking)
UR_FINAL_DEFAULT = np.deg2rad([-15, -80, -95, -95, 90, 0])
# Settings (these replace command-line options)
MANUAL_MODE = False  # True: START only enables motion; production waits for the RUN PRODUCTION button
HW_ESTOP_BUTTON = 0  # gamepad button index used as the physical e-stop input (Lab 11)
HW_ESTOP_INVERT = False  # True if the input reads 1 when the e-stop is healthy
JOG_SPEED = 0.10  # m/s for a Cartesian jog (RMRC)
MAX_QUEUED_JOGS = 5

CURTAIN_INSET = 0.15  # the light-curtain plane stands this far inside the door line (m)
CURTAIN_ABS_Y = FENCE_Y - CURTAIN_INSET
PERSON_R = 0.18  # body radius used for the beam-break test
PERSON_START = FENCE_Y + 1.6  # |y| of the person while outside the cell
PERSON_INSIDE = 1.0  # |y| of the person once inside the cell
PERSON_SPEED = 0.6  # m/s
WARN_ZONE = 1.0  # strip outside the fence (m) in which the cell is slowed
SLOW_SCALE = 0.5  # fraction of normal speed inside the warning strip
BEAM_Z = (0.25, 0.55, 0.85)
GREEN, RED, AMBER = (0.1, 0.9, 0.2, 1.0), (1.0, 0.05, 0.05, 1.0), (1.0, 0.65, 0.0, 1.0)
PARK = transl(0, 0, -5)  # shapes that are 'switched off' are parked below the floor (same trick as Obstacle.remove)

READY, RUNNING, STOPPED = "READY", "RUNNING", "STOPPED"


class Assembly:
    """Several shapes that move together: `.T` places the whole group (same pattern as ToyCar)."""

    def __init__(self, env, specs):
        self.parts = []
        for shape, rel in specs:
            env.add(shape)
            self.parts.append((shape, rel))
        self._T = np.eye(4)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = np.array(value, dtype=float)
        base = SE3(self._T)
        for shape, rel in self.parts:
            shape.T = (base * rel).A


class Person(Assembly):
    """Simulated person who can walk in through the door. Controlled from the GUI / key z; used as the unsafe-zone signal."""

    def __init__(self, env):
        super().__init__(env, [
            (Cylinder(radius=0.11, length=0.80, color=[0.10, 0.15, 0.45, 1.0]), SE3(0, 0, 0.40)),  # legs
            (Cylinder(radius=0.17, length=0.50, color=[0.95, 0.80, 0.05, 1.0]), SE3(0, 0, 1.05)),  # hi-vis torso
            (Sphere(radius=0.11, color=[0.90, 0.70, 0.55, 1.0]), SE3(0, 0, 1.42))])  # head
        self.abs_y, self.target = PERSON_START, PERSON_START
        self.T = SE3(SORT_X, SORT_SIDE * self.abs_y, 0).A

    def go_inside(self):
        self.target = PERSON_INSIDE

    def go_outside(self):
        self.target = PERSON_START

    def toggle(self):
        self.go_inside() if self.target >= PERSON_START else self.go_outside()

    def update(self, dt):
        if self.abs_y == self.target:
            return
        step = PERSON_SPEED * dt
        self.abs_y = float(self.abs_y + np.clip(self.target - self.abs_y, -step, step))
        self.T = SE3(SORT_X, SORT_SIDE * self.abs_y, 0).A

    def in_curtain(self):
        """True while any part of the person is at / beyond the light-curtain plane (beam broken or inside the cell)."""
        return self.abs_y - PERSON_R <= CURTAIN_ABS_Y

    def in_warning_zone(self):
        return self.abs_y - PERSON_R < FENCE_Y + WARN_ZONE

    @property
    def status(self):
        if abs(self.abs_y - self.target) > 1e-9:
            return "entering" if self.target < self.abs_y else "leaving"
        return "outside" if self.abs_y >= PERSON_START - 1e-9 else "INSIDE the cell"


class LightCurtain:
    """Emitter / receiver posts with three beams across the door (Lab 3 Cuboids). Green beams = clear, red = tripped."""

    def __init__(self, env, x0, x1, y):
        self.green, self.red, self.tripped = [], [], None
        for z in BEAM_Z:
            for store, colour in ((self.green, GREEN), (self.red, RED)):
                beam = Cuboid(scale=[x1 - x0, 0.012, 0.012], color=list(colour))
                env.add(beam)
                store.append((beam, transl((x0 + x1) / 2, y, z)))
        for x in (x0, x1):
            _box(env, (0.07, 0.07, 1.0), (x, y, 0.5), (0.10, 0.10, 0.12, 1.0))
            _box(env, (0.075, 0.075, 0.06), (x, y, 1.03), RED)
        self.set(False)

    def set(self, tripped):
        if tripped == self.tripped:
            return
        self.tripped = tripped
        for beam, T in self.green:
            beam.T = PARK if tripped else T
        for beam, T in self.red:
            beam.T = T if tripped else PARK


class StackLight:
    """Red / amber / green tower: red = STOPPED, amber = READY, green = RUNNING."""

    def __init__(self, env, x, y):
        _box(env, (0.06, 0.06, 1.2), (x, y, 0.6), (0.35, 0.35, 0.38, 1.0))
        self.lit, self.state = {}, None
        for name, z, colour in ((STOPPED, 1.46, RED), (READY, 1.34, AMBER), (RUNNING, 1.22, GREEN)):
            dim = [0.3 * c for c in colour[:3]] + [1.0]
            env.add(Cylinder(radius=0.05, length=0.10, color=dim, pose=SE3(x, y, z)))
            lamp = Cylinder(radius=0.055, length=0.105, color=list(colour))
            env.add(lamp)
            self.lit[name] = (lamp, transl(x, y, z))
        self.set(READY)

    def set(self, state):
        if state == self.state:
            return
        self.state = state
        for name, (lamp, T) in self.lit.items():
            lamp.T = T if name == state else PARK


# ---------------- laser gates: the arms 'see' a car in front of them ----------------
LASER_Z = BELT_HEIGHT + 0.020  # beam height: inside the car body, above the belt
LASER_HALF_SPAN = 0.225  # emitter post / receiver post stand this far (m) either side of the belt centre line
LASER_INSET = CAR_SIZE[0] / 2 - 0.005  # beams sit this far either side of the stop point (5 mm inside the car's ends), so
#                                        BOTH beams are broken only when the car is centred on the stop point to within 5 mm
LASER_WAIT_S = 3.0  # seconds (of running time) an arm waits for its laser before it reports a fault


def ray_hit_prism(p0, p1, vertices, faces, face_normals):
    """Lab 3 Q3 / Lab 6 Q1 laser reading: cast the segment p0 -> p1 at a prism (Lab 5 Q2/3 RectangularPrism data) with
    line_plane_intersection + is_intersection_point_inside_triangle. Returns (range from p0, hit point) of the closest
    face it crosses, or None if the beam passes the prism by."""
    best = None
    for j, face in enumerate(faces):
        point, check = line_plane_intersection(face_normals[j], vertices[face][0], p0, p1)
        if check == 1:
            for triangle in np.array(list(combinations(face, 3)), dtype=int):
                if is_intersection_point_inside_triangle(point, vertices[triangle]):
                    dist = float(np.linalg.norm(np.asarray(point, float) - p0))
                    if best is None or dist < best[0]:
                        best = (dist, np.asarray(point, float))
                    break
    return best


class LaserGate:
    """Two laser beams across the belt at one robot's stop point (Lab 5 Q1 / Lab 3 Q3 thin Cylinder as the beam, Sphere
    marker where it hits, Lab 6 Q1 range reading). The emitter stands on the robot's side of the belt, the receiver on the
    other. Red beam = clear, green beam + green dot on the car = the car breaks it. `detected` is True only while BOTH beams
    are broken, i.e. the car is sitting at the stop point, so the arm may stamp / pick."""

    def __init__(self, env, x_stop, side):
        self.x_stop, self.side = x_stop, side
        self.detected, self.range, self.tripped = False, None, None
        self.beams = []  # per beam: emitter point, receiver point, clear beam, broken beam, hit marker (shape, pose) pairs
        span = 2 * LASER_HALF_SPAN
        for dx in (-LASER_INSET, +LASER_INSET):
            x = x_stop + dx
            p0 = np.array([x, side * LASER_HALF_SPAN, LASER_Z])  # emitter (robot side)
            p1 = np.array([x, -side * LASER_HALF_SPAN, LASER_Z])  # receiver (far side)
            T = SE3(*((p0 + p1) / 2)) * SE3.Rx(pi / 2)  # cylinder axis turned from z to y, i.e. across the belt
            clear = Cylinder(radius=0.0025, length=span, color=list(RED))
            broken = Cylinder(radius=0.0025, length=span, color=list(GREEN))
            marker = Sphere(radius=0.012, color=list(GREEN))
            for shape in (clear, broken, marker):
                env.add(shape)
            for y in (side * LASER_HALF_SPAN, -side * LASER_HALF_SPAN):  # the two sensor heads on their stands
                _box(env, (0.03, 0.03, 0.05), (x, y, LASER_Z), (0.10, 0.10, 0.12, 1.0))
                _box(env, (0.02, 0.02, LASER_Z - 0.025), (x, y, (LASER_Z - 0.025) / 2), (0.35, 0.35, 0.38, 1.0))
            self.beams.append({"p0": p0, "p1": p1, "clear": (clear, T.A), "broken": (broken, T.A), "marker": marker})
        self.update([])

    def update(self, cars):
        """Read the gate once (called every frame, before the production logic runs, like a PLC input scan)."""
        prisms = []
        for car in cars:
            c = car.T[:3, 3]
            if abs(c[0] - self.x_stop) < 0.6 and abs(c[1]) < BELT_WIDTH / 2 and abs(c[2] - BELT_HEIGHT) < 1e-3:  # on the belt
                prisms.append(RectangularPrism(CAR_SIZE[0], CAR_SIZE[1], CAR_SIZE[2],
                                               center=[c[0], c[1], c[2] + CAR_SIZE[2] / 2]).get_data())
        ranges = []
        for beam in self.beams:
            hits = [h for h in (ray_hit_prism(beam["p0"], beam["p1"], v, f, n) for v, f, n in prisms) if h is not None]
            hit = min(hits, key=lambda h: h[0]) if hits else None
            ranges.append(None if hit is None else hit[0])
            beam["clear"][0].T = PARK if hit else beam["clear"][1]  # swap beam colour (same trick as LightCurtain)
            beam["broken"][0].T = beam["broken"][1] if hit else PARK
            beam["marker"].T = transl(*hit[1]) if hit else PARK
        self.detected = all(r is not None for r in ranges)
        self.range = min((r for r in ranges if r is not None), default=None)  # the LTM reading (distance to the car's side)
        self.tripped = any(r is not None for r in ranges)


class HardwareEstop:
    """NON-BLOCKING reader of the physical e-stop input (a gamepad button read with pygame, Lab 11).
    pressed() is called once per frame and never waits. By default a pressed button = e-stop pressed
    (HW_ESTOP_INVERT flips that). If the gamepad is unplugged while running, the e-stop counts as PRESSED (fail-safe).
    With no gamepad connected only the simulated GUI e-stop is available."""

    def __init__(self, button=HW_ESTOP_BUTTON, invert=HW_ESTOP_INVERT):
        self.button, self.invert = button, invert
        self.joy, self.status, self.connected = None, "NOT CONNECTED", False
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()
            self.connected, self.status = True, "released"
            print(f"Hardware e-stop: gamepad '{self.joy.get_name()}', button {button}")
        else:
            print("WARNING: no gamepad found - only the simulated GUI e-stop is active.")

    def pressed(self):
        """True = e-stop pressed (or input lost). Never blocks."""
        if not self.connected:
            return False
        try:
            pygame.event.pump()
            state = bool(self.joy.get_button(self.button)) != self.invert
        except Exception:
            self.status = "LINK LOST"
            return True
        self.status = "PRESSED" if state else "released"
        return state

    def close(self):
        pygame.quit()


class SafetySystem:
    """E-stop latch state machine. No loops: update() is called once per frame and only changes state on events.

        READY --START--> RUNNING --(e-stop / light curtain)--> STOPPED --RESET (cause cleared)--> READY --START--> RUNNING

    A stop is LATCHED: releasing the e-stop or clearing the zone never restarts the cell. RESET is refused while any cause is
    still active, and RESET only goes to READY (motion still inhibited); a separate START is needed to run again."""

    def __init__(self):
        self.state, self.latched = READY, []
        self.gui_estop = self.hw_pressed = self.zone_tripped = False
        self.reset_req = self.start_req = False
        self.message = "READY: press START to enable motion"

    # --- requests (called from the GUI button callbacks; they only set flags) ---
    def press_gui_estop(self):
        self.gui_estop = True

    def release_gui_estop(self):
        self.gui_estop = False

    def request_reset(self):
        self.reset_req = True

    def request_start(self):
        self.start_req = True

    def update(self, hw_pressed, zone_tripped):
        """Returns True on the frame the system enters STOPPED."""
        self.hw_pressed, self.zone_tripped = hw_pressed, zone_tripped
        active = []
        if self.gui_estop:
            active.append("GUI E-STOP")
        if hw_pressed:
            active.append("HW E-STOP")
        if zone_tripped:
            active.append("LIGHT CURTAIN")
        for cause in active:
            if cause not in self.latched:
                self.latched.append(cause)
                self.message = f"STOP: {cause} - motion halted and latched"
        newly_stopped = bool(active) and self.state != STOPPED
        if newly_stopped:
            self.state = STOPPED
        reset, start = self.reset_req, self.start_req
        self.reset_req = self.start_req = False
        if self.state == STOPPED:
            if reset:
                if active:
                    self.message = "RESET refused - still active: " + ", ".join(active)
                else:
                    self.latched.clear()
                    self.state = READY
                    self.message = "Reset acknowledged - system READY (nothing moves). Press START to resume."
            elif start:
                self.message = "START refused - release the cause, press RESET first"
            elif not active and not newly_stopped:
                self.message = "Cause cleared - press RESET to acknowledge (the cell does NOT restart by itself)"
        elif self.state == READY and start:
            self.state = RUNNING
            self.message = "RUNNING"
        return newly_stopped


# =============================================================================
# 6. THE CELL: frame loop + safety state, motion (generators, one yield = one frame), stamping, sorting, GUI
# =============================================================================
class ToyCarCell:
    def __init__(self, env):
        self.env = env
        # --- scene (Lab 3 shapes) ---
        _box(env, (BELT_LENGTH, BELT_WIDTH, BELT_HEIGHT), (BELT_LENGTH / 2, 0, BELT_HEIGHT / 2), BELT_COLOUR)
        _box(env, PAD_SIZE, (STATION_X, STATION_SIDE * PAD_OFFSET_Y, PAD_HEIGHT / 2), PAD_COLOUR)  # FANUC pad
        build_sorting_equipment(env)
        build_safety_cell(env)
        self.cars = [ToyCar(env, x, CAR_COLOURS[c]) for x, c in CARS]
        self.colour_of = {id(c): name for c, (_, name) in zip(self.cars, CARS)}
        self.order = sorted(self.cars, key=lambda c: -c.T[0, 3])  # the car furthest along the belt goes first
        env.set_camera_pose([BELT_LENGTH / 2, -5.0, 3.0], [BELT_LENGTH / 2, 0.0, 0.5])  # Lab 4 / 5
        self.marks = []  # (car, stroke shape, pose relative to car) - stamp marks travel with their car
        # --- FANUC (the personally assessed model) ---
        fanuc = make_fanuc(env)
        self.stamp_tool = StampTool(env, fanuc)
        self.fanuc = Arm("FANUC", fanuc, Obstacle(env, FANUC_OBS_SIZE, FANUC_OBS_MARGIN), TOOL_LEN, FANUC_EPS, FANUC_STEP)
        # --- UR3e (Lab 3 DHRobot + Lab 4 meshes) ---
        ur = make_ur3e(env)
        self.gripper = RG2Gripper(env, ur)
        self.ur = Arm("UR3e", ur, Obstacle(env, UR_OBS_SIZE, UR_OBS_MARGIN), 0.0, UR_EPS, UR_STEP, UR_SELF_RADII)
        self.ur.safe_q = Q_UR_HOME.copy()
        self.arms = [self.fanuc, self.ur]
        # --- laser gates: each arm only starts when its own gate sees the car at the stop point ---
        self.gates = {"FANUC": LaserGate(env, STATION_X, STATION_SIDE), "UR3e": LaserGate(env, PICK_X, SORT_SIDE)}
        self.carried, self.rel = None, None  # car in the gripper and its pose in the TCP frame
        self.counts = {k: 0 for k in TUB_XY}
        self.stage, self.stamped = "idle", 0
        # --- safety equipment + state ---
        door_l, door_r = SORT_X - DOOR_W / 2, SORT_X + DOOR_W / 2
        self.curtain = LightCurtain(env, door_l + 0.10, door_r - 0.10, SORT_SIDE * CURTAIN_ABS_Y)
        self.tower = StackLight(env, door_r + 0.35, SORT_SIDE * (FENCE_Y + 0.35))
        self.person = Person(env)
        self.safety = SafetySystem()
        self.hw = HardwareEstop()
        # --- jobs: one generator at a time, advanced one frame per tick ONLY while the system is RUNNING ---
        self.cmds = []  # GUI / console requests (appended from other threads, drained in the main loop)
        self.jobs, self.job, self.job_kind = [], None, None
        self.production_state = "not started"
        self.speed_acc, self.frame = 0.0, 0
        self._key_prev = {}
        # --- jogging / final pose ---
        self.sel, self.jog_joint, self.jog_deg, self.jog_mm = 0, 1, 5.0, 20.0
        self.final_q = {"FANUC": FANUC_FINAL_DEFAULT.copy(), "UR3e": UR_FINAL_DEFAULT.copy()}
        self.message = self.safety.message
        self.build_gui()
        env.step()

    # =========================================================================
    # GUI (Swift side panel): labels for state / faults, buttons and sliders
    # =========================================================================
    def build_gui(self):
        env, s, cmds, person = self.env, self.safety, self.cmds, self.person
        self.labels = [swift.Label("") for _ in range(8)]
        for lb in self.labels:
            env.add(lb)
        self._last = [None] * len(self.labels)

        def button(desc, fn):
            def cb(*_):  # Swift calls this from its own thread: callbacks only set flags / append requests
                fn()
            env.add(swift.Button(cb=cb, desc=desc))

        def slider(desc, lo, hi, step, value, fn, unit=""):
            def cb(v):
                fn(v)
            env.add(swift.Slider(cb=cb, min=lo, max=hi, step=step, value=value, desc=desc, unit=unit))

        def request(*cmd):
            return lambda: cmds.append(cmd)

        env.add(swift.Label(f"--- SAFETY  (assessed model: FANUC M-10iA, student {STUDENT_NUMBER}) ---"))
        button("E-STOP (simulated GUI)", s.press_gui_estop)
        button("Release GUI e-stop", s.release_gui_estop)
        button("RESET (acknowledge fault)", s.request_reset)
        button("START / RESUME", s.request_start)
        env.add(swift.Label("--- SIMULATED SENSOR SIGNALS ---"))
        button("Person ENTERS through the door (light curtain)", person.go_inside)
        button("Person LEAVES", person.go_outside)
        button("Place obstacle in FANUC path (key o)", request("obstacle", self.fanuc, True))
        button("Remove FANUC obstacle (key p)", request("obstacle", self.fanuc, False))
        button("Place obstacle in UR3e path (key k)", request("obstacle", self.ur, True))
        button("Remove UR3e obstacle (key l)", request("obstacle", self.ur, False))
        env.add(swift.Label("--- OPERATION ---"))
        button("RUN PRODUCTION (auto)", request("production"))
        button("Jog: select FANUC", request("select", 0))
        button("Jog: select UR3e", request("select", 1))
        env.add(swift.Label("--- JOINT JOGGING ---"))
        slider("Joint to jog (1-6)", 1, 6, 1, 1, lambda v: setattr(self, "jog_joint", int(round(v))))
        slider("Joint step", 1, 20, 1, 5, lambda v: setattr(self, "jog_deg", float(v)), "deg")
        button("Joint  -", request("jog_joint", -1))
        button("Joint  +", request("jog_joint", +1))
        env.add(swift.Label("--- CARTESIAN JOGGING (tool / flange x y z, world frame) ---"))
        slider("Cartesian step", 5, 100, 5, 20, lambda v: setattr(self, "jog_mm", float(v)), "mm")
        for axis, name in enumerate("XYZ"):
            button(f"{name} +", request("jog_cart", axis, +1))
            button(f"{name} -", request("jog_cart", axis, -1))
        env.add(swift.Label("--- FINAL JOINT STATE ---"))
        button("STORE current pose of selected robot as FINAL", request("store_final"))
        button("GO TO FINAL POSE (selected robot)", request("go_final"))

    def say(self, text):
        self.message = text
        print(text)

    def status_lines(self):
        s, arm = self.safety, self.arms[self.sel]
        if s.state == STOPPED:
            sys_t = "*** STOPPED (latched): " + ", ".join(s.latched) + "  ->  clear cause, RESET, then START"
        elif s.state == READY:
            sys_t = "READY - motion inhibited. Press START"
        else:
            sys_t = f"RUNNING - speed {int(100 * self.speed_scale())}%"
        hw_t = self.hw.status
        inputs = (f"GUI e-stop: {'PRESSED' if s.gui_estop else 'released'}  |  HW e-stop: {hw_t}  |  "
                  f"light curtain: {'TRIPPED' if s.zone_tripped else 'clear'}  |  person: {self.person.status}")
        prod = (f"production: {self.production_state}  |  {self.stage}  |  stamped {self.stamped}/{len(self.cars)}  "
                f"sorted {sum(self.counts.values())}/{len(self.cars)} (R{self.counts['red']} G{self.counts['green']} "
                f"B{self.counts['blue']})")
        def gate_text(g):
            return ("CAR DETECTED" if g.detected else "car entering" if g.tripped else "clear") + \
                   (f" ({g.range:.3f} m)" if g.range is not None else "")
        lasers = "LASER gates:  " + "  |  ".join(f"{n}: {gate_text(g)}" for n, g in self.gates.items())
        lines = [f"SYSTEM: {sys_t}", inputs, f"MESSAGE: {self.message}", prod, lasers]
        for a in self.arms:
            held = "  ** HELD: obstacle in path **" if a.blocked else ""
            lines.append(f"{a.name}: {a.note}  |  obstacle {'IN PATH' if a.obstacle.active else 'none'}  |  "
                         f"re-plans {a.replans}{held}")
        q = np.rad2deg(np.asarray(arm.robot.q, float))
        t = arm.robot.fkine(arm.robot.q).t
        lines.append(f"JOG [{arm.name}] joint J{self.jog_joint} step {self.jog_deg:g} deg, cart step {self.jog_mm:g} mm  |  "
                     f"q(deg) = {np.round(q, 1).tolist()}  |  xyz(m) = {np.round(t, 3).tolist()}")
        return lines

    def update_status(self):
        s = self.safety
        self.curtain.set(s.zone_tripped)
        self.tower.set(s.state)
        if self.frame % 4 and self.frame > 1:  # pose text changes every frame: refresh it at a quarter of the frame rate
            return
        for k, (lbl, t) in enumerate(zip(self.labels, self.status_lines())):
            if t != self._last[k]:
                lbl.desc = t
                self._last[k] = t

    # =========================================================================
    # one frame: inputs -> safety state machine -> (only if RUNNING) advance the active job -> draw -> step
    # =========================================================================
    def key_edge(self, k):
        """Lab 11: keyboard.is_pressed inside the loop (rising edge so one tap = one action)."""
        try:
            now = keyboard.is_pressed(k)
        except Exception:  # the keyboard library needs admin / root on some systems
            now = False
        was, self._key_prev[k] = self._key_prev.get(k, False), now
        return now and not was

    def poll_keys(self):
        if self.key_edge("o"):
            self.set_obstacle(self.fanuc, True)
        if self.key_edge("p"):
            self.set_obstacle(self.fanuc, False)
        if self.key_edge("k"):
            self.set_obstacle(self.ur, True)
        if self.key_edge("l"):
            self.set_obstacle(self.ur, False)
        if self.key_edge("z"):
            self.person.toggle()
        if self.key_edge("x"):
            self.safety.press_gui_estop()

    def set_obstacle(self, arm, on):
        if not on:
            arm.obstacle.remove()
        elif arm is self.fanuc and not arm.obstacle.active:
            arm.obstacle.place([STATION_X, STATION_SIDE * OBSTACLE_Y, BELT_HEIGHT + FANUC_OBS_SIZE / 2])
        elif arm is self.ur and not arm.obstacle.active:
            arm.obstacle.place(UR_OBSTACLE_P)

    def speed_scale(self):
        return SLOW_SCALE if self.person.in_warning_zone() else 1.0

    def follow(self):
        """Everything attached to an arm follows it (Lab 4 Q1: shape.T = fkine(q) @ ...)."""
        self.fanuc.robot.draw()
        self.ur.robot.draw()
        self.stamp_tool.update()
        if self.carried is not None:
            self.carried.T = self.ur.robot.fkine(self.ur.robot.q).A @ self.rel
        self.gripper.update()
        for car, stroke, rel in self.marks:
            stroke.T = car.T @ rel

    def tick(self):
        self.frame += 1
        self.poll_keys()
        self.person.update(DT)  # the person moves whatever the cell is doing
        entered_stop = self.safety.update(self.hw.pressed(), self.person.in_curtain())
        if entered_stop:
            self.on_stop()
        if (self.safety.state == RUNNING and self.production_state == "not started" and not MANUAL_MODE
                and not self.jobs and self.job is None):  # START begins the production run (use --manual to jog first)
            self.start_production()
        if self.safety.message != self.message and self.safety.message != getattr(self, "_printed", None):
            self.say(self.safety.message)
            self._printed = self.safety.message
        self.process_commands()
        for gate in self.gates.values():  # sensors are read every frame (input scan), whatever the cell state
            gate.update(self.cars)
        if self.safety.state == RUNNING:  # STOPPED / READY: no job is advanced, so nothing moves
            self.speed_acc += self.speed_scale()  # 50 % speed = advance the job every second frame
            if self.speed_acc >= 1.0:
                self.speed_acc -= 1.0
                self.advance_job()
        self.follow()
        self.update_status()
        self.env.step(DT)

    def on_stop(self):
        """Known safe state: queued / running jogs are cancelled; production and final moves stay paused where they are."""
        self.jobs = [j for j in self.jobs if j[0] != "jog"]
        if self.job_kind == "jog" and self.job is not None:
            self.job.close()
            self.job = None
        for a in self.arms:
            if a.note.startswith("jog"):
                a.note = "idle (jog cancelled by stop)"

    def advance_job(self):
        if self.job is None and self.jobs:
            self.job_kind, self.job = self.jobs.pop(0)
        if self.job is None:
            return
        try:
            next(self.job)
        except StopIteration:
            self.job = None
        except RuntimeError as e:  # e.g. no IK solution: report once, stop commanding, keep the window alive
            self.say(f"SOFTWARE FAULT ({self.job_kind}): {e}")
            if self.job_kind == "production":
                self.production_state = "FAULT"
            self.job = None

    def run(self):
        """Main loop (Lab 11 structure). Paced to DT per frame - never a busy spin."""
        while True:
            t0 = time.perf_counter()
            self.tick()
            spare = DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    # =========================================================================
    # requests from the GUI / terminal (drained once per frame, in the main thread)
    # =========================================================================
    def process_commands(self):
        while self.cmds:  # finite: only what has arrived since the last frame
            cmd = self.cmds.pop(0)
            kind, arm = cmd[0], self.arms[self.sel]
            if kind == "select":
                self.sel = cmd[1]
                self.say(f"Jog target: {self.arms[self.sel].name}")
            elif kind == "obstacle":
                self.set_obstacle(cmd[1], cmd[2])
            elif kind == "production":
                self.start_production()
            elif kind == "jog_joint":
                j = self.jog_joint - 1
                self.queue_jog(self.joint_jog(arm, j, cmd[1] * np.deg2rad(self.jog_deg)))
            elif kind == "jog_cart":
                self.queue_jog(self.cart_jog(arm, cmd[1], cmd[2] * self.jog_mm / 1000.0))
            elif kind == "store_final":
                self.final_q[arm.name] = np.array(arm.robot.q, float)
                self.say(f"{arm.name}: FINAL pose stored = {np.round(np.rad2deg(self.final_q[arm.name]), 1).tolist()} deg")
            elif kind == "go_final":
                self.go_final([arm])
            elif kind == "console":
                self.handle_console(cmd[1])

    def motion_allowed(self, what):
        if self.safety.state != RUNNING:
            self.say(f"{what} ignored: system is {self.safety.state} (press START when READY)")
            return False
        if self.production_state == "running":
            self.say(f"{what} ignored: production is running (manual control is available when it is not)")
            return False
        return True

    def queue_jog(self, gen):
        if self.motion_allowed("Jog") and len(self.jobs) < MAX_QUEUED_JOGS:
            self.jobs.append(("jog", gen))
        else:
            gen.close()

    def start_production(self):
        if self.safety.state != RUNNING:
            self.say(f"RUN PRODUCTION ignored: system is {self.safety.state} (press START when READY)")
        elif self.production_state != "not started":
            self.say(f"Production is already '{self.production_state}' - restart the script to run it again")
        else:
            self.production_state = "running"
            self.jobs.append(("production", self.production()))

    def set_final(self, arm, q, go=True):
        """Validate a final joint state against the joint limits (Lab 4 Q2/3 check), store it, optionally go there."""
        lo, hi = arm.robot.qlim
        bad = [j + 1 for j in range(6) if q[j] < lo[j] or q[j] > hi[j]]
        if bad:
            self.say(f"{arm.name}: final state REJECTED - outside the joint limits of J{bad}")
            return
        self.final_q[arm.name] = np.asarray(q, float)
        self.say(f"{arm.name}: FINAL pose = {np.round(np.rad2deg(q), 1).tolist()} deg")
        if go:
            self.go_final([arm])

    def go_final(self, arms):
        if self.production_state == "running":
            self.say("Final pose stored - production will finish by moving each arm to it")
        elif self.motion_allowed("GO TO FINAL"):
            self.jobs.append(("final", self.final_job([a.name for a in arms])))

    def handle_console(self, line):
        """Typed command from the terminal:   final fanuc|ur d1 d2 d3 d4 d5 d6   (degrees)."""
        p = line.split()
        if len(p) == 8 and p[0] == "final" and p[1] in ("fanuc", "ur"):
            try:
                q = np.deg2rad([float(v) for v in p[2:]])
            except ValueError:
                self.say("final: the six joint angles must be numbers (degrees)")
                return
            self.set_final(self.fanuc if p[1] == "fanuc" else self.ur, q)
        elif p:
            print("Terminal command:  final fanuc d1 d2 d3 d4 d5 d6   |   final ur d1 d2 d3 d4 d5 d6   (degrees)")

    # =========================================================================
    # MOTION: every method below is a generator - `yield` = "give the frame back"; the main loop only resumes it while
    # the system is RUNNING, so an e-stop freezes it mid-move and a later START continues from exactly that pose.
    # =========================================================================
    def wait(self, seconds):
        for _ in range(int(seconds / DT)):
            yield

    def apply(self, arm, q):
        """Set robot.q = q, unless that pose would hit the obstacle: then hold the arm (one frame at a time) until the
        way is clear (Lab 5 Q2/3 is_collision on the NEXT pose). Returns True if it had to wait."""
        waited = False
        while arm.hits([q]):
            arm.blocked = waited = True
            yield
        arm.blocked = False
        arm.robot.q = q
        yield
        return waited

    # ---------------- planned joint move with look-ahead + re-planning (Lab 5 Q2/3 primitive RRT) ----------------
    def replan(self, arm, q_start, q_goal):
        """Straight joint path is blocked (obstacle or the arm hitting itself): first try going via the arm's safe upright
        pose, then insert random waypoints (not in collision) until every segment is clear (Lab 5 Q2/3 primitive RRT)."""
        qlim = arm.robot.qlim
        q_waypoints = [np.asarray(q_start, float), np.asarray(q_goal, float)]
        if arm.safe_q is not None:  # known good upright pose: start -> safe pose -> goal
            via = [q_waypoints[0], np.asarray(arm.safe_q, float), q_waypoints[1]]
            segs = [interpolate_waypoints_radians([via[k], via[k + 1]], arm.step) for k in range(2)]
            if not any(arm.unsafe(seg[::2]) for seg in segs):
                return segs[0] + segs[1]
        for _ in range(RRT_TRIES):
            path, bad = [], None
            for i in range(len(q_waypoints) - 1):
                seg = interpolate_waypoints_radians([q_waypoints[i], q_waypoints[i + 1]], arm.step)
                if arm.unsafe(seg):
                    bad = i
                    break
                path.extend(seg)
            if bad is None:
                return path
            for _ in range(50):  # randomly pick a pose that is not in collision
                q_rand = np.clip(q_waypoints[bad] + (2 * np.random.rand(6) - 1) * RRT_SPREAD, qlim[0], qlim[1])
                if not arm.unsafe([q_rand]):
                    break
            q_waypoints.insert(bad + 1, q_rand)
        return None

    def plan_move(self, arm, q_goal, note=None):
        """Joint move that looks ahead every frame; if the obstacle appears in the path it RE-PLANS around it. If no way
        round is found the arm is held (apply) and stops before contact. Used by both robots."""
        if note:
            arm.note = note
        q_goal = np.asarray(q_goal, float)
        path = interpolate_waypoints_radians([arm.robot.q, q_goal], arm.step)
        if arm.self_hits(path[::2]):  # the straight joint path would push the arm through itself: plan another way
            new = self.replan(arm, np.asarray(arm.robot.q, float).copy(), q_goal)
            if new is None:
                raise RuntimeError(f"{arm.name}: no joint path that keeps the arm from passing through itself")
            path = new
            arm.replans += 1
            arm.note = "RE-PLANNED to avoid self-collision"
        i, tries = 0, 0
        while i < len(path):
            if not arm.obstacle.active:
                tries = 0
            if arm.obstacle.active and tries < 3 and arm.hits(path[i:i + LOOKAHEAD:3]):
                tries += 1
                new = self.replan(arm, arm.robot.q.copy(), q_goal)
                if new is not None:
                    path, i = new, 0
                    arm.replans += 1
                    arm.note = "RE-PLANNED around the obstacle"
                    continue
            yield from self.apply(arm, path[i])
            i += 1

    # ---------------- RMRC (Lab 9 Q1, with the Lab 6 Q3 trapezoidal blend and the Lab 7 / 9 DLS) ----------------
    def rmrc(self, arm, p_end, speed, Rd):
        robot, qlim = arm.robot, arm.robot.qlim.T
        p_start = np.asarray(robot.fkine(robot.q).t, float)
        p_end = np.asarray(p_end, float)
        steps = max(int(np.linalg.norm(p_end - p_start) / (speed * DT)), 2)
        s = trapezoidal(0, 1, steps).q
        x = np.zeros([3, steps])
        for i in range(steps):
            x[:, i] = (1 - s[i]) * p_start + s[i] * p_end
        W = np.diag([1, 1, 1, 0.1, 0.1, 0.1])  # weighting matrix for the velocity vector (Lab 9)
        i = 0
        while i < steps - 1:
            T = robot.fkine(robot.q).A
            delta_x = x[:, i + 1] - T[:3, 3]
            Ra = T[:3, :3]
            Rdot = (1 / DT) * (Rd - Ra)
            S = Rdot @ Ra.T
            linear_velocity = (1 / DT) * delta_x
            angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
            xdot = W @ np.r_[linear_velocity, angular_velocity]
            J = robot.jacob0(robot.q)
            m = np.sqrt(max(linalg.det(J @ J.T), 0.0))  # measure of manipulability (Lab 7 / 9); max() avoids NaN at a singularity
            m_lambda = (1 - m / arm.epsilon) * 0.05 if m < arm.epsilon else 0  # DLS only near a singularity
            inv_j = linalg.inv(J.T @ J + m_lambda * np.eye(6)) @ J.T
            qdot = inv_j @ xdot
            for j in range(6):  # joint limits: stop the motor (Lab 9)
                if robot.q[j] + DT * qdot[j] < qlim[j, 0] or robot.q[j] + DT * qdot[j] > qlim[j, 1]:
                    qdot[j] = 0
            waited = yield from self.apply(arm, robot.q + DT * qdot)
            if not waited:  # if the arm had to wait, redo this step from where it stopped
                i += 1

    # ---------------- inverse kinematics (Lab 3 Q1 ikine_LM, Lab 4 Q2/3 joint-limit check) ----------------
    @staticmethod
    def ik(robot, T, seeds):
        qlim = robot.qlim
        for q0 in seeds:
            sol = robot.ikine_LM(T, q0=q0)
            if sol.success and np.all(sol.q >= qlim[0]) and np.all(sol.q <= qlim[1]):
                return sol.q
        return None

    def ik_safe(self, arm, T, seeds):
        """Like ik(), but looks at ALL the seeds instead of stopping at the first one that converges, and returns the
        best solution that (a) is inside the joint limits, (b) does not put the arm through itself and (c) needs the
        smallest joint travel. Every solution is first wrapped by whole turns to the nearest equivalent of the current q,
        so a joint never takes the long way round (the UR3e joints are +-2 pi, so q and q +- 2 pi are the same pose)."""
        robot, qlim = arm.robot, arm.robot.qlim
        ref = np.asarray(robot.q, float)
        best, best_cost = None, None
        for q0 in seeds:
            sol = robot.ikine_LM(T, q0=q0)
            if not sol.success:
                continue
            q = ref + (sol.q - ref + pi) % (2 * pi) - pi
            if np.any(q < qlim[0]) or np.any(q > qlim[1]):
                q = np.asarray(sol.q, float)
            if np.any(q < qlim[0]) or np.any(q > qlim[1]) or arm.self_hits([q]):
                continue
            cost = float(np.abs(q - ref).max())
            if best is None or cost < best_cost:
                best, best_cost = q, cost
        return best

    # ---------------- manual jogging ----------------
    def joint_jog(self, arm, j, delta):
        """Move ONE joint by delta (rad), clipped to its joint limit (read when the job starts, so queued jogs chain)."""
        q = np.array(arm.robot.q, float)
        lo, hi = arm.robot.qlim
        goal = q.copy()
        goal[j] = np.clip(q[j] + delta, lo[j], hi[j])
        if abs(goal[j] - q[j]) < 1e-6:
            self.say(f"{arm.name}: J{j + 1} is at its joint limit")
            return
        yield from self.plan_move(arm, goal, f"jog J{j + 1} {np.rad2deg(goal[j] - q[j]):+.1f} deg")
        arm.note = "idle (manual)"

    def cart_jog(self, arm, axis, delta):
        """Move the tool / flange `delta` metres along world x, y or z with RMRC, holding its orientation."""
        T = arm.robot.fkine(arm.robot.q).A
        p = T[:3, 3].copy()
        p[axis] += delta
        arm.note = f"jog {'XYZ'[axis]} {delta * 1000:+.0f} mm (RMRC)"
        yield from self.rmrc(arm, p, JOG_SPEED, T[:3, :3])
        arm.note = "idle (manual)"

    # ---------------- final joint state (chosen / supplied at marking) ----------------
    def final_job(self, names):
        for name in names:
            arm = self.fanuc if name == "FANUC" else self.ur
            self.stage = f"{name}: planning to the final joint state"
            yield from self.plan_move(arm, self.final_q[name], "moving to the FINAL joint state")
            arm.note = "at FINAL pose"
        self.stage = "final pose reached"

    # ---------------- conveyor ----------------
    def slide(self, car, x_goal):
        """Slide `car` along the belt: trapezoidal blend (accelerate, cruise, decelerate), one step per frame."""
        x0 = float(car.T[0, 3])
        steps = max(int(abs(x_goal - x0) / (CAR_SLIDE_SPEED * DT)), 2)
        s = trapezoidal(0, 1, steps).q
        for k in range(steps):
            car.T = transl(x0 + s[k] * (x_goal - x0), 0, BELT_HEIGHT)
            yield
        car.T = transl(x_goal, 0, BELT_HEIGHT)

    def wait_for_laser(self, arm, gate):
        """The arm does nothing until its laser gate reports a car at the stop point (both beams broken). If the laser has
        not seen a car after LASER_WAIT_S seconds of running time the job faults instead of waiting forever."""
        arm.note = "waiting for the laser to detect a car"
        for _ in range(int(LASER_WAIT_S / DT)):
            if gate.detected:
                arm.note = f"laser: car detected ({gate.range:.3f} m)"
                return
            yield
        raise RuntimeError(f"{arm.name}: the laser gate never detected a car at the stop point")

    def add_stroke(self, car, a, b, roof_z):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length, mid, thick = float(np.linalg.norm(b - a)), (a + b) / 2, 0.003
        angle = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
        pose = SE3(mid[0], mid[1], roof_z + thick / 2) * SE3.Rz(angle)
        stroke = Cuboid(scale=[length + STROKE_W, STROKE_W, thick], color=list(STROKE_COLOUR), pose=pose)
        self.env.add(stroke)
        self.marks.append((car, stroke, linalg.inv(car.T) @ pose.A))

    # ---------------- FANUC: hover -> press -> dwell -> lift -> home ----------------
    def stamp_cycle(self, car):
        roof_z = BELT_HEIGHT + CAR_SIZE[2]
        z_hover, z_press = roof_z + TOOL_LEN + STAMP_HOVER, roof_z + TOOL_LEN
        q_hover = self.ik(self.fanuc.robot, SE3(STATION_X, 0.0, z_hover) * SE3.Rx(pi), FANUC_SEEDS)
        if q_hover is None:
            raise RuntimeError("FANUC: no IK solution above the car - check base pose / joint limits")
        yield from self.plan_move(self.fanuc, q_hover, "moving to the hover pose")
        self.fanuc.note = "pressing the IR die onto the roof (RMRC)"
        yield from self.rmrc(self.fanuc, [STATION_X, 0.0, z_press], PLUNGE_SPEED, TOOL_DOWN)  # down onto the roof
        for stroke in letter_strokes(float(car.T[0, 3])):  # letters centred on the car roof at the moment of the press
            for a, b in zip(stroke[:-1], stroke[1:]):
                self.add_stroke(car, a, b, roof_z)
        yield from self.wait(STAMP_DWELL)
        self.fanuc.note = "lifting away (RMRC)"
        yield from self.rmrc(self.fanuc, [STATION_X, 0.0, z_hover], PLUNGE_SPEED, TOOL_DOWN)  # lift away
        yield from self.plan_move(self.fanuc, Q_HOME.copy(), "returning home")
        self.fanuc.note = "idle (home)"

    # ---------------- UR3e ----------------
    def arm_goto(self, arm, p, note, yaw=0.0):
        """TCP at p, tool pointing down, turned by yaw. A few vertical offsets are tried before giving up."""
        robot = arm.robot
        for dz in (0.0, 0.04, 0.08, -0.04):
            T = SE3(*(np.asarray(p, float) + [0, 0, dz])) * SE3.Rz(yaw) * SE3.Rx(pi)
            q = self.ik_safe(arm, T, [robot.q.copy()] + UR_SEEDS)  # never a pose where the arm passes through itself
            if q is not None:
                yield from self.plan_move(arm, q, note)
                return
        raise RuntimeError(f"{arm.name}: no IK solution for {np.round(p, 3)} - check the layout / reach")

    def gripper_move(self, closing):
        w0, w1 = self.gripper.opening, (GRIP_CLOSED if closing else GRIP_OPEN)
        n = max(int(PICK_DWELL / DT), 1)
        for k in range(1, n + 1):
            self.gripper.opening = w0 + (w1 - w0) * k / n
            yield

    def sort_cycle(self, car, colour):
        """HOVER over the car -> PICK (RMRC down, close the RG2, RMRC up) -> CARRY over its compartment ->
        PLACE (RMRC down, open the RG2) -> HOME."""
        ur, r = self.ur, self.ur.robot
        p_pick = np.array([float(car.T[0, 3]), 0.0, BELT_HEIGHT + TCP_Z])
        p_above = p_pick + [0, 0, PICK_HOVER]
        yield from self.arm_goto(ur, p_above, "moving the tool over the car")
        ur.note = "picking: lowering the tool onto the car (RMRC)"
        yield from self.rmrc(ur, p_pick, UR_LINE_SPEED, TOOL_DOWN)
        ur.note = "closing the RG2 on the car"
        yield from self.gripper_move(True)
        self.rel = linalg.inv(r.fkine(r.q).A) @ car.T  # the car is now held by the tool
        self.carried = car
        yield from self.rmrc(ur, p_above, UR_LINE_SPEED, TOOL_DOWN)
        tx, ty = TUB_XY[colour]
        rest_z = PAD_HEIGHT + TUB_FLOOR + self.counts[colour] * CAR_SIZE[2]  # cars stack if a tub already has one
        p_over = np.array([tx, ty, TRAVEL_Z])
        p_drop = np.array([tx, ty, rest_z + TCP_Z])
        Rd_box = SE3.Rz(BOX_CAR_YAW).R @ TOOL_DOWN  # wrist turned so the car's long axis lines up with the compartment
        yield from self.arm_goto(ur, p_over, f"carrying {colour} car to the {colour} tub", yaw=BOX_CAR_YAW)
        ur.note = f"placing in the {colour} tub (RMRC)"
        yield from self.rmrc(ur, p_drop, UR_LINE_SPEED, Rd_box)
        ur.note = "opening the RG2"
        yield from self.gripper_move(False)
        car.T = (SE3(tx, ty, rest_z) * SE3.Rz(BOX_CAR_YAW)).A  # settles on the tub floor / on the car below
        self.carried, self.rel = None, None
        self.counts[colour] += 1
        yield from self.rmrc(ur, p_over, UR_LINE_SPEED, Rd_box)
        yield from self.plan_move(ur, Q_UR_HOME, "returning home")
        ur.note = "waiting for a car"

    # ---------------- production line ----------------
    def production(self):
        """For each car in turn: slide to the FANUC -> stamp -> slide to the end of the belt -> UR3e sorts it.
        Then both arms plan to their FINAL joint states."""
        for car in self.order:
            colour = self.colour_of[id(car)]
            self.stage = f"{colour} car: moving to the FANUC"
            yield from self.slide(car, STATION_X)
            yield from self.wait_for_laser(self.fanuc, self.gates["FANUC"])  # FANUC starts only when ITS laser sees the car
            self.stage = f"{colour} car: FANUC stamping"
            yield from self.stamp_cycle(car)
            self.stamped += 1
            self.stage = f"{colour} car: moving to the end of the belt"
            yield from self.slide(car, PICK_X)
            yield from self.wait_for_laser(self.ur, self.gates["UR3e"])  # UR3e starts only when ITS laser sees the car
            self.stage = f"{colour} car: UR3e sorting"
            yield from self.sort_cycle(car, colour)
        yield from self.final_job(["FANUC", "UR3e"])
        self.stage = "all cars stamped and sorted, arms at final pose"
        self.production_state = "complete"
        self.say("Production complete - jogging and GO TO FINAL POSE are available again")


# =============================================================================
# 7. MAIN (Lab 11 structure: launch, loop, try / except KeyboardInterrupt / finally cleanup)
# =============================================================================
def console_listener(cell):
    """Lab 4 Q1 pattern: input() in a helper thread so typing never blocks the simulation."""
    while True:
        try:
            line = input()
        except EOFError:
            return
        cell.cmds.append(("console", line))


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    cell = ToyCarCell(env)
    threading.Thread(target=console_listener, args=(cell,), daemon=True).start()
    print("Scene loaded. Press START in the side panel to begin. Keys: o/p FANUC obstacle, k/l UR3e obstacle, "
          "z person at the door, x GUI e-stop. Terminal: final fanuc|ur d1..d6. Ctrl+C to quit.")
    try:
        cell.run()
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        cell.hw.close()
        env.close()


if __name__ == "__main__":
    main()