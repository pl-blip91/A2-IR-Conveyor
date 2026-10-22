from math import pi

import numpy as np
from scipy import linalg
from spatialgeometry import Cuboid
from spatialmath import SE3

from shared import line_config as cfg
from shared.motion import Arm, CubeObstacle, ik_safe, plan_move, rmrc, wait_for_sensor
from shared.scene import box
from shared.sensors import LASER_WAIT_S, LaserGate
from shared.station_base import RobotStation

from .ur3e import GRIP_CLOSED, GRIP_OPEN, RG2Gripper, make_ur3e

SORT_X, SORT_SIDE = cfg.SORT_X, cfg.SORT_SIDE  # UR3e (sorting station) at the end of the belt, on the -y side
PAD_HEIGHT = cfg.PAD_HEIGHT
CAR_SIZE = cfg.CAR_SIZE
CAR_H = CAR_SIZE[2]
SORT_BASE_INSET = 0.07
SORT_BASE = (SE3(SORT_X, SORT_SIDE * (cfg.PAD_OFFSET_Y - cfg.PAD_SIZE[1] / 2 + SORT_BASE_INSET), PAD_HEIGHT)
             * SE3.Rz(-SORT_SIDE * pi / 2))  # +x of the robot base faces the belt
PICK_X = cfg.BELT_LENGTH - CAR_SIZE[0] / 2  # where the lead car stops at the end of the belt

UR_EPS = 0.003  # manipulability below which RMRC damping switches on (the UR3e is small, so its values are small)
UR_STEP = np.deg2rad(1.8) * cfg.DT / 0.03  # largest joint step per frame (~1.05 rad/s peak: UR default movej speed)
UR_LINE_SPEED = 0.25  # m/s straight-line RMRC (UR default movel speed)
UR_OBS_SIZE, UR_OBS_MARGIN = 0.10, 0.04  # test obstacle: cube side and clearance kept round it (m)
TOOL_DOWN = SE3.Rx(pi).R
TCP_Z = 0.020
PICK_HOVER = 0.10
TRAVEL_Z = 0.80
PICK_DWELL = 0.5
Q_UR_HOME = np.deg2rad([29, -76.2, -91.9, -101.8, 90, 29])
UR_FINAL_DEFAULT = np.deg2rad([-15, -80, -95, -95, 90, 0])
UR_SEEDS = [np.deg2rad(g) for g in ([0, -90, -90, -90, 90, 0], [0, -60, -100, -110, 90, 0], [0, -100, -60, -110, 90, 0],
                                    [90, -90, -90, -90, 90, 0], [-90, -90, -90, -90, 90, 0], [180, -90, -90, -90, 90, 0])]
UR_OBSTACLE_P = (PICK_X - 0.215, -0.10, cfg.BELT_HEIGHT + 0.20)  # in the belt-end -> tub transit
# Self-collision clearance radius (m) of each UR3e segment: base column, upper arm, forearm, wrist 1, wrist 2, wrist 3, gripper.
UR_SELF_RADII = (0.07, 0.09, 0.06, 0.04, 0.04, 0.04, 0.05)

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
    box(env, (L, W, t), (cx, cy, z0 + t / 2), PLY)  # floor
    for sy in (-1, 1):  # front and back walls
        box(env, (L, t, h), (cx, cy + sy * (BOX_INNER_Y / 2 + t / 2), z0 + h / 2), PLY)
    for sx in (-1, 1):  # end walls
        box(env, (t, BOX_INNER_Y, h), (cx + sx * (L / 2 - t / 2), cy, z0 + h / 2), PLY)
    for x in (cx - BOX_PITCH / 2, cx + BOX_PITCH / 2):  # the two dividers
        box(env, (t, BOX_INNER_Y, h), (x, cy, z0 + h / 2), PLY)
    for key, letter in (("blue", "B"), ("red", "R"), ("green", "G")):
        lx, zc = TUB_XY[key][0], z0 + h * 0.55
        for (u0, v0, u1, v1) in LETTER_SEGMENTS[letter]:
            pa = (lx + _VIEW_RIGHT * (u0 - 0.5) * BOX_LETTER_W, zc + (v0 - 0.5) * BOX_LETTER_H)
            pb = (lx + _VIEW_RIGHT * (u1 - 0.5) * BOX_LETTER_W, zc + (v1 - 0.5) * BOX_LETTER_H)
            _segment(env, pa, pb, 0.0016, BOX_LETTER_STROKE, LETTER_COLOURS[letter], face_y + SORT_SIDE * 0.0016)


def build_sorting_equipment(env):
    """Stainless-steel bench that carries the UR3e and the plywood box."""
    deck_y = SORT_SIDE * 0.525
    box(env, (0.95, 0.55, PAD_HEIGHT), (SORT_X, deck_y, PAD_HEIGHT / 2), STEEL)
    box(env, (0.97, 0.57, 0.012), (SORT_X, deck_y, PAD_HEIGHT - 0.006), STEEL_DARK)
    build_storage_box(env)


class SortingStation(RobotStation):
    number, key, name = 4, "ur", "UR3e"
    title = "Sorting into boxes by colour"
    model = "Universal Robots UR3e + OnRobot RG2"
    owner = "Pauras"

    def __init__(self, env):
        build_sorting_equipment(env)
        robot = make_ur3e(env, SORT_BASE, Q_UR_HOME)
        self.gripper = RG2Gripper(env, robot)
        self.arm = Arm(self.name, robot, CubeObstacle(env, UR_OBS_SIZE, UR_OBS_MARGIN), 0.0, UR_EPS, UR_STEP,
                       UR_SELF_RADII)
        self.arm.safe_q = Q_UR_HOME.copy()
        self.sensor = LaserGate(env, PICK_X, SORT_SIDE)
        self.home_q = Q_UR_HOME.copy()
        self.final_q = UR_FINAL_DEFAULT.copy()
        self.carried, self.rel = None, None  # car in the gripper and its pose in the TCP frame
        self.counts = {k: 0 for k in TUB_XY}

    # ------------------------------------------------------------------ sorting one car
    def arm_goto(self, p, note, yaw=0.0):
        """TCP at p, tool pointing down, turned by yaw. A few vertical offsets are tried before giving up."""
        arm = self.arm
        for dz in (0.0, 0.04, 0.08, -0.04):
            T = SE3(*(np.asarray(p, float) + [0, 0, dz])) * SE3.Rz(yaw) * SE3.Rx(pi)
            q = ik_safe(arm, T, [arm.q.copy()] + UR_SEEDS)  # never a pose where the arm passes through itself
            if q is not None:
                yield from plan_move(arm, q, note)
                return
        raise RuntimeError(f"{arm.name}: no IK solution for {np.round(p, 3)} - check the layout / reach")

    def gripper_move(self, closing):
        w0, w1 = self.gripper.opening, (GRIP_CLOSED if closing else GRIP_OPEN)
        n = max(int(PICK_DWELL / cfg.DT), 1)
        for k in range(1, n + 1):
            self.gripper.opening = w0 + (w1 - w0) * k / n
            yield

    def sort_cycle(self, car, colour):
        """HOVER over the car -> PICK (RMRC down, close the RG2, RMRC up) -> CARRY over its compartment ->
        PLACE (RMRC down, open the RG2) -> HOME."""
        ur, r = self.arm, self.arm.robot
        p_pick = np.array([float(car.x), 0.0, cfg.BELT_HEIGHT + TCP_Z])
        p_above = p_pick + [0, 0, PICK_HOVER]
        yield from self.arm_goto(p_above, "moving the tool over the car")
        ur.note = "picking: lowering the tool onto the car (RMRC)"
        yield from rmrc(ur, p_pick, UR_LINE_SPEED, TOOL_DOWN)
        ur.note = "closing the RG2 on the car"
        yield from self.gripper_move(True)
        self.rel = linalg.inv(r.fkine(r.q).A) @ car.T  # the car is now held by the tool
        self.carried = car
        yield from rmrc(ur, p_above, UR_LINE_SPEED, TOOL_DOWN)
        tx, ty = TUB_XY[colour]
        rest_z = PAD_HEIGHT + TUB_FLOOR + self.counts[colour] * CAR_SIZE[2]  # cars stack if a tub already has one
        p_over = np.array([tx, ty, TRAVEL_Z])
        p_drop = np.array([tx, ty, rest_z + TCP_Z])
        Rd_box = SE3.Rz(BOX_CAR_YAW).R @ TOOL_DOWN  # wrist turned so the car's long axis lines up with the compartment
        yield from self.arm_goto(p_over, f"carrying {colour} car to the {colour} tub", yaw=BOX_CAR_YAW)
        ur.note = f"placing in the {colour} tub (RMRC)"
        yield from rmrc(ur, p_drop, UR_LINE_SPEED, Rd_box)
        ur.note = "opening the RG2"
        yield from self.gripper_move(False)
        car.T = (SE3(tx, ty, rest_z + CAR_H / 2) * SE3.Rz(BOX_CAR_YAW)).A  # settles on the tub floor / on the car below
        self.carried, self.rel = None, None
        self.counts[colour] += 1
        yield from rmrc(ur, p_over, UR_LINE_SPEED, Rd_box)
        yield from plan_move(ur, self.home_q, "returning home")
        ur.note = "waiting for a car"

    # ------------------------------------------------------------------ production
    def run(self, line):
        for car in line.order:
            line.set_stage(f"{car.colour} car: moving to the end of the belt")
            yield from line.conveyor.move_car_to(car, PICK_X)
            yield from wait_for_sensor(self.arm, self.sensor, LASER_WAIT_S, "the car")  # starts only when ITS laser sees it
            line.say(f"UR3e laser gate: {car.colour} car detected ({self.sensor.range:.3f} m)")
            line.conveyor.detach(car)  # the gripper takes the car off the belt
            line.set_stage(f"{car.colour} car: UR3e sorting")
            yield from self.sort_cycle(car, car.colour)
            line.say(f"UR3e: {car.colour} car sorted into the {car.colour} box")

    # ------------------------------------------------------------------ line services
    def draw(self):
        self.arm.robot.draw()
        if self.carried is not None:
            self.carried.T = self.arm.robot.fkine(self.arm.robot.q).A @ self.rel
        self.gripper.update()

    def set_obstacle(self, on):
        o = self.arm.obstacle
        if not on:
            o.remove()
        elif not o.active:
            o.place(UR_OBSTACLE_P)

    @property
    def sorted(self):
        return sum(self.counts.values())

    def info(self):
        c = self.counts
        return [("Laser gate", self.sensor.text),
                ("Sorted", f"{self.sorted} / {cfg.CARS_IN_JOB}   (B {c['blue']}  R {c['red']}  G {c['green']})"),
                ("Gripper", f"{self.gripper.opening * 1000:.0f} mm" + ("  (holding a car)" if self.carried else "")),
                ("Self-collision check", "on (link radii)")]

    def reset(self):
        self.carried, self.rel = None, None
        self.counts = {k: 0 for k in TUB_XY}
        self.gripper.opening = GRIP_OPEN
        self.arm.note = "idle"
