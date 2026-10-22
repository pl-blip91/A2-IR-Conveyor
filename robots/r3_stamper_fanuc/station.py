from math import pi

import numpy as np
from spatialgeometry import Cuboid, Cylinder
from spatialmath import SE3
from spatialmath.base import transl

from shared import line_config as cfg
from shared.motion import Arm, CubeObstacle, ik, plan_move, rmrc, wait, wait_for_sensor
from shared.scene import box
from shared.sensors import LASER_WAIT_S, LaserGate
from shared.station_base import RobotStation

from .fanuc_m10ia import make_fanuc

STATION_X = cfg.STAMP_X  # where the car stops to be stamped
STATION_SIDE = cfg.STAMP_SIDE  # FANUC pad is on the +y side of the belt
ROBOT_BASE = SE3(STATION_X, STATION_SIDE * cfg.PAD_OFFSET_Y, cfg.PAD_HEIGHT) * SE3.Rz(-STATION_SIDE * pi / 2)

# Home pose: J1 = +90 deg turns the arm 90 deg to the robot's left (positive = anticlockwise about the base z axis).
# q5 = 0 would be a wrist singularity, so home uses q5 = 60 deg.
Q_HOME = np.deg2rad([90, 30, 30, 0, 60, 0])
FANUC_FINAL_DEFAULT = np.deg2rad([35, 25, 35, 0, 70, 0])
FANUC_SEEDS = [np.deg2rad(g) for g in ([0, 30, 30, 0, 60, 0], [0, 45, 0, 0, 45, 0], [0, 20, 60, 0, 40, 0],
                                       [0, 60, -20, 0, 50, 0], [90, 30, 30, 0, 60, 0], [-90, 30, 30, 0, 60, 0],
                                       [60, 30, 30, 0, 60, 0], [-60, 30, 30, 0, 60, 0])]
TOOL_DOWN = SE3.Rx(pi).R  # Lab 4 Q1: a rotation of pi about x turns the tool z axis downward
FANUC_EPS = 0.01  # manipulability below which the FANUC RMRC damping switches on
FANUC_STEP = np.deg2rad(1.5) * cfg.DT / 0.03  # largest joint step per frame (same joint speed as at a dt of 0.03 s)
FANUC_OBS_SIZE, FANUC_OBS_MARGIN = 0.16, 0.05  # test obstacle: cube side and clearance kept round it (m)
OBSTACLE_Y = 0.12  # the test obstacle sits on the belt, to the side of the car

# Stamping
TOOL_LEN = 0.10  # stamp die length beyond the FANUC flange (m)
STAMP_HOVER = 0.05
STAMP_DWELL = 0.3
PLUNGE_SPEED = 0.10  # m/s
LETTER_H = 0.038
LETTER_W = {"I": 0.024, "R": 0.030}
LETTER_GAP = 0.012
STROKE_W = 0.005
STROKE_COLOUR = (0.04, 0.04, 0.07, 1.0)
TEXT = "IR"
TEXT_W = sum(LETTER_W[c] for c in TEXT) + LETTER_GAP * (len(TEXT) - 1)


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


def letter_strokes(stamp_x, stamp_y=0.0):
    """World-space (x, y) polylines that make up TEXT, centred on the die at (stamp_x, stamp_y)."""
    H = LETTER_H
    shapes = {
        "I": lambda w: [[(0, H), (w, H)], [(w / 2, H), (w / 2, 0)], [(0, 0), (w, 0)]],
        "R": lambda w: [[(0, 0), (0, H), (0.6 * w, H), (w, 0.82 * H), (w, 0.68 * H), (0.6 * w, 0.5 * H), (0, 0.5 * H)],
                        [(0.4 * w, 0.5 * H), (w, 0)]],
    }
    x0, y0 = stamp_x - TEXT_W / 2, stamp_y - H / 2
    strokes = []
    for c in TEXT:
        w = LETTER_W[c]
        for poly in shapes[c](w):
            strokes.append([(x0 + u, y0 + v) for u, v in poly])
        x0 += w + LETTER_GAP
    return strokes


class StampingStation(RobotStation):
    number, key, name = 3, "fanuc", "FANUC"
    title = "Stamping the 'IR' logo"
    model = "FANUC M-10iA"
    owner = "Pauras"

    def __init__(self, env, profile):
        self.env = env
        box(env, cfg.PAD_SIZE, (STATION_X, STATION_SIDE * cfg.PAD_OFFSET_Y, cfg.PAD_HEIGHT / 2), cfg.PAD_COLOUR)
        robot = make_fanuc(env, ROBOT_BASE, Q_HOME)
        self.tool = StampTool(env, robot)
        self.arm = Arm(self.name, robot, CubeObstacle(env, FANUC_OBS_SIZE, FANUC_OBS_MARGIN), TOOL_LEN, FANUC_EPS,
                       FANUC_STEP)
        self.sensor = LaserGate(env, STATION_X, STATION_SIDE)
        self.home_q = Q_HOME.copy()
        self.final_q = FANUC_FINAL_DEFAULT.copy()
        # the flattest part of the Jaban's roof (car-body coordinates) and the height the die touches there
        self.stamp_dx, self.stamp_dy, roof = profile.best_stamp_spot(TEXT_W + 0.01, LETTER_H + 0.01)
        self.stamp_roof = cfg.BELT_HEIGHT + roof
        self.stamped = 0

    # ------------------------------------------------------------------ stamping one car
    def add_stroke(self, car, a, b, roof_z):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length, mid, thick = float(np.linalg.norm(b - a)), (a + b) / 2, 0.003
        angle = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
        pose = SE3(mid[0], mid[1], roof_z + thick / 2) * SE3.Rz(angle)
        stroke = Cuboid(scale=[length + STROKE_W, STROKE_W, thick], color=list(STROKE_COLOUR))
        self.env.add(stroke)
        car.attach_mark("stamp", stroke, pose)  # the logo travels with the car

    def stamp_cycle(self, car):
        """HOVER over the roof -> PRESS (RMRC down) -> DWELL -> LIFT (RMRC up) -> HOME."""
        arm = self.arm
        x_s, y_s = car.x + self.stamp_dx, self.stamp_dy  # the flattest part of this car's roof
        roof_z = self.stamp_roof
        z_hover, z_press = roof_z + TOOL_LEN + STAMP_HOVER, roof_z + TOOL_LEN
        q_hover = ik(arm.robot, SE3(x_s, y_s, z_hover) * SE3.Rx(pi), FANUC_SEEDS)
        if q_hover is None:
            raise RuntimeError("FANUC: no IK solution above the car - check base pose / joint limits")
        yield from plan_move(arm, q_hover, "moving to the hover pose")
        arm.note = "pressing the IR die onto the roof (RMRC)"
        yield from rmrc(arm, [x_s, y_s, z_press], PLUNGE_SPEED, TOOL_DOWN)  # down onto the roof
        for stroke in letter_strokes(x_s, y_s):  # letters centred on the die at the moment of the press
            for a, b in zip(stroke[:-1], stroke[1:]):
                self.add_stroke(car, a, b, roof_z)
        yield from wait(STAMP_DWELL)
        arm.note = "lifting away (RMRC)"
        yield from rmrc(arm, [x_s, y_s, z_hover], PLUNGE_SPEED, TOOL_DOWN)  # lift away
        yield from plan_move(arm, self.home_q, "returning home")
        arm.note = "idle (home)"

    # ------------------------------------------------------------------ production
    def run(self, line):
        for car in line.order:
            line.set_stage(f"{car.colour} car: moving to the FANUC")
            yield from line.conveyor.move_car_to(car, STATION_X)
            yield from wait_for_sensor(self.arm, self.sensor, LASER_WAIT_S, "the car")  # starts only when ITS laser sees it
            line.say(f"FANUC laser gate: {car.colour} car detected ({self.sensor.range:.3f} m)")
            line.set_stage(f"{car.colour} car: FANUC stamping")
            yield from self.stamp_cycle(car)
            self.stamped += 1
            line.say(f"FANUC: {car.colour} car stamped")

    # ------------------------------------------------------------------ line services
    def draw(self):
        self.arm.robot.draw()
        self.tool.update()

    def set_obstacle(self, on):
        o = self.arm.obstacle
        if not on:
            o.remove()
        elif not o.active:
            o.place([STATION_X, STATION_SIDE * OBSTACLE_Y, cfg.BELT_HEIGHT + FANUC_OBS_SIZE / 2])

    def info(self):
        return [("Laser gate", self.sensor.text),
                ("Cars stamped", f"{self.stamped} / {cfg.CARS_IN_JOB}"),
                ("Home J1", f"{np.degrees(self.home_q[0]):.0f} deg (turned left, along the belt)")]

    def reset(self):
        self.stamped = 0
        self.arm.note = "idle"
