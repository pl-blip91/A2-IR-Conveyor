import numpy as np
from roboticstoolbox import jtraj
from spatialgeometry import Cuboid

from shared import line_config as cfg
from shared.motion import Arm, CubeObstacle, apply, wait_for_sensor
from shared.sensors import LASER_WAIT_S, LaserGate
from shared.station_base import RobotStation

from .paint_station import PaintStation

MPX_EPS = 0.05  # the MPX3500 is a very large robot, so its manipulability values are large
MPX_STEP = np.deg2rad(1.2)  # largest joint step per frame in MPX3500 joint moves
MPX_OBS_SIZE, MPX_OBS_MARGIN = 0.16, 0.05  # test obstacle: cube side and clearance kept round it (m)
MARK_EVERY = 2  # one wet-paint dot every N frames of spraying
FINAL_BELT_RUN = 1.0  # after the last car is painted the belt carries all three on this far (Aarav's final run)


class PaintingStation(RobotStation):
    number, key, name = 2, "mpx", "MPX3500"
    title = "Spray painting"
    model = "Yaskawa Motoman MPX3500"
    owner = "Aarav"

    def __init__(self, env, profile):
        self.env = env
        self.paint = PaintStation(env, profile)
        self.arm = Arm(self.name, self.paint.robot, CubeObstacle(env, MPX_OBS_SIZE, MPX_OBS_MARGIN),
                       self.paint.tool_len, MPX_EPS, MPX_STEP)
        self.sensor = LaserGate(env, cfg.PAINT_X, cfg.PAINT_SIDE)
        self.home_q = self.paint.home_pose.copy()
        self.final_q = self.paint.home_pose.copy()
        self.spray_idx = None  # which car colour is being sprayed (None = spray off)
        self.spray_live = False  # the line is RUNNING (the cell sets this every frame), so the spray is really on
        self.painted = 0
        self.plan = None
        self.reach = None

    def prepare(self):
        """Slow set-up done once before the line goes live: plan the spray passes (a few seconds of IK) and measure the
        arm's reach so the guard fence can be checked to be out of it."""
        self.plan = self.paint.plan()
        self.reach = self.paint.max_reach()
        ok = self.reach <= cfg.PAINT_REACH + 1e-3
        print(f"MPX3500 full reach {self.reach:.3f} m from the S axis; guard fence {cfg.PAINT_GUARD_R:.2f} m away "
              f"-> {'OK, fence is out of reach' if ok else 'WARNING: reach exceeds line_config.PAINT_REACH'}")

    # ------------------------------------------------------------------ painting one car
    def paint_pass(self, k, car, Q):
        """Follow one planned spray pass; the arm is held (spray off) while an obstacle is in the way."""
        self.spray_idx = k
        for i, q in enumerate(Q):
            yield from apply(self.arm, q)
            if i % MARK_EVERY == 0 and not self.arm.blocked:
                dot = Cuboid(scale=[0.010, 0.010, 0.001], color=list(cfg.CAR_COLOURS[car.colour][:3]) + [1.0])
                self.env.add(dot)
                car.attach_mark("paint", dot, self.paint.mark_pose(car.T, self.paint.tip_position()))
        self.spray_idx = None

    def paint_cycle(self, k, car):
        plan, arm = self.plan, self.arm
        arm.note = f"moving to the spray start point for the {car.colour} car"
        for q in jtraj(arm.q, plan.q_start, 60).q:
            yield from apply(arm, q)
        arm.note = f"spraying the {car.colour} car: forward sine pass"
        yield from self.paint_pass(k, car, plan.Q_fwd)
        arm.note = f"spraying the {car.colour} car: return sine pass"
        yield from self.paint_pass(k, car, plan.Q_bwd)
        car.set_painted(True)  # the paint has 'dried': the car takes on its final colour
        self.painted += 1
        arm.note = f"the {car.colour} car is painted"

    def home_and_advance(self, conveyor, dist):
        """The arm returns to its default pose while, at the same time, the belt runs `dist` (Aarav's step 4)."""
        arm = self.arm
        arm.note = "returning home while the belt moves"
        Q_home = jtraj(arm.q, self.home_q, 70).q
        s_end = conveyor.s + dist
        s_list = conveyor.profile(dist)
        for i in range(max(len(Q_home), len(s_list))):
            if i < len(s_list):
                conveyor.set_s(s_list[i])
            yield from apply(arm, Q_home[i] if i < len(Q_home) else self.home_q)
        conveyor.set_s(s_end)
        arm.note = "idle (home)"

    # ------------------------------------------------------------------ production
    def run(self, line):
        cv, order = line.conveyor, line.order
        line.set_stage("painting: belt bringing the blue car to the paint station")
        yield from cv.move_car_to(order[0], cfg.PAINT_X)
        for k, car in enumerate(order):
            line.set_stage(f"{car.colour} car: waiting for the MPX3500 laser to see it")
            yield from wait_for_sensor(self.arm, self.sensor, LASER_WAIT_S, "the car")
            line.say(f"MPX3500 laser gate: {car.colour} car detected ({self.sensor.range:.3f} m)")
            line.set_stage(f"{car.colour} car: MPX3500 spraying")
            yield from self.paint_cycle(k, car)
            line.say(f"MPX3500: {car.colour} car painted")
            if k < len(order) - 1:  # arm goes home while the belt brings the next car forward
                line.set_stage(f"{car.colour} car done: arm homing, belt bringing the {order[k + 1].colour} car forward")
                yield from self.home_and_advance(cv, cfg.PAINT_X - order[k + 1].x)
            else:  # ... and after the last car the belt carries all three on down the line
                line.set_stage("all cars painted: arm homing, belt carrying the cars down the line")
                yield from self.home_and_advance(cv, FINAL_BELT_RUN)

    # ------------------------------------------------------------------ line services
    def draw(self):
        self.paint.draw(self.spray_idx if (self.spray_live and not self.arm.blocked) else None)

    def set_obstacle(self, on):
        o = self.arm.obstacle
        if not on:
            o.remove()
        elif not o.active:  # across the spray path, round the nozzle, above the car
            o.place([cfg.PAINT_X, 0.0, cfg.BELT_HEIGHT + cfg.CAR_SIZE[2] + self.paint.spray_standoff - 0.02])

    def info(self):
        tip = self.paint.tip_position()
        return [("Laser gate", self.sensor.text),
                ("Cars painted", f"{self.painted} / {cfg.CARS_IN_JOB}"),
                ("Spray", cfg.CAR_ORDER[self.spray_idx] if self.spray_idx is not None else "off"),
                ("Nozzle tip", "{:.3f}, {:.3f}, {:.3f} m".format(*tip)),
                ("Reach / fence", f"{self.reach:.2f} m / {cfg.PAINT_GUARD_R:.2f} m" if self.reach else "-")]

    def reset(self):
        self.spray_idx = None
        self.painted = 0
        self.arm.note = "idle"
