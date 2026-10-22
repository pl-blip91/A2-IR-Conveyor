import numpy as np
import spatialgeometry as sg
from roboticstoolbox import trapezoidal
from spatialmath import SE3

from shared import compat  # noqa: F401
from shared import line_config as cfg


class Conveyor:
    def __init__(self, env):
        L, W, H, t = cfg.BELT_LENGTH, cfg.BELT_WIDTH, cfg.BELT_HEIGHT, cfg.BELT_THICK
        self.length, self.top = L, H
        self.s = 0.0
        self.riders = {}  # id(car) -> (car, position of the car along the belt when s = 0)
        self.stripes = []

        def box(size, centre, colour):
            shape = sg.Cuboid(scale=list(size), pose=SE3(*centre), color=list(colour))
            env.add(shape)
            return shape

        box((L, W, t), (L / 2, 0.0, H - t / 2), cfg.BELT_COLOUR)  # the belt surface
        for side in (-1, 1):  # side rails
            box((L, 0.02, 0.07), (L / 2, side * (W / 2 + 0.01), H - 0.03), cfg.BELT_FRAME_COLOUR)
        leg_h = H - t
        for lx in np.linspace(0.3, L - 0.3, 7):  # legs, tucked under the belt so the loader's base can park close
            for side in (-1, 1):
                box((0.05, 0.05, leg_h), (lx, side * (W / 2 - 0.06), leg_h / 2), cfg.BELT_FRAME_COLOUR)
        for side in (-1, 1):  # two cross-members tie the legs together
            box((L - 0.6, 0.03, 0.03), (L / 2, side * (W / 2 - 0.06), 0.25), cfg.BELT_FRAME_COLOUR)
        n = int(round(L / cfg.STRIPE_PITCH))
        for i in range(n):  # stripes that slide along when the belt runs (purely visual)
            stripe = sg.Cuboid(scale=[0.02, W * 0.9, 0.002], color=list(cfg.STRIPE_COLOUR))
            env.add(stripe)
            self.stripes.append((stripe, i * cfg.STRIPE_PITCH))
        self.set_s(0.0)

    # ------------------------------------------------------------------ riders
    def attach(self, car):
        """The car now rides the belt where it stands."""
        self.riders[id(car)] = (car, car.x - self.s)

    def detach(self, car):
        self.riders.pop(id(car), None)

    def clear(self):
        """Nothing rides the belt any more (used when the line is re-armed)."""
        self.riders.clear()

    def rides(self, car) -> bool:
        return id(car) in self.riders

    def set_s(self, s):
        """Put the belt at travelled distance s: riders and stripes follow."""
        self.s = float(s)
        for car, ref in self.riders.values():
            car.T = SE3(ref + self.s, 0.0, self.top + cfg.CAR_SIZE[2] / 2).A
        for stripe, offset in self.stripes:
            sx = (offset + self.s) % self.length
            stripe.T = SE3(sx, 0.0, self.top + 0.001).A

    # ------------------------------------------------------------------ motion
    def profile(self, dist, speed=cfg.BELT_SPEED):
        """Belt distances s, one per frame, for a trapezoidal move of `dist` from the current position."""
        n = max(int(np.ceil(abs(dist) / (speed * cfg.DT))), 2)
        prof = np.asarray(trapezoidal(0, 1, n).q).flatten()
        return self.s + prof * dist

    def advance(self, dist, speed=cfg.BELT_SPEED):
        """Generator: run the belt by `dist` metres (one frame per yield)."""
        if abs(dist) < 1e-9:
            return
        s_end = self.s + dist
        for s in self.profile(dist, speed):
            self.set_s(s)
            yield
        self.set_s(s_end)

    def move_car_to(self, car, x_goal, speed=cfg.BELT_SPEED):
        """Generator: run the belt until `car` (and so every other car) has travelled to x_goal."""
        yield from self.advance(x_goal - car.x, speed)
