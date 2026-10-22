from itertools import combinations
from math import pi

import numpy as np
from ir_support import RectangularPrism, line_plane_intersection
from spatialgeometry import Cylinder, Sphere
from spatialmath import SE3
from spatialmath.base import transl

from shared import line_config as cfg
from shared.motion import is_intersection_point_inside_triangle
from shared.scene import PARK, box

RED, GREEN, AMBER = (1.0, 0.05, 0.05, 1.0), (0.1, 0.9, 0.2, 1.0), (1.0, 0.65, 0.0, 1.0)
SENSOR_BLACK, POST_GREY = (0.10, 0.10, 0.12, 1.0), (0.35, 0.35, 0.38, 1.0)
CAR_L, CAR_W, CAR_H = cfg.CAR_SIZE

LASER_Z = cfg.BELT_HEIGHT + 0.020  # beam height: inside the car body, above the belt
LASER_HALF_SPAN = 0.225  # emitter post / receiver post stand this far (m) either side of the belt centre line
LASER_INSET = CAR_L / 2 - 0.005  # the two beams sit this far either side of the stop point (5 mm inside the car's ends)
LASER_WAIT_S = 3.0  # seconds (of running time) an arm waits for its laser before it reports a fault


def ray_hit_prism(p0, p1, vertices, faces, face_normals):
    """Cast the segment p0 -> p1 at a prism. Returns (range from p0, hit point) of the closest face it crosses, or None."""
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


def cars_on_belt_near(cars, x, reach=0.6):
    """Bounding-box prisms of the cars resting on the belt (not in a gripper) within `reach` of x."""
    prisms = []
    for car in cars:
        c = car.T[:3, 3]
        on_belt = (abs(c[0] - x) < reach and abs(c[1]) < cfg.BELT_WIDTH / 2
                   and abs(c[2] - (cfg.BELT_HEIGHT + CAR_H / 2)) < 1.5e-3)
        if on_belt:
            prisms.append((car, RectangularPrism(CAR_L, CAR_W, CAR_H, center=[c[0], c[1], c[2]]).get_data()))
    return prisms


class _Beam:
    """One drawn beam: red while clear, green with a dot where it hits a car while broken."""

    def __init__(self, env, p0, p1):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        span = float(np.linalg.norm(self.p1 - self.p0))
        T = (SE3(*((self.p0 + self.p1) / 2)) * SE3.Rx(pi / 2)).A  # cylinder axis turned from z to y (across the belt)
        self.clear = (Cylinder(radius=0.0025, length=span, color=list(RED)), T)
        self.broken = (Cylinder(radius=0.0025, length=span, color=list(GREEN)), T)
        self.marker = Sphere(radius=0.012, color=list(GREEN))
        for shape in (self.clear[0], self.broken[0], self.marker):
            env.add(shape)
        self.show(None)

    def show(self, hit):
        self.clear[0].T = PARK if hit else self.clear[1]
        self.broken[0].T = self.broken[1] if hit else PARK
        self.marker.T = transl(*hit[1]) if hit else PARK


class LaserGate:
    """Two laser beams across the belt at one robot's stop point (Pauras).  The emitter stands on the robot's side of the
    belt, the receiver on the other.  `detected` is True only while ONE car breaks BOTH beams (with cars only 25 cm apart
    two different cars could otherwise break one beam each and fake a detection, so the test is made per car)."""

    def __init__(self, env, x_stop, side):
        self.x_stop, self.side = x_stop, side
        self.detected, self.range, self.tripped = False, None, False
        self.beams = []
        for dx in (-LASER_INSET, +LASER_INSET):
            x = x_stop + dx
            p0 = (x, side * LASER_HALF_SPAN, LASER_Z)  # emitter (robot side)
            p1 = (x, -side * LASER_HALF_SPAN, LASER_Z)  # receiver (far side)
            for y in (p0[1], p1[1]):  # the two sensor heads on their stands
                box(env, (0.03, 0.03, 0.05), (x, y, LASER_Z), SENSOR_BLACK)
                box(env, (0.02, 0.02, LASER_Z - 0.025), (x, y, (LASER_Z - 0.025) / 2), POST_GREY)
            self.beams.append(_Beam(env, p0, p1))
        self.update([])

    def update(self, cars):
        """Read the gate once (called every frame, before the production logic runs, like a PLC input scan)."""
        per_beam = [None] * len(self.beams)
        detected = False
        for _car, (v, f, n) in cars_on_belt_near(cars, self.x_stop):
            hits = [ray_hit_prism(b.p0, b.p1, v, f, n) for b in self.beams]
            if all(h is not None for h in hits):
                detected = True  # this one car breaks every beam
            for k, h in enumerate(hits):
                if h is not None and (per_beam[k] is None or h[0] < per_beam[k][0]):
                    per_beam[k] = h
        for beam, hit in zip(self.beams, per_beam):
            beam.show(hit)
        ranges = [h[0] for h in per_beam if h is not None]
        self.detected = detected
        self.range = min(ranges) if ranges else None  # the range reading (distance to the car's side)
        self.tripped = bool(ranges)

    @property
    def text(self):
        return (("CAR DETECTED" if self.detected else "car entering" if self.tripped else "clear")
                + (f" ({self.range:.3f} m)" if self.range is not None else ""))


class BeltSlotSensors:
    """Retro-reflective photo-eyes, one per loading slot, on the far (-y) side of the belt with a reflector on the loader's
    side (so nothing tall stands between the loader and the belt).  Each has an indicator lamp on top of its head: red =
    slot empty, green = car present.  `present[k]` is the reading of slot k; `car_at[k]` is the car that breaks it."""

    LAMP_OFF = (0.25, 0.25, 0.25, 1.0)

    def __init__(self, env, slot_xs=cfg.BELT_SLOT_XS, side=+1):
        self.slot_xs = tuple(slot_xs)
        self.present = [False] * len(self.slot_xs)
        self.car_at = [None] * len(self.slot_xs)
        self.beams, self.lamps = [], []
        z = cfg.BELT_HEIGHT + 0.012  # just above the belt: the beam is broken by any car standing in the slot
        for x in self.slot_xs:
            p0 = (x, -side * LASER_HALF_SPAN, z)  # sensor head (emitter + receiver), far side
            p1 = (x, side * (cfg.BELT_WIDTH / 2 + 0.012), z)  # reflector on the near rail
            box(env, (0.035, 0.03, 0.045), (x, p0[1], z), SENSOR_BLACK)  # head
            box(env, (0.02, 0.02, z - 0.0225), (x, p0[1], (z - 0.0225) / 2), POST_GREY)  # stand
            box(env, (0.04, 0.004, 0.03), (x, p1[1] + side * 0.002, z), (0.95, 0.20, 0.10, 1.0))  # red reflector
            lamp_T = transl(x, p0[1], z + 0.032)
            lamps = {}
            for key, colour in (("off", RED), ("on", GREEN)):
                lamp = Sphere(radius=0.011, color=list(colour))
                env.add(lamp)
                lamps[key] = (lamp, lamp_T)
            self.lamps.append(lamps)
            self.beams.append(_Beam(env, p0, p1))
        self.update([])

    def update(self, cars):
        for k, (x, beam, lamps) in enumerate(zip(self.slot_xs, self.beams, self.lamps)):
            hit, who = None, None
            for car, (v, f, n) in cars_on_belt_near(cars, x, reach=0.2):
                h = ray_hit_prism(beam.p0, beam.p1, v, f, n)
                if h is not None and (hit is None or h[0] < hit[0]):
                    hit, who = h, car
            beam.show(hit)
            self.present[k], self.car_at[k] = hit is not None, who
            lamps["on"][0].T = lamps["on"][1] if hit else PARK
            lamps["off"][0].T = PARK if hit else lamps["off"][1]

    @property
    def all_present(self):
        return all(self.present)

    @property
    def text(self):
        return "  ".join(f"slot {k + 1}: {'CAR' if p else 'empty'}" for k, p in enumerate(self.present))
