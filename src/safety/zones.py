"""Simulated safety inputs: a person, a light curtain and the guarded-cell zone."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import spatialgeometry as sg
from spatialmath import SE3

from workcell import loading_config as lc

_HIDE_Z = -5.0


@dataclass
class Person:
    """A person who can walk towards a target point at constant speed."""

    x: float = lc.PERSON_OUTSIDE[0]
    y: float = lc.PERSON_OUTSIDE[1]
    radius: float = lc.PERSON_RADIUS
    speed: float = lc.PERSON_SPEED
    target: tuple = field(default=lc.PERSON_OUTSIDE)
    body: sg.Cylinder = field(default=None, repr=False)
    head: sg.Sphere = field(default=None, repr=False)

    def add_to_env(self, env) -> None:
        self.body = sg.Cylinder(radius=0.18, length=1.35, color=[0.95, 0.45, 0.05, 1.0])
        self.head = sg.Sphere(radius=0.11, color=[0.85, 0.7, 0.55, 1.0])
        env.add(self.body)
        env.add(self.head)
        self._place()

    def walk_to(self, x: float, y: float) -> None:
        self.target = (float(x), float(y))

    def walk_in(self) -> None:
        self.walk_to(*lc.PERSON_INSIDE)

    def walk_out(self) -> None:
        self.walk_to(*lc.PERSON_OUTSIDE)

    @property
    def moving(self) -> bool:
        return math.hypot(self.target[0] - self.x, self.target[1] - self.y) > 1e-6

    def update(self, dt: float) -> None:
        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 1e-9:
            step = min(dist, self.speed * dt)
            self.x += dx / dist * step
            self.y += dy / dist * step
            self._place()

    def _place(self) -> None:
        if self.body is not None:
            self.body.T = SE3(self.x, self.y, 0.675).A
            self.head.T = SE3(self.x, self.y, 1.45).A


@dataclass(frozen=True)
class LightCurtain:
    """A vertical beam plane across the gate in the +y fence."""

    x_min: float = lc.GATE_X[0]
    x_max: float = lc.GATE_X[1]
    y: float = lc.ZONE_Y[1]

    def broken_by(self, person: Person) -> bool:
        """True if the person's footprint crosses the beam line."""
        cx = min(max(person.x, self.x_min), self.x_max)
        return math.hypot(person.x - cx, person.y - self.y) < person.radius


@dataclass(frozen=True)
class UnsafeZone:
    """The guarded floor area around the loading station."""

    x_min: float = lc.ZONE_X[0]
    x_max: float = lc.ZONE_X[1]
    y_min: float = lc.ZONE_Y[0]
    y_max: float = lc.ZONE_Y[1]

    def occupied_by(self, person: Person) -> bool:
        """True if the person's footprint overlaps the guarded area."""
        cx = min(max(person.x, self.x_min), self.x_max)
        cy = min(max(person.y, self.y_min), self.y_max)
        return math.hypot(person.x - cx, person.y - cy) < person.radius


class SafetyInputs:
    """Evaluates the simulated safety sensors each tick.

    forced_curtain lets an operator (or a test) trip the light curtain
    directly, without moving the person.
    """

    def __init__(self, person: Person | None = None):
        self.person = person or Person()
        self.curtain = LightCurtain()
        self.zone = UnsafeZone()
        self.forced_curtain = False
        self.curtain_broken = False
        self.zone_occupied = False

    def update(self, dt: float) -> bool:
        """Advance the person and re-evaluate the sensors; True if any are tripped."""
        self.person.update(dt)
        self.curtain_broken = self.forced_curtain or self.curtain.broken_by(self.person)
        self.zone_occupied = self.zone.occupied_by(self.person)
        return self.tripped

    @property
    def tripped(self) -> bool:
        return self.curtain_broken or self.zone_occupied
