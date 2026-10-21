"""Safety equipment models for the loading cell: fence, light curtain, beacon, e-stop post.

Everything is placed from the layout in loading_config: the fence encloses the area
the mobile MH5 drives and works in, the only gate is covered by a light curtain, the
beacon is visible from the gate and the e-stop post stands just inside it.
"""

from __future__ import annotations

import spatialgeometry as sg
from spatialmath import SE3

from workcell import loading_config as lc

_HIDE = -5.0
_FENCE = [0.95, 0.55, 0.05, 0.35]
_POST = [0.25, 0.25, 0.28, 1.0]
_YELLOW = [0.95, 0.80, 0.10, 1.0]

# (lamp name) -> (centre z, lit colour, dark colour)
_LAMPS = {
    "red": (1.21, [1.0, 0.1, 0.1, 1.0], [0.25, 0.04, 0.04, 1.0]),
    "amber": (1.13, [1.0, 0.65, 0.0, 1.0], [0.28, 0.18, 0.0, 1.0]),
    "green": (1.05, [0.1, 0.9, 0.2, 1.0], [0.03, 0.22, 0.06, 1.0]),
}
BEACON_XY = (0.68, 1.48)
ESTOP_XY = (-0.68, 1.48)

# Run states that light each lamp.
_RED_STATES = {"ESTOPPED", "FAULT"}
_AMBER_STATES = {"PROTECTIVE", "STOPPED", "BLOCKED"}


class _Switchable:
    """Two shapes at one place; exactly one is visible at a time."""

    def __init__(self, env, make_a, make_b, pose: SE3):
        self.a, self.b, self.pose = make_a(), make_b(), pose
        env.add(self.a)
        env.add(self.b)
        self.state = None
        self.show(True)

    def show(self, first: bool) -> None:
        if first is self.state:
            return
        self.state = first
        hidden = SE3(self.pose.t[0], self.pose.t[1], _HIDE)
        self.a.T = (self.pose if first else hidden).A
        self.b.T = (hidden if first else self.pose).A


class SafetyEquipment:
    """All safety hardware models in the scene."""

    def __init__(self, env):
        self.env = env
        self.shapes: list = []
        self._build_fence()
        self._build_light_curtain()
        self._build_beacon()
        self._build_estop_post()
        self._build_area_scanners()
        self._build_signs_and_markings()

    # ---------------------------------------------------------------- helpers
    def _box(self, size, centre, colour):
        shape = sg.Cuboid(scale=list(size), pose=SE3(*centre), color=list(colour))
        self.env.add(shape)
        self.shapes.append(shape)
        return shape

    # ------------------------------------------------------------------ build
    def _build_fence(self) -> None:
        h, t = lc.FENCE_HEIGHT, lc.FENCE_THICKNESS
        x0, x1 = lc.ZONE_X
        y0, y1 = lc.ZONE_Y
        g0, g1 = lc.GATE_X
        # Back fence along +y with the gate gap, and the two end panels.
        self._box((g0 - x0, t, h), ((x0 + g0) / 2, y1, h / 2), _FENCE)
        self._box((x1 - g1, t, h), ((g1 + x1) / 2, y1, h / 2), _FENCE)
        self._box((t, y1 - y0, h), (x0, (y0 + y1) / 2, h / 2), _FENCE)
        self._box((t, y1 - y0, h), (x1, (y0 + y1) / 2, h / 2), _FENCE)
        for px, py in ((x0, y1), (x1, y1), (g0, y1), (g1, y1), (x0, y0), (x1, y0),
                       ((x0 + g0) / 2, y1), ((g1 + x1) / 2, y1)):
            self._box((0.06, 0.06, h + 0.05), (px, py, (h + 0.05) / 2), _POST)

    def _build_light_curtain(self) -> None:
        g0, g1 = lc.GATE_X
        y = lc.ZONE_Y[1]
        h = lc.CURTAIN_HEIGHT
        for px in (g0, g1):
            self._box((0.07, 0.07, h + 0.2), (px, y, (h + 0.2) / 2), [0.15, 0.15, 0.17, 1.0])
            self._box((0.075, 0.075, 0.05), (px, y, h + 0.2 + 0.025), [1.0, 0.1, 0.1, 1.0])
        width = g1 - g0 - 0.07
        size = (width, 0.01, h)
        self.beam = _Switchable(
            self.env,
            lambda: sg.Cuboid(scale=list(size), color=[0.1, 0.9, 0.2, 0.30]),
            lambda: sg.Cuboid(scale=list(size), color=[1.0, 0.1, 0.1, 0.45]),
            SE3((g0 + g1) / 2, y, 0.10 + h / 2),
        )

    def _build_beacon(self) -> None:
        bx, by = BEACON_XY
        pole = sg.Cylinder(radius=0.025, length=1.0, pose=SE3(bx, by, 0.5), color=_POST)
        self.env.add(pole)
        self.shapes.append(pole)
        self.lamps = {}
        for name, (z, lit, dark) in _LAMPS.items():
            self.lamps[name] = _Switchable(
                self.env,
                lambda lit=lit: sg.Cylinder(radius=0.055, length=0.08, color=lit),
                lambda dark=dark: sg.Cylinder(radius=0.055, length=0.08, color=dark),
                SE3(bx, by, z),
            )
            self.lamps[name].show(False)

    def _build_estop_post(self) -> None:
        ex, ey = ESTOP_XY
        self._box((0.14, 0.14, 0.85), (ex, ey, 0.425), _YELLOW)
        self._box((0.16, 0.16, 0.02), (ex, ey, 0.86), [0.1, 0.1, 0.1, 1.0])
        self.estop_button = _Switchable(
            self.env,
            lambda: sg.Cylinder(radius=0.045, length=0.045, color=[0.9, 0.05, 0.05, 1.0]),
            lambda: sg.Cylinder(radius=0.045, length=0.025, color=[0.45, 0.02, 0.02, 1.0]),
            SE3(ex, ey, 0.895),
        )
        # The "pressed" version sits lower; shift it by changing its own pose.
        self.estop_button.pose = SE3(ex, ey, 0.895)
        self.estop_button.state = None
        self._press_pose = SE3(ex, ey, 0.883)

    def _build_area_scanners(self) -> None:
        """Floor-level laser area scanners at the belt-side corners.

        The belt side of the cell cannot be fenced because the line carries on past it,
        so two scanners watch the open edge. They are what the unsafe-zone input models.
        """
        x0, x1 = lc.ZONE_X
        y0 = lc.ZONE_Y[0]
        for sx, side in ((x0 + 0.1, 1), (x1 - 0.1, -1)):
            self._box((0.10, 0.10, 0.18), (sx, y0 + 0.08, 0.09), _YELLOW)
            self._box((0.11, 0.11, 0.025), (sx, y0 + 0.08, 0.19), [0.1, 0.1, 0.1, 1.0])
            self._box((0.02, 0.06, 0.05), (sx - side * 0.055, y0 + 0.08, 0.12), [1.0, 0.1, 0.1, 1.0])

    def _build_signs_and_markings(self) -> None:
        x0, x1 = lc.ZONE_X
        y0, y1 = lc.ZONE_Y
        g0, g1 = lc.GATE_X
        sign = [0.98, 0.85, 0.05, 1.0]
        black = [0.05, 0.05, 0.05, 1.0]
        for sx in ((x0 + g0) / 2, (g1 + x1) / 2):  # warning signs either side of the gate
            self._box((0.34, 0.012, 0.26), (sx, y1 - 0.02, 0.85), sign)
            self._box((0.06, 0.014, 0.12), (sx, y1 - 0.024, 0.85), black)
        for sy in (y0 + 0.4, y1 - 0.4):  # signs on the end panels
            for ex_, side in ((x0, 1), (x1, -1)):
                self._box((0.012, 0.34, 0.26), (ex_ + side * 0.02, sy, 0.85), sign)
        # Hazard border strips on the floor, just inside the fence.
        w = 0.06
        z = 0.002
        for size, centre in (
            ((x1 - x0, w, 0.004), ((x0 + x1) / 2, y1 - w, z)),
            ((x1 - x0, w, 0.004), ((x0 + x1) / 2, y0 + w / 2, z)),
            ((w, y1 - y0, 0.004), (x0 + w, (y0 + y1) / 2, z)),
            ((w, y1 - y0, 0.004), (x1 - w, (y0 + y1) / 2, z)),
        ):
            self._box(size, centre, sign)

    # ----------------------------------------------------------------- update
    def update(self, run_state: str, estop_pressed: bool, curtain_broken: bool) -> None:
        """Show the beam, beacon and e-stop button for the current station state."""
        self.beam.show(not curtain_broken)
        red = run_state in _RED_STATES or estop_pressed
        amber = not red and (run_state in _AMBER_STATES or curtain_broken)
        green = not red and not amber
        for name, on in (("red", red), ("amber", amber), ("green", green)):
            self.lamps[name].show(bool(on))
        self._set_estop(estop_pressed)

    def _set_estop(self, pressed: bool) -> None:
        b = self.estop_button
        if b.state is pressed:
            return
        b.state = pressed
        ex, ey = ESTOP_XY
        hidden = SE3(ex, ey, _HIDE)
        b.a.T = (hidden if pressed else b.pose).A
        b.b.T = (self._press_pose if pressed else hidden).A
