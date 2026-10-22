"""
    Robot 1  load   (Lachlan)  Yaskawa MH5 on a TurtleBot3, +y side, parks on a floor dock
    Robot 2  paint  (Aarav)    Yaskawa MPX3500 on its table, +y side, 1.8 m back from the belt
    Robot 3  stamp  (Pauras)   FANUC M-10iA stamping the roof, +y side
    Robot 4  sort   (Pauras)   UR3e dropping each car into the box of its own colour, -y side
"""

import math
import os
from dataclasses import dataclass

from spatialmath import SE3

SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SHARED_DIR)
ASSETS = os.path.join(SHARED_DIR, "assets")

# One simulation step for every station (s). All speeds below are in real units, so they do not depend on this.
DT = 0.05

# --- Conveyor (Aarav's belt: thin belt on legs, side rails, moving stripes) ------------------------------------------
BELT_LENGTH = 6.4
BELT_WIDTH = 0.40
BELT_HEIGHT = 0.60  # top surface of the belt
BELT_THICK = 0.05
BELT_SPEED = 0.20  # m/s, mean speed of a belt move (Aarav's value)
BELT_COLOUR = (0.12, 0.12, 0.12, 1.0)
BELT_FRAME_COLOUR = (0.60, 0.60, 0.65, 1.0)
STRIPE_PITCH = 0.20
STRIPE_COLOUR = (0.35, 0.35, 0.35, 1.0)

# --- Robot pads / dock -------------------------------------------------------------------------------------------------
PAD_SIZE = (0.5, 0.5, 0.6)  # top of a pad is where a robot base sits
PAD_HEIGHT = PAD_SIZE[2]
PAD_OFFSET_Y = 0.5
PAD_COLOUR = (0.55, 0.55, 0.58, 1.0)
DOCK_OFFSET_Y = 0.45  # the mobile loader parks on a floor marking this far from the belt centre line
DOCK_SIZE = (0.5, 0.5, 0.014)  # painted dock marking (stands just proud of the factory floor slab)
DOCK_COLOUR = (0.95, 0.80, 0.10, 1.0)

# --- Toy cars (the 'Incredible Jaban' STL, 15 x 7.5 x 4 cm, nose along the belt) ---------------------------------------
CAR_FILE = os.path.join(ASSETS, "Incredible_Jaban_m.stl")
CAR_SIZE = (0.15, 0.075, 0.04)  # length (along the belt), width, height
PRIMER = (0.70, 0.70, 0.72, 1.0)  # unpainted
CAR_ORDER = ("blue", "red", "green")  # the order the cars go through the whole line
CAR_COLOURS = {
    "blue": (0.10, 0.25, 0.95, 1.0),
    "red": (0.90, 0.10, 0.10, 1.0),
    "green": (0.10, 0.75, 0.20, 1.0),
}
CAR_PITCH = 0.25  # spacing of the cars on the belt (the loader's slot pitch)
CARS_IN_JOB = len(CAR_ORDER)


@dataclass(frozen=True)
class Station:
    """One station on the line and the pad reserved for its robot."""

    key: str
    title: str
    x: float  # position along the belt
    side: int  # +1: robot on the +y side of the belt, -1: on the -y side
    robot: str
    mobile: bool = False

    @property
    def robot_base(self) -> SE3:
        """Pose for a robot standing on this pad (or parked on the dock), facing the belt."""
        if self.mobile:
            return SE3(self.x, self.side * DOCK_OFFSET_Y, 0.0) * SE3.Rz(-self.side * math.pi / 2)
        return SE3(self.x, self.side * PAD_OFFSET_Y, PAD_HEIGHT) * SE3.Rz(-self.side * math.pi / 2)


LOAD_X = 0.95
PAINT_X = BELT_LENGTH / 3
STAMP_X = 2 * BELT_LENGTH / 3
SORT_X = BELT_LENGTH
PAINT_SIDE, STAMP_SIDE, SORT_SIDE = +1, +1, -1

STATIONS = (
    Station("load", "Loading", LOAD_X, +1, "Yaskawa Motoman MH5 on TurtleBot3 Waffle", mobile=True),
    Station("paint", "Spray painting", PAINT_X, PAINT_SIDE, "Yaskawa Motoman MPX3500"),
    Station("label", "Logo / label", STAMP_X, STAMP_SIDE, "FANUC M-10iA"),
    Station("sort", "Sorting", SORT_X, SORT_SIDE, "UR3e"),
)


def station(key: str) -> Station:
    for s in STATIONS:
        if s.key == key:
            return s
    raise KeyError(key)


# --- Belt slot sensors (how the conveyor knows the loader has put a car down) ------------------------------------------
# One retro-reflective photo-eye per loading slot, looking across the belt just above its surface.  Slot k is where the
# loader puts car k (the same numbers as loading_config.BELT_SLOT_XS).
BELT_SLOT_XS = tuple(LOAD_X + (1 - k) * CAR_PITCH for k in range(CARS_IN_JOB))

# --- MPX3500 paint cell (Aarav's numbers; the robot is far too big for a 0.5 m pad) -----------------------------------
# Aarav tuned his IK and spray paths with the robot axis 1.80 m from the belt centre line and the belt top 0.05 m above
# the robot's table top.  Keeping those two numbers means every one of his joint solutions is still valid here.
PAINT_BELT_DIST = 1.80
PAINT_TABLE_SIZE = 1.20
PAINT_TABLE_TOP = BELT_HEIGHT - 0.05
PAINT_BASE = SE3(PAINT_X, PAINT_SIDE * PAINT_BELT_DIST, PAINT_TABLE_TOP) * SE3.Rz(-PAINT_SIDE * math.pi / 2)

# Full horizontal reach of the MPX3500 from its S (base) axis, spray can included: L-arm 1.264 m + forearm 1.443 m
# (a3 = 0.056, d4 = 1.442) + flange 0.136 m + spray tool 0.150 m = 2.993 m (checked numerically against the DH model
# at start-up, see PaintStation.max_reach()).  The guard fence round the robot is drawn at least this far plus a clearance
# from the S axis in EVERY direction, so the arm cannot reach the fence even fully stretched out.
PAINT_REACH = 2.993
PAINT_FENCE_CLEARANCE = 0.30
PAINT_GUARD_R = PAINT_REACH + PAINT_FENCE_CLEARANCE  # apothem of the octagonal enclosure round the S axis (3.29 m)

# --- Safety cell ---------------------------------------------------------------------------------------------------------
FENCE_Y = 1.5  # perimeter fence stands this far from the belt centre line on both sides
FENCE_END = 1.5
DOOR_W = 1.06  # sorting door (-y side, beside the UR3e)
FENCE_X1 = SORT_X + DOOR_W / 2 + FENCE_END
# The MPX3500 enclosure reaches upstream to PAINT_X - PAINT_GUARD_R (about -1.16 m), so the loading gate (which must open
# from OUTSIDE the cell, not into the paint enclosure) sits upstream of it and the cell is extended at that end.
GATE_X = (-2.65, -1.65)  # loading gate in the +y fence (Lachlan)
FENCE_X0 = GATE_X[0] - 0.75
# Octagonal paint-cell enclosure on the +y side: (x, y) corners from where it leaves the main +y fence to where it rejoins
_K = PAINT_GUARD_R * (math.sqrt(2.0) - 1.0)  # half-length of each octagon side
_PCX, _PCY = PAINT_X, PAINT_SIDE * PAINT_BELT_DIST
PAINT_GUARD = (
    (_PCX - PAINT_GUARD_R, FENCE_Y),
    (_PCX - PAINT_GUARD_R, _PCY + _K),
    (_PCX - _K, _PCY + PAINT_GUARD_R),
    (_PCX + _K, _PCY + PAINT_GUARD_R),
    (_PCX + PAINT_GUARD_R, _PCY + _K),
    (_PCX + PAINT_GUARD_R, FENCE_Y),
)
PAINT_BULGE_X = (_PCX - PAINT_GUARD_R, _PCX + PAINT_GUARD_R)
PAINT_BULGE_Y = _PCY + PAINT_GUARD_R

# Guarded floor area watched by the loading station's area scanners (Lachlan's unsafe zone): from the loading gate to
# the belt slots.  It stops 0.25 m inside the fence, so a person coming through the gate breaks the light curtain first.
LOAD_ZONE_X = (FENCE_X0 + 0.15, 1.7)
LOAD_ZONE_Y = (0.25, FENCE_Y - 0.25)

# The two ways into the cell.  A person walks in at the door's x, from the outside to the inside along y.
DOORS = {
    "load": dict(name="loading gate", x=0.5 * (GATE_X[0] + GATE_X[1]), side=+1, half_w=0.5 * (GATE_X[1] - GATE_X[0])),
    "sort": dict(name="sorting door", x=SORT_X, side=SORT_SIDE, half_w=DOOR_W / 2),
}

# --- Camera: the whole cell (including the paint enclosure) in view -------------------------------------------------------
CAMERA_POSITION = [2.4, -7.4, 5.6]
CAMERA_LOOK_AT = [2.4, 1.4, 0.5]
