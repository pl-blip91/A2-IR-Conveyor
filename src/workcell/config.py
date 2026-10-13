"""Layout constants for the Toy Car Production workcell (initial sketch).

All lengths are in metres. The world frame has +x along the conveyor, +y across
the conveyor and +z up, with the floor at z = 0.
"""

from dataclasses import dataclass

import numpy as np
from spatialmath import SE3

# --- Conveyor (one box) -----------------------------------------------------
BELT_LENGTH = 6.4
BELT_WIDTH = 0.40
BELT_HEIGHT = 0.60  # the belt is a box from the floor up to this height

# --- Robot pads (one box per station) ---------------------------------------
PAD_SIZE = (0.5, 0.5, 0.6)  # the top of a pad is where a robot base sits
PAD_HEIGHT = PAD_SIZE[2]
PAD_OFFSET_Y = 0.5  # distance of each pad from the belt centre line
PAD_COLOUR = (0.55, 0.55, 0.58, 1.0)  # all pads share one colour

# --- Mobile robot dock (loading station) ------------------------------------
# A mobile station has no pad: its robot drives on the floor and parks on a dock
# marking this far from the belt centre line.
DOCK_OFFSET_Y = 0.45
DOCK_SIZE = (0.5, 0.5, 0.004)
DOCK_COLOUR = (0.95, 0.80, 0.10, 1.0)

# --- Toy cars (one box each) ------------------------------------------------
CAR_SIZE = (0.20, 0.10, 0.06)
UNPAINTED = (0.7, 0.7, 0.7, 1.0)
CAR_COLOURS = {
    "red": (0.85, 0.12, 0.12, 1.0),
    "green": (0.12, 0.65, 0.22, 1.0),
    "blue": (0.15, 0.35, 0.90, 1.0),
}


@dataclass(frozen=True)
class Station:
    """One station on the line and the pad reserved for its robot."""

    key: str
    title: str
    x: float  # position along the belt
    side: int  # +1: pad on the +y side of the belt, -1: on the -y side
    robot: str  # the robot planned for this pad (description only)
    mobile: bool = False  # True: the robot drives on the floor and parks on a dock

    @property
    def robot_base(self) -> SE3:
        """Pose for a robot standing on this pad (or parked on the dock), facing the belt.

        For a mobile station this is the floor pose of the mobile base (z = 0).
        """
        if self.mobile:
            return SE3(self.x, self.side * DOCK_OFFSET_Y, 0.0) * SE3.Rz(-self.side * np.pi / 2)
        return (
            SE3(self.x, self.side * PAD_OFFSET_Y, PAD_HEIGHT)
            * SE3.Rz(-self.side * np.pi / 2)
        )


# The loading station is near the start of the belt and the unloading (sorting)
# station sits at the far end. The loading robot is a Yaskawa MH5 on a TurtleBot3
# Waffle: `x` is where the base parks at the middle belt slot (see loading_config).
STATIONS = (
    Station("load", "Loading", 0.95, +1, "Yaskawa Motoman MH5 on TurtleBot3 Waffle", mobile=True),
    Station("paint", "Spray painting", BELT_LENGTH / 3, -1, "Staubli TX60"),
    Station("label", "Logo / label", 2 * BELT_LENGTH / 3, +1, "FANUC M-10iA"),
    Station("sort", "Sorting", BELT_LENGTH, -1, "UR3e"),
)

# (x along the belt, colour name or None if unpainted)
CARS = ((0.3, None), (1.5, None), (3.2, "red"), (4.8, "green"), (6.0, "blue"))


def station(key: str) -> Station:
    """Return the station with the given key."""
    for s in STATIONS:
        if s.key == key:
            return s
    raise KeyError(key)
