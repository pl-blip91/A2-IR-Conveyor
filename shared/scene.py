import numpy as np
from spatialgeometry import Cuboid
from spatialmath import SE3
from spatialmath.base import transl

from shared import compat  # noqa: F401  (Swift 1.1.0 / spatialgeometry compatibility patch)

# Shapes that are 'switched off' are parked below the floor (Swift 1.1.0 cannot env.remove a shape)
PARK = transl(0, 0, -5)


def box(env, size, centre, colour, yaw=0.0):
    """Add a Cuboid of `size` (x, y, z) centred at `centre`, turned by `yaw` about z (Lab 3 Cuboid)."""
    pose = SE3(*centre) * SE3.Rz(yaw) if yaw else SE3(*centre)
    shape = Cuboid(scale=list(size), pose=pose, color=list(colour))
    env.add(shape)
    return shape


class Assembly:
    """Several shapes that move together: `.T` places the whole group."""

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
