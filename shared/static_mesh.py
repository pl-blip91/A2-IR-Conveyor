import os
import struct

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

from shared import compat  # noqa: F401

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

_BOX_CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
_BOX_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def _rot(yaw=0.0, pitch=0.0, roll=0.0):
    return (SE3.Rz(yaw) * SE3.Ry(pitch) * SE3.Rx(roll)).R


def _outward(tris, centre):
    """Flip any triangle whose normal points towards the centre of its (convex) primitive."""
    out = []
    for a, b, c in tris:
        n = np.cross(b - a, c - a)
        if np.dot(n, (a + b + c) / 3 - centre) < 0:
            b, c = c, b
        out.append((a, b, c))
    return out


class StaticScene:
    def __init__(self, name):
        self.name = name
        self.groups = {}  # colour (rgba tuple) -> list of triangles (3 x 3 arrays)

    def _add(self, colour, tris):
        self.groups.setdefault(tuple(round(float(c), 3) for c in colour), []).extend(tris)

    def box(self, size, centre, colour, yaw=0.0, pitch=0.0, roll=0.0):
        """Box of `size` (x, y, z) centred at `centre`, turned by yaw (about z), then pitch (y), then roll (x)."""
        c = np.asarray(centre, float)
        pts = (_BOX_CORNERS * np.asarray(size, float)) @ _rot(yaw, pitch, roll).T + c
        tris = []
        for i, j, k, m in _BOX_FACES:
            tris += [(pts[i], pts[j], pts[k]), (pts[i], pts[k], pts[m])]
        self._add(colour, _outward(tris, c))

    def cylinder(self, radius, length, centre, colour, axis="z", n=18, yaw=0.0):
        """Cylinder (an n-sided prism) of `length` along `axis` ('x', 'y' or 'z', then turned by yaw about z)."""
        c = np.asarray(centre, float)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
        bot = np.column_stack([ring, np.full(n, -length / 2)])
        top = np.column_stack([ring, np.full(n, length / 2)])
        R = {"z": np.eye(3), "x": _rot(pitch=np.pi / 2), "y": _rot(roll=np.pi / 2)}[axis]
        R = _rot(yaw) @ R
        bot, top = bot @ R.T + c, top @ R.T + c
        cb, ct = c - R[:, 2] * length / 2, c + R[:, 2] * length / 2
        tris = []  # the ring runs anticlockwise about the axis, so this winding already gives outward normals
        for i in range(n):
            j = (i + 1) % n
            tris += [(bot[i], bot[j], top[j]), (bot[i], top[j], top[i])]
            tris += [(cb, bot[j], bot[i]), (ct, top[i], top[j])]
        self._add(colour, tris)

    @property
    def triangles(self):
        return sum(len(t) for t in self.groups.values())

    def add_to(self, env, out_dir=OUT_DIR):
        """Write one STL per colour and add each to Swift as a Mesh.  Returns the meshes."""
        os.makedirs(out_dir, exist_ok=True)
        meshes = []
        for k, (colour, tris) in enumerate(sorted(self.groups.items())):
            path = os.path.join(out_dir, f"{self.name}_{k:02d}.stl")
            write_stl(path, tris)
            mesh = sg.Mesh(filename=path, color=list(colour))
            env.add(mesh)
            meshes.append(mesh)
        return meshes


def write_stl(path, tris):
    """Binary STL with outward face normals."""
    with open(path, "wb") as f:
        f.write(b"toy car production line static scenery".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-12 else n
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))
