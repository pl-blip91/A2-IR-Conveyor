import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

from shared import compat  # noqa: F401
from shared import line_config as cfg

PARK = SE3(0, 0, -5.0).A  # parked below the floor = 'switched off' (Swift 1.1.0 cannot env.remove)
CAR_L, CAR_W, CAR_H = cfg.CAR_SIZE
MESH_REL = (SE3(0, 0, -CAR_H / 2) * SE3.Rz(-np.pi / 2)).A  # STL frame -> car-centre frame


def read_binary_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        d = np.frombuffer(f.read(n * 50), dtype=dt)
    return d["v"].astype(float)


def make_mesh(filename, color):
    try:
        return sg.Mesh(filename=filename, color=list(color))
    except TypeError:
        return sg.Mesh(filename=filename)


class CarProfile:
    """Top-surface height z(lx, ly) of the car, rasterised from the STL (Aarav's height map).

    (lx, ly) are in the STL frame: origin at the centre of the footprint, ly along the car's length (nose +ly),
    lx across it, z = 0 on the belt.  ``smooth`` is what the spray nozzle follows, ``raw`` is the real surface.
    """

    def __init__(self, path=cfg.CAR_FILE, res=0.0015, smooth_window=0.030):
        tris = read_binary_stl(path)
        xs = np.arange(-CAR_W / 2 - 0.01, CAR_W / 2 + 0.01 + res, res)
        ys = np.arange(-CAR_L / 2 - 0.01, CAR_L / 2 + 0.01 + res, res)
        H = np.zeros((len(xs), len(ys)))
        for t in tris:
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            i0 = max(int((min(x0, x1, x2) - xs[0]) / res), 0)
            i1 = min(int((max(x0, x1, x2) - xs[0]) / res) + 1, len(xs) - 1)
            j0 = max(int((min(y0, y1, y2) - ys[0]) / res), 0)
            j1 = min(int((max(y0, y1, y2) - ys[0]) / res) + 1, len(ys) - 1)
            gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            z = w0 * z0 + w1 * z1 + w2 * z2
            sub = H[i0:i1 + 1, j0:j1 + 1]
            H[i0:i1 + 1, j0:j1 + 1] = np.where(inside, np.maximum(sub, z), sub)
        k = max(int(smooth_window / res) | 1, 3)
        ker = np.ones(k) / k
        pad = k // 2
        Hs = np.pad(H, pad, mode="edge")
        Hs = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 0, Hs)
        Hs = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 1, Hs)
        self.xs, self.ys, self.raw, self.smooth, self.res = xs, ys, H, Hs, res
        self.roof = float(H.max())

    def surface_height(self, lx, ly, smooth=True):
        """Bilinear lookup of the surface height at STL-frame (lx, ly) (arrays ok)."""
        Z = self.smooth if smooth else self.raw
        lx = np.atleast_1d(np.asarray(lx, float))
        ly = np.atleast_1d(np.asarray(ly, float))
        fx = np.clip((lx - self.xs[0]) / self.res, 0, len(self.xs) - 1.001)
        fy = np.clip((ly - self.ys[0]) / self.res, 0, len(self.ys) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy
        return ((1 - ax) * (1 - ay) * Z[ix, iy] + ax * (1 - ay) * Z[ix + 1, iy]
                + (1 - ax) * ay * Z[ix, iy + 1] + ax * ay * Z[ix + 1, iy + 1])

    @staticmethod
    def body_to_stl(bx, by):
        """Car-body coordinates (x along the nose, y to the left) -> STL frame (lx, ly)."""
        return -by, bx

    def best_stamp_spot(self, w, h):
        """Where on the roof a flat die of w (along the car) x h (across it) sits flattest.

        Returns (bx, by, roof_height): the die centre in car-body coordinates and the height (above the belt) of the
        highest roof point under it, which is where the die touches.
        """
        us = np.linspace(-w / 2, w / 2, 7)
        vs = np.linspace(-h / 2, h / 2, 5)
        best = None
        for bx in np.arange(-0.045, 0.0451, 0.003):
            for by in np.arange(-0.012, 0.0121, 0.003):
                pts = [self.body_to_stl(bx + u, by + v) for u in us for v in vs]
                z = self.surface_height([p[0] for p in pts], [p[1] for p in pts], smooth=False)
                score = float(z.min()) - 0.0005 * (abs(bx) + abs(by))
                if best is None or score > best[0]:
                    best = (score, float(bx), float(by), float(z.max()))
        return best[1], best[2], best[3]


class Car:
    """A toy car on the line.  ``car.T`` places it; paint dots and stamp strokes ride along with it."""

    def __init__(self, env, colour):
        self.colour = colour
        self.rgba = cfg.CAR_COLOURS[colour]
        self.primer = make_mesh(cfg.CAR_FILE, cfg.PRIMER)
        self.coat = make_mesh(cfg.CAR_FILE, self.rgba)
        env.add(self.primer)
        env.add(self.coat)
        self.painted = False
        self.marks = []  # (kind, shape, pose in the car-centre frame)
        self._T = None
        self.T = SE3(0, 0, -3.0).A

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        value = np.array(value.A if isinstance(value, SE3) else value, dtype=float)
        if self._T is not None and np.array_equal(value, self._T):
            return
        self._T = value
        self._place()

    @property
    def x(self) -> float:
        return float(self._T[0, 3])

    def _place(self):
        on = self._T @ MESH_REL
        self.coat.T = on if self.painted else PARK
        self.primer.T = PARK if self.painted else on
        for _kind, shape, rel in self.marks:
            shape.T = self._T @ rel

    def attach_mark(self, kind, shape, world_pose):
        """Fix `shape` to the car at world_pose; it follows the car from now on."""
        world_pose = np.asarray(world_pose.A if isinstance(world_pose, SE3) else world_pose, float)
        rel = np.linalg.inv(self._T) @ world_pose
        self.marks.append((kind, shape, rel))
        shape.T = world_pose

    def reset(self):
        """Back to a bare grey car (used when the line is re-armed): paint and stamp marks are put away below the floor
        (Swift 1.1.0 cannot remove shapes) and the primer body is shown again."""
        for _kind, shape, _rel in self.marks:
            shape.T = PARK
        self.marks = []
        self.painted = False
        self._place()

    def set_painted(self, painted=True):
        """The paint has dried: the car takes its final colour and the (redundant) wet dots are put away."""
        self.painted = painted
        keep = []
        for kind, shape, rel in self.marks:
            if kind == "paint" and painted:
                shape.T = PARK
            else:
                keep.append((kind, shape, rel))
        self.marks = keep
        self._place()
