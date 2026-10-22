import numpy as np

from shared import line_config as cfg

# --- colours -----------------------------------------------------------------------------------------------------------
CONCRETE = (0.58, 0.58, 0.60, 1.0)
EPOXY = (0.40, 0.48, 0.45, 1.0)
WALKWAY = (0.22, 0.52, 0.30, 1.0)
LINE_YELLOW = (0.95, 0.80, 0.10, 1.0)
LINE_WHITE = (0.92, 0.92, 0.92, 1.0)
WALL = (0.82, 0.82, 0.80, 1.0)
DADO = (0.40, 0.43, 0.47, 1.0)
WINDOW = (0.62, 0.78, 0.90, 1.0)
STEEL = (0.45, 0.48, 0.52, 1.0)
DARK = (0.15, 0.15, 0.17, 1.0)
GUARD_YELLOW = (0.98, 0.82, 0.05, 1.0)

# --- the hall ------------------------------------------------------------------------------------------------------------
HALL_X = (-11.0, 17.0)
HALL_Y = (-14.0, 11.5)
WALL_H = 7.0
FLOOR_TOP = 0.006  # the slab covers Swift's own floor plane and axes at z = 0 (they would z-fight with it)
EPOXY_TOP = FLOOR_TOP + 0.002  # painted floor areas
LINE_TOP = FLOOR_TOP + 0.004  # painted lines on top of them


def _floor(sb):
    (x0, x1), (y0, y1) = HALL_X, HALL_Y
    sb.box((x1 - x0, y1 - y0, 0.10), ((x0 + x1) / 2, (y0 + y1) / 2, FLOOR_TOP - 0.05), CONCRETE)
    # epoxy floor inside the guarded cell (main rectangle + the paint enclosure)
    fx0, fx1, Y = cfg.FENCE_X0, cfg.FENCE_X1, cfg.FENCE_Y
    t, z = EPOXY_TOP - FLOOR_TOP, (EPOXY_TOP + FLOOR_TOP) / 2  # painted areas
    lt, lz = LINE_TOP - FLOOR_TOP, (LINE_TOP + FLOOR_TOP) / 2  # painted lines
    sb.box((fx1 - fx0, 2 * Y, t), ((fx0 + fx1) / 2, 0.0, z), EPOXY)
    bx0, bx1 = cfg.PAINT_BULGE_X
    sb.box((bx1 - bx0, cfg.PAINT_BULGE_Y - Y, t), ((bx0 + bx1) / 2, (Y + cfg.PAINT_BULGE_Y) / 2, z), EPOXY)
    # green pedestrian walkway along the front of the line, yellow edges, white crossing bars every 2 m
    wy0, wy1 = -Y - 2.4, -Y - 1.2
    sb.box((x1 - x0 - 1.0, wy1 - wy0, t), ((x0 + x1) / 2, (wy0 + wy1) / 2, z), WALKWAY)
    for y in (wy0, wy1):
        sb.box((x1 - x0 - 1.0, 0.08, lt), ((x0 + x1) / 2, y, lz), LINE_YELLOW)
    for x in np.arange(x0 + 2.0, x1 - 1.0, 2.0):
        sb.box((0.5, 0.10, lt), (x, (wy0 + wy1) / 2, lz), LINE_WHITE)
    # yellow aisle line round the back of the cell and the paint enclosure
    ay = cfg.PAINT_BULGE_Y + 1.6
    sb.box((x1 - x0 - 1.0, 0.08, lt), ((x0 + x1) / 2, ay, lz), LINE_YELLOW)
    for x in (fx0 - 1.0, fx1 + 1.0):
        sb.box((0.08, ay - wy1, lt), (x, (ay + wy1) / 2, lz), LINE_YELLOW)


def _walls(sb):
    (x0, x1), (y0, y1) = HALL_X, HALL_Y
    th = 0.25
    for (size, centre) in (((x1 - x0, th, WALL_H), ((x0 + x1) / 2, y1, WALL_H / 2)),  # back wall
                           ((th, y1 - y0, WALL_H), (x0, (y0 + y1) / 2, WALL_H / 2)),  # left wall
                           ((th, y1 - y0, WALL_H), (x1, (y0 + y1) / 2, WALL_H / 2))):  # right wall
        sb.box(size, centre, WALL)
    # dark dado band and high windows on the inside faces
    sb.box((x1 - x0, 0.02, 1.2), ((x0 + x1) / 2, y1 - th / 2 - 0.01, 0.6), DADO)
    for x in (x0, x1):
        s = 1 if x == x0 else -1
        sb.box((0.02, y1 - y0, 1.2), (x + s * (th / 2 + 0.01), (y0 + y1) / 2, 0.6), DADO)
    for wx in np.arange(x0 + 2.5, x1 - 1.5, 3.5):
        sb.box((2.4, 0.03, 1.4), (wx, y1 - th / 2 - 0.02, 4.9), WINDOW)
    for wy in np.arange(y0 + 3.0, y1 - 1.5, 3.5):
        sb.box((0.03, 2.4, 1.4), (x0 + th / 2 + 0.02, wy, 4.9), WINDOW)
        sb.box((0.03, 2.4, 1.4), (x1 - th / 2 - 0.02, wy, 4.9), WINDOW)
    # roller door in the right wall for the forklifts
    sb.box((0.04, 4.0, 4.2), (x1 - th / 2 - 0.03, -6.0, 2.1), STEEL)
    for z in np.arange(0.3, 4.2, 0.3):
        sb.box((0.05, 4.0, 0.03), (x1 - th / 2 - 0.05, -6.0, z), DADO)


def _columns(sb):
    (x0, x1), (y0, y1) = HALL_X, HALL_Y
    for x in np.arange(x0 + 0.4, x1, 6.0):  # columns along the back wall
        _column(sb, x, y1 - 0.4)
    for y in np.arange(y0 + 2.0, y1 - 1.0, 6.0):  # ... and along the side walls
        _column(sb, x0 + 0.4, y)
        _column(sb, x1 - 0.4, y)
    sb.box((x1 - x0, 0.3, 0.5), ((x0 + x1) / 2, y1 - 0.4, WALL_H - 0.25), STEEL)  # eaves beam


def _column(sb, x, y):
    sb.box((0.30, 0.30, WALL_H), (x, y, WALL_H / 2), STEEL)
    sb.box((0.42, 0.42, 1.0), (x, y, 0.5), GUARD_YELLOW)
    for z in (0.25, 0.75):
        sb.box((0.43, 0.43, 0.12), (x, y, z), DARK)


def build_factory(sb):
    """Add the whole factory hall to the StaticScene `sb`."""
    _floor(sb)
    _walls(sb)
    _columns(sb)
