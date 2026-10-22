# Required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import swift
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Cuboid, Cylinder, Mesh
from roboticstoolbox import DHLink, DHRobot, jtraj, trapezoidal
from ir_support import CylindricalDHRobotPlot
import spatialgeometry as sg
from math import pi

# Compatibility shim
for _cls in (sg.Cuboid, sg.Cylinder, sg.Sphere, sg.Arrow, sg.Mesh):
    if not hasattr(_cls, "_update_pyb"):
        _cls._update_pyb = lambda self: None


class PaintRobotSimulation:
    """
    Industrial paint cell: Motoman MPX3500 on Table R, a conveyor carrying three
    'Incredible Jaban' cars past the robot, and a spray can (or 3x2 pack)
    attached to the robot's flange.

    Cycle (for each car):
        belt brings the car in front of the arm -> arm sprays a sine-wave pattern
        over the car's top surface and back again (the car turns its paint colour)
        -> arm returns home while the belt brings the next car forward.
    Car 1 is painted blue, car 2 red, car 3 green. After car 3 the belt runs all
    three cars off down the line.

    World frame: +X = direction the robot faces, the belt runs along Y, +Z up.
    Robot q = 0 is the manufacturer's CAD pose: lower arm vertical, forearm
    horizontal along +X.

    Required next to this file:  ./assets/   (see prepare_assets.py)
    """

    # ------------------------------------------------------------------
    # Which DH table to use for the robot.
    #   "cad"  : link dimensions measured from the supplied MPX3500 CAD model.
    #            The 3D robot meshes only line up with this one.
    #   "user" : the table supplied with the brief. Meshes don't fit it, so the
    #            robot is drawn as simple cylinders instead.
    # ------------------------------------------------------------------
    DH_SOURCE = "cad"

    def __init__(self):
        print("Motoman MPX3500 Paint Cell")
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.assets = os.path.join(self.here, "assets")
        self.interactive = True        # set False to run without Enter prompts
        self.dt = 0.05                 # seconds per Swift frame

        # ------------------------------------------------------------------
        # Robot kinematics (all lengths in metres)
        # ------------------------------------------------------------------
        r = np.radians
        self.dh_tables = {
            # Measured from the manufacturer CAD (zero pose reproduces: L axis 0.795 m,
            # U axis 2.059 m, forearm axis 2.003 m, wrist centre 1.442 m, flange 1.578 m)
            "cad": dict(
                d=[0.795, 0.0, 0.0, 1.442, 0.0, 0.136],
                a=[0.0, 1.264, -0.056, 0.0, 0.0, 0.0],
            ),
            # As supplied in the brief
            "user": dict(
                d=[0.550, 0.0, 0.0, 1.300, 0.0, 0.175],
                a=[0.180, 1.150, 0.250, 0.0, 0.0, 0.0],
            ),
        }
        self.dh_alpha = [r(-90), 0.0, r(-90), r(90), r(-90), 0.0]
        self.dh_offset = [0.0, -pi/2, 0.0, 0.0, 0.0, 0.0]
        # S/L from published data (L: -65/+115 deg); U/R/B/T are typical Motoman ranges
        self.qlim = [
            [r(-170), r(170)],   # S
            [r(-65),  r(115)],   # L
            [r(-110), r(160)],   # U  (assumed)
            [r(-360), r(360)],   # R
            [r(-125), r(125)],   # B
            [r(-360), r(360)],   # T
        ]

        # ------------------------------------------------------------------
        # Table R (robot table)
        # ------------------------------------------------------------------
        self.table_pos = np.array([0.0, 0.0, 0.0])
        self.table_size = 1.20
        self.table_top_h = 0.40
        self.robot_base_pos = np.array([0.0, 0.0, self.table_top_h])
        self.robot_base_T = SE3(self.robot_base_pos)

        # Default (home) pose = CAD zero pose
        self.home_pose = np.zeros(6)

        # ------------------------------------------------------------------
        # Conveyor (runs along Y, perpendicular to the robot's facing)
        # ------------------------------------------------------------------
        self.belt_x = 1.80
        self.belt_len = 4.40
        self.belt_w = 0.26
        self.belt_top_h = 0.45
        self.belt_thick = 0.05
        self.belt_y_min = -self.belt_len / 2
        self.belt_speed = 0.20         # m/s
        self.stripe_pitch = 0.20
        self.stripes = []
        self.belt_s = 0.0              # distance the belt has travelled so far

        # ------------------------------------------------------------------
        # Cars (Incredible Jaban STL, 7.5 x 15 x 4 cm, nose towards +Y)
        # ------------------------------------------------------------------
        self.car_file = os.path.join(self.assets, "Incredible_Jaban_m.stl")
        self.car_len, self.car_wid, self.car_hgt = 0.15, 0.075, 0.04
        self.car_pitch = 0.45          # spacing between cars on the belt
        self.station_y = 0.0           # spray station: directly in front of the robot
        self.car_start_y = [-1.00 - k * self.car_pitch for k in range(3)]
        self.car_colours = [
            ("blue",  [0.10, 0.25, 0.95, 1.0]),
            ("red",   [0.90, 0.10, 0.10, 1.0]),
            ("green", [0.10, 0.75, 0.20, 1.0]),
        ]
        self.primer_colour = [0.70, 0.70, 0.72, 1.0]
        self.final_run_distance = 1.00  # belt run after the last car, to clear the line
        self.cars = []                 # runtime handles, one dict per car
        self.hidden_z = -5.0           # parking spot for hidden geometry

        # ------------------------------------------------------------------
        # Spray can(s) on the flange
        #   can_layout = (1, 1) -> single can
        #   can_layout = (3, 2) -> pack of 6 (3 x 2)
        # ------------------------------------------------------------------
        self.can_layout = (1, 1)
        self.can_radius = 0.03
        self.can_body_len = 0.12
        self.nozzle_len = 0.02
        self.can_pitch = 0.07
        self.plate_thick = 0.01
        self.tool_len = self.plate_thick + self.can_body_len + self.nozzle_len

        # ------------------------------------------------------------------
        # Spray pattern
        # ------------------------------------------------------------------
        self.spray_standoff = 0.20     # nozzle -> car surface (must be 0.15-0.30 m)
        assert 0.15 <= self.spray_standoff <= 0.30, "Standoff outside 15-30 cm"
        self.sine_cycles = 3
        self.sine_amp = self.car_wid / 2
        self.spray_speed = 0.10        # m/s along the path (medium pace)
        self.mark_every = 2            # paint dot every N frames
        self.smooth_window = 0.030     # m, smoothing of the surface the tip follows

        # Runtime handles
        self.robot = None
        self.robot_has_meshes = False
        self.link_meshes = []          # list of (mesh, link frame index)
        self.tool_parts = []           # list of (geometry, local 4x4)
        self.rays = []                 # one spray ray per colour
        self.height_grid = None

    # ----------------------------------------------------------------------
    # Robot
    # ----------------------------------------------------------------------
    def build_robot(self):
        """DH model of the MPX3500 using the selected DH table."""
        t = self.dh_tables[self.DH_SOURCE]
        links = []
        for i in range(6):
            links.append(DHLink(
                d=t["d"][i], a=t["a"][i], alpha=self.dh_alpha[i],
                offset=self.dh_offset[i], qlim=self.qlim[i]
            ))
        robot = DHRobot(links, name='MPX3500')
        robot.base = self.robot_base_T

        mesh_files = ["base", "S", "L", "U", "R", "B", "T"]
        have_meshes = all(
            os.path.exists(os.path.join(self.assets, f"MPX3500_{n}.stl"))
            for n in mesh_files
        )
        if self.DH_SOURCE == "cad" and have_meshes:
            self.robot_has_meshes = True
            return robot

        if self.DH_SOURCE == "cad":
            print("MPX3500 meshes not found in ./assets - drawing cylinders instead.")
        else:
            print("Using the DH table from the brief - drawing cylinders "
                  "(the CAD meshes only fit the CAD-derived table).")
        cyl_viz = CylindricalDHRobotPlot(
            robot, cylinder_radius=0.08, color="#3478f6"
        )
        return cyl_viz.create_cylinders()

    def _make_mesh(self, filename, color):
        try:
            return Mesh(filename=filename, color=color)
        except TypeError:
            return Mesh(filename=filename)

    def build_robot_meshes(self, env):
        """One mesh per link, already expressed in that link's own DH frame."""
        spec = [  # (file, link frame index, colour)
            ("base", 0, [0.25, 0.25, 0.28, 1.0]),
            ("S",    1, [0.88, 0.88, 0.90, 1.0]),
            ("L",    2, [0.92, 0.92, 0.94, 1.0]),
            ("U",    3, [0.92, 0.92, 0.94, 1.0]),
            ("R",    4, [0.75, 0.75, 0.78, 1.0]),
            ("B",    5, [0.85, 0.85, 0.88, 1.0]),
            ("T",    6, [0.55, 0.55, 0.60, 1.0]),
        ]
        for name, idx, colour in spec:
            m = self._make_mesh(
                os.path.join(self.assets, f"MPX3500_{name}.stl"), colour
            )
            env.add(m)
            self.link_meshes.append((m, idx))

    def link_frames(self, q):
        """World pose (4x4) of the base frame and of every link frame."""
        T = self.robot.base.A.copy()
        frames = [T]
        for link, qi in zip(self.robot.links, q):
            T = T @ link.A(qi).A
            frames.append(T)
        return frames

    # ----------------------------------------------------------------------
    # Scene
    # ----------------------------------------------------------------------
    def build_scene(self, env):
        # Table R
        table = Cuboid(
            scale=[self.table_size, self.table_size, self.table_top_h],
            color=[0.55, 0.4, 0.25, 1.0]
        )
        table.T = SE3(
            self.table_pos + np.array([0, 0, self.table_top_h / 2])
        ).A
        env.add(table)

        # Conveyor: belt surface
        belt = Cuboid(
            scale=[self.belt_w, self.belt_len, self.belt_thick],
            color=[0.12, 0.12, 0.12, 1.0]
        )
        belt.T = SE3(
            self.belt_x, 0.0, self.belt_top_h - self.belt_thick / 2
        ).A
        env.add(belt)

        # Conveyor: side rails
        for side in (-1, 1):
            rail = Cuboid(
                scale=[0.02, self.belt_len, 0.07],
                color=[0.6, 0.6, 0.65, 1.0]
            )
            rail.T = SE3(
                self.belt_x + side * (self.belt_w / 2 + 0.01),
                0.0,
                self.belt_top_h - 0.035 + 0.005
            ).A
            env.add(rail)

        # Conveyor: legs
        leg_h = self.belt_top_h - self.belt_thick
        for ly in (-self.belt_len / 2 + 0.2, 0.0, self.belt_len / 2 - 0.2):
            for side in (-1, 1):
                leg = Cuboid(
                    scale=[0.05, 0.05, leg_h],
                    color=[0.6, 0.6, 0.65, 1.0]
                )
                leg.T = SE3(
                    self.belt_x + side * (self.belt_w / 2 + 0.01),
                    ly, leg_h / 2
                ).A
                env.add(leg)

        # Moving stripes (purely visual: show the belt is running)
        n_stripes = int(round(self.belt_len / self.stripe_pitch))
        for i in range(n_stripes):
            stripe = Cuboid(
                scale=[self.belt_w * 0.9, 0.02, 0.002],
                color=[0.35, 0.35, 0.35, 1.0]
            )
            env.add(stripe)
            self.stripes.append((stripe, i * self.stripe_pitch))

        # Cars: a grey 'primer' copy and a hidden painted copy of each
        for k, (cname, ccol) in enumerate(self.car_colours):
            grey = self._make_mesh(self.car_file, self.primer_colour)
            painted = self._make_mesh(self.car_file, ccol)
            env.add(grey)
            env.add(painted)
            self.cars.append(dict(
                name=cname, colour=ccol, grey=grey, painted=painted,
                is_painted=False, marks=[], start_y=self.car_start_y[k]
            ))

        # One spray 'ray' per colour (thin cylinder nozzle -> car, hidden when off)
        for cname, ccol in self.car_colours:
            ray = Cylinder(
                radius=0.004, length=self.spray_standoff,
                color=[ccol[0], ccol[1], ccol[2], 0.5]
            )
            ray.T = SE3(0, 0, self.hidden_z).A
            env.add(ray)
            self.rays.append(ray)

        self.set_belt_s(0.0)

    def build_tool(self, env):
        """
        Adapter plate + spray can(s) rigidly attached to the flange.
        Tool frame = flange frame (z points along the can towards the nozzle).
        """
        nx, ny = self.can_layout
        plate_w = (nx - 1) * self.can_pitch + 2 * self.can_radius + 0.02
        plate_d = (ny - 1) * self.can_pitch + 2 * self.can_radius + 0.02

        plate = Cuboid(
            scale=[plate_w, plate_d, self.plate_thick],
            color=[0.3, 0.3, 0.3, 1.0]
        )
        env.add(plate)
        self.tool_parts.append((plate, SE3(0, 0, self.plate_thick / 2).A))

        for ix in range(nx):
            for iy in range(ny):
                cx = (ix - (nx - 1) / 2) * self.can_pitch
                cy = (iy - (ny - 1) / 2) * self.can_pitch

                body = Cylinder(
                    radius=self.can_radius, length=self.can_body_len,
                    color=[0.85, 0.85, 0.85, 1.0]
                )
                env.add(body)
                self.tool_parts.append((
                    body,
                    SE3(cx, cy, self.plate_thick + self.can_body_len / 2).A
                ))

                nozzle = Cylinder(
                    radius=0.008, length=self.nozzle_len,
                    color=[0.1, 0.1, 0.1, 1.0]
                )
                env.add(nozzle)
                self.tool_parts.append((
                    nozzle,
                    SE3(cx, cy, self.plate_thick + self.can_body_len
                        + self.nozzle_len / 2).A
                ))

    def create_minimal_axes(self, env, scale=3.0, radius=0.01,
                            head_scale=0.5, head_radius=0.1):
        axes_objects = []
        x_pos = Arrow(length=scale, radius=radius,
                      head_length=head_scale, head_radius=head_radius,
                      color=[1.0, 0.0, 0.0, 1.0])
        x_pos.T = SE3.Ry(pi/2).A
        env.add(x_pos)
        axes_objects.append(x_pos)

        y_pos = Arrow(length=scale, radius=radius,
                      head_length=head_scale, head_radius=head_radius,
                      color=[0.0, 1.0, 0.0, 1.0])
        y_pos.T = SE3.Rx(-pi/2).A
        env.add(y_pos)
        axes_objects.append(y_pos)

        z_pos = Arrow(length=scale, radius=radius,
                      head_length=head_scale, head_radius=head_radius,
                      color=[0.0, 0.0, 1.0, 1.0])
        z_pos.T = SE3(0, 0, 0).A
        env.add(z_pos)
        axes_objects.append(z_pos)
        return axes_objects

    # ----------------------------------------------------------------------
    # Car surface height map (for a contour-following spray path)
    # ----------------------------------------------------------------------
    @staticmethod
    def _read_binary_stl(path):
        with open(path, 'rb') as f:
            f.read(80)
            n = int(np.frombuffer(f.read(4), dtype='<u4')[0])
            dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            d = np.frombuffer(f.read(n * 50), dtype=dt)
        return d['v'].astype(float)

    def build_height_map(self, res=0.0015):
        """
        Top-surface height z(x, y) of the car in its own frame (origin at the
        centre of its footprint, z = 0 on the belt), rasterised from the STL.
        Stored raw (true surface) and smoothed (what the nozzle tip follows).
        """
        tris = self._read_binary_stl(self.car_file)
        xs = np.arange(-self.car_wid / 2 - 0.01, self.car_wid / 2 + 0.01 + res, res)
        ys = np.arange(-self.car_len / 2 - 0.01, self.car_len / 2 + 0.01 + res, res)
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
            gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing='ij')
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            z = w0 * z0 + w1 * z1 + w2 * z2
            sub = H[i0:i1 + 1, j0:j1 + 1]
            H[i0:i1 + 1, j0:j1 + 1] = np.where(inside, np.maximum(sub, z), sub)

        # smooth with a separable box filter so the nozzle glides over the body
        k = max(int(self.smooth_window / res) | 1, 3)
        ker = np.ones(k) / k
        pad = k // 2
        Hs = np.pad(H, pad, mode='edge')
        Hs = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='valid'), 0, Hs)
        Hs = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='valid'), 1, Hs)
        self.height_grid = dict(xs=xs, ys=ys, raw=H, smooth=Hs, res=res)
        print(f"Car height map: roof {H.max()*1000:.0f} mm, "
              f"lowest sampled body point {H[H > 0].min()*1000:.0f} mm")

    def surface_height(self, x, y, smooth=True):
        """Bilinear lookup of the car surface height at local (x, y) (arrays ok)."""
        g = self.height_grid
        Z = g['smooth'] if smooth else g['raw']
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        fx = np.clip((x - g['xs'][0]) / g['res'], 0, len(g['xs']) - 1.001)
        fy = np.clip((y - g['ys'][0]) / g['res'], 0, len(g['ys']) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax_, ay_ = fx - ix, fy - iy
        return ((1 - ax_) * (1 - ay_) * Z[ix, iy] + ax_ * (1 - ay_) * Z[ix + 1, iy]
                + (1 - ax_) * ay_ * Z[ix, iy + 1] + ax_ * ay_ * Z[ix + 1, iy + 1])

    # ----------------------------------------------------------------------
    # Frame updates (belt / cars / robot / tool / spray)
    # ----------------------------------------------------------------------
    def car_y(self, k):
        return self.cars[k]['start_y'] + self.belt_s

    def set_belt_s(self, s):
        """Move the belt to travelled distance s: cars, paint dots and stripes follow."""
        self.belt_s = s
        for k, car in enumerate(self.cars):
            y = self.car_y(k)
            on = SE3(self.belt_x, y, self.belt_top_h).A
            off = SE3(self.belt_x, y, self.hidden_z).A
            car['grey'].T = off if car['is_painted'] else on
            car['painted'].T = on if car['is_painted'] else off
            for mark, rx, ry, rz in car['marks']:
                mark.T = SE3(self.belt_x + rx, y + ry, self.belt_top_h + rz).A

        for stripe, offset in self.stripes:
            sy = self.belt_y_min + ((offset + s) % self.belt_len)
            stripe.T = SE3(self.belt_x, sy, self.belt_top_h + 0.001).A

    def tip_position(self, q=None):
        """World position of the nozzle tip (centre of the tool)."""
        q = self.robot.q if q is None else q
        T = self.link_frames(q)[-1]
        return T[:3, 3] + T[:3, 2] * self.tool_len

    def update_robot(self, q):
        frames = self.link_frames(q)
        if self.robot_has_meshes:
            for mesh, idx in self.link_meshes:
                mesh.T = frames[idx]
        for geom, local in self.tool_parts:
            geom.T = frames[-1] @ local

    def update_spray(self, env, spraying, car_idx=None, add_mark=False):
        for ray in self.rays:
            ray.T = SE3(0, 0, self.hidden_z).A
        if not spraying:
            return
        tip = self.tip_position()
        self.rays[car_idx].T = SE3(
            tip[0], tip[1], tip[2] - self.spray_standoff / 2
        ).A

        if add_mark:
            car = self.cars[car_idx]
            y = self.car_y(car_idx)
            lx, ly = tip[0] - self.belt_x, tip[1] - y
            h = float(self.surface_height(lx, ly, smooth=False)[0])
            rz = max(h, 0.002) + 0.001
            mark = Cuboid(
                scale=[0.010, 0.010, 0.001],
                color=[car['colour'][0], car['colour'][1], car['colour'][2], 1.0]
            )
            mark.T = SE3(tip[0], tip[1], self.belt_top_h + rz).A
            env.add(mark)
            car['marks'].append((mark, lx, ly, rz))

    def frame(self, env, q=None, belt_s=None, spraying=False,
              car_idx=None, add_mark=False):
        """Advance the simulation by one Swift frame."""
        if q is not None:
            self.robot.q = q               # (drives the cylinder model if used)
        if belt_s is not None:
            self.set_belt_s(belt_s)
        self.update_robot(self.robot.q)
        self.update_spray(env, spraying, car_idx, add_mark)
        env.step(self.dt)

    def mark_painted(self, car_idx):
        """Paint has 'dried': the car takes on its final colour."""
        self.cars[car_idx]['is_painted'] = True
        self.set_belt_s(self.belt_s)

    def pause(self, msg):
        if self.interactive:
            input(msg + "\n")
        else:
            print(msg)

    # ----------------------------------------------------------------------
    # Kinematics helpers
    # ----------------------------------------------------------------------
    def check_joint_limits(self, Q, robot, verbose=True):
        success = True
        Q = np.atleast_2d(Q)
        qlim = robot.qlim
        for i, q in enumerate(Q):
            lower_violation = q < qlim[0]
            upper_violation = q > qlim[1]
            if np.any(lower_violation) or np.any(upper_violation):
                if verbose:
                    print(f"q[{i}] is out of joint limits:")
                    for j in range(len(q)):
                        if lower_violation[j]:
                            print(f"  Joint {j}: {q[j]:.3f} < lower limit {qlim[0, j]:.3f}")
                        elif upper_violation[j]:
                            print(f"  Joint {j}: {q[j]:.3f} > upper limit {qlim[1, j]:.3f}")
                success = False
        return success

    def demo_forward_kinematics(self, robot, q, label=""):
        T = robot.fkine(q)
        print(f"[FK] {label} joint angles (rad): {np.round(q, 3)}")
        print(f"[FK] {label} end-effector pose:\n{T}\n")
        return T

    def tip_to_flange_target(self, tip_xyz):
        """
        Full 6-DOF flange target so the nozzle tip is at tip_xyz with the can
        pointing straight down (flange z axis = world -Z).
        """
        flange_xyz = np.asarray(tip_xyz, dtype=float) + np.array(
            [0.0, 0.0, self.tool_len]
        )
        return SE3(flange_xyz[0], flange_xyz[1], flange_xyz[2]) * SE3.Rx(pi)

    def solve_pose_ik(self, target, q_seed, label="", verbose=True):
        """Full-pose IK (position + 'can pointing down' orientation)."""
        sol = self.robot.ikine_LM(target, q0=np.asarray(q_seed, dtype=float))
        if getattr(sol, "success", True) and \
                self.check_joint_limits(sol.q, self.robot, verbose=verbose):
            return sol.q
        raise RuntimeError(f"IK failed for {label}")

    def find_spray_start_pose(self, target):
        """
        A tool-down pose has several valid joint solutions (wrist flips etc.).
        Try a spread of seeds and keep the in-limit solution whose wrist roll
        joints (R and T) are closest to zero - the natural, no-flip posture.
        """
        rng = np.random.default_rng(0)
        seeds = [np.array([0.0, 0.78, 0.07, 0.0, 0.72, 0.0])]
        for _ in range(60):
            seeds.append(np.array([
                rng.uniform(-0.4, 0.4), rng.uniform(-0.5, 1.2),
                rng.uniform(-0.8, 1.5), rng.uniform(-2.5, 2.5),
                rng.uniform(-1.8, 1.8), rng.uniform(-2.5, 2.5)
            ]))
        best, best_cost = None, np.inf
        for seed in seeds:
            try:
                q = self.solve_pose_ik(target, seed, "spray start", verbose=False)
            except RuntimeError:
                continue
            cost = abs(q[3]) + abs(q[5]) + 0.1 * np.sum(np.abs(q))
            if cost < best_cost:
                best, best_cost = q, cost
        if best is None:
            raise RuntimeError("No valid spray posture found - check layout / DH table")
        return best

    # ----------------------------------------------------------------------
    # Sine-wave spray path (follows the car's contour)
    # ----------------------------------------------------------------------
    def generate_sine_pass(self, forward=True, dense=800):
        """
        Dense nozzle-tip waypoints for ONE pass over the car, in world coordinates,
        for a car sitting at the station.

        Travels along the 15 cm length (Y) while oscillating across the 7.5 cm
        width (X). The tip height follows the (smoothed) car surface, held
        'spray_standoff' above it. The return pass runs the opposite way with an
        integer number of cycles, so its sine is in anti-phase with the forward
        pass and the two passes together fill in the surface.
        """
        half = self.car_len / 2
        y0, y1 = (-half, half) if forward else (half, -half)
        local_y = np.linspace(y0, y1, dense)
        phase = 2 * pi * self.sine_cycles * (local_y - y0) / (y1 - y0)
        local_x = self.sine_amp * np.sin(phase)
        z = (self.belt_top_h + self.surface_height(local_x, local_y, smooth=True)
             + self.spray_standoff)
        return np.column_stack([
            self.belt_x + local_x, self.station_y + local_y, z
        ])

    def resample_constant_speed(self, pts):
        """
        Re-time a dense path to the medium spray speed using a trapezoidal
        velocity profile (smooth start/stop) over its arc length.
        """
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc[-1]
        n_steps = int(np.ceil(total / (self.spray_speed * self.dt)))
        s = np.asarray(trapezoidal(0, 1, n_steps).q).flatten()
        targets = s * total
        out = np.column_stack([
            np.interp(targets, arc, pts[:, k]) for k in range(3)
        ])
        return out, total

    def path_to_joints(self, tip_path, q_seed, label=""):
        """Sequential IK along the path, each solve seeded by the previous."""
        Q = np.empty((len(tip_path), self.robot.n))
        q_prev = np.asarray(q_seed, dtype=float)
        for i, p in enumerate(tip_path):
            q_prev = self.solve_pose_ik(
                self.tip_to_flange_target(p), q_prev,
                label=f"{label} waypoint {i}"
            )
            Q[i, :] = q_prev
        max_jump = np.max(np.abs(np.diff(Q, axis=0)))
        print(f"[{label}] {len(Q)} waypoints, max joint step "
              f"{np.degrees(max_jump):.2f} deg/frame")
        if max_jump > np.radians(10):
            print(f"WARNING: large joint jump in {label} (possible IK branch flip)")
        return Q

    # ----------------------------------------------------------------------
    # One full cycle for one car
    # ----------------------------------------------------------------------
    def run_belt(self, env, s_from, s_to):
        dist = s_to - s_from
        n = int(np.ceil(abs(dist) / (self.belt_speed * self.dt)))
        prof = np.asarray(trapezoidal(0, 1, n).q).flatten()
        for i in range(n):
            self.frame(env, belt_s=s_from + prof[i] * dist)
        self.set_belt_s(s_to)

    def paint_car(self, env, k, q_start, Q_fwd, Q_bwd):
        name = self.cars[k]['name']
        print(f"STEP 3: spraying car {k+1} ({name}) - moving to the start point...")
        for q in jtraj(self.home_pose, q_start, 60).q:
            self.frame(env, q=q)

        print(f"STEP 3: car {k+1} forward sine pass...")
        for i, q in enumerate(Q_fwd):
            self.frame(env, q=q, spraying=True, car_idx=k,
                       add_mark=(i % self.mark_every == 0))

        print(f"STEP 3: car {k+1} return sine pass...")
        for i, q in enumerate(Q_bwd):
            self.frame(env, q=q, spraying=True, car_idx=k,
                       add_mark=(i % self.mark_every == 0))

        self.frame(env, spraying=False)
        self.mark_painted(k)
        print(f"Car {k+1} is now {name}.")

    def return_home_and_advance(self, env, q_end, s_from, s_to):
        """Arm returns to the default pose while the belt moves at the same time."""
        Q_home = jtraj(q_end, self.home_pose, 70).q
        dist = s_to - s_from
        n_belt = int(np.ceil(abs(dist) / (self.belt_speed * self.dt)))
        prof = np.asarray(trapezoidal(0, 1, n_belt).q).flatten()
        for i in range(max(len(Q_home), n_belt)):
            q = Q_home[i] if i < len(Q_home) else self.home_pose
            s = s_from + prof[min(i, n_belt - 1)] * dist
            self.frame(env, q=q, belt_s=s)
        self.set_belt_s(s_to)

    # ----------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------
    def run(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([4.6, -3.6, 2.6], [1.2, 0.0, 0.9])
        self.create_minimal_axes(
            env, scale=1.0, radius=0.005,
            head_scale=0.03, head_radius=0.15
        )

        # Robot on Table R
        self.robot = self.build_robot()
        self.robot.q = self.home_pose
        self.robot.base = self.robot_base_T
        if self.robot_has_meshes:
            self.build_robot_meshes(env)        # meshes follow the DH frames
        else:
            env.add(self.robot, readonly=False)

        self.build_scene(env)
        self.build_tool(env)
        self.build_height_map()
        self.frame(env, q=self.home_pose)

        # -------- STEP 1: default position, cars at the start of the belt --------
        self.demo_forward_kinematics(
            self.robot, self.home_pose, label="MPX3500 (default pose)"
        )
        self.pause("STEP 1: Scene built. Arm in default pose, three cars waiting at "
                   "the start of the conveyor. Press Enter to start the belt.")

        # -------- Plan the spray passes once (identical for every car) --------
        print("Planning sine-wave spray passes...")
        fwd_tip, len_f = self.resample_constant_speed(
            self.generate_sine_pass(forward=True))
        bwd_tip, len_b = self.resample_constant_speed(
            self.generate_sine_pass(forward=False))
        surf = self.belt_top_h + self.surface_height(
            fwd_tip[:, 0] - self.belt_x, fwd_tip[:, 1] - self.station_y, smooth=False)
        gap = fwd_tip[:, 2] - surf
        print(f"Pass length {len_f:.2f} m each, nozzle-to-surface distance "
              f"{gap.min()*100:.1f}-{gap.max()*100:.1f} cm")

        q_start = self.find_spray_start_pose(
            self.tip_to_flange_target(fwd_tip[0]))
        Q_fwd = self.path_to_joints(fwd_tip, q_start, label="forward pass")
        Q_bwd = self.path_to_joints(bwd_tip, Q_fwd[-1], label="return pass")
        self.demo_forward_kinematics(
            self.robot, q_start, label="MPX3500 (spray start)"
        )

        # -------- STEP 2: belt brings car 1 in front of the arm --------
        s_station = [-c['start_y'] + self.station_y for c in self.cars]
        print("STEP 2: conveyor moving car 1 to the spray station...")
        self.run_belt(env, 0.0, s_station[0])

        # -------- STEPS 3 + 4 for each car --------
        for k in range(len(self.cars)):
            self.pause(f"Car {k+1} ({self.cars[k]['name']}) is in front of the arm. "
                       f"Press Enter to start spraying.")
            self.paint_car(env, k, q_start, Q_fwd, Q_bwd)

            if k < len(self.cars) - 1:
                s_next = s_station[k + 1]
                print(f"STEP 4: arm returning home, belt bringing car {k+2} forward...")
            else:
                s_next = s_station[k] + self.final_run_distance
                print("STEP 4: arm returning home, belt carrying all cars down the line...")
            self.return_home_and_advance(env, Q_bwd[-1], s_station[k], s_next)

        # -------- STEP 5: end --------
        print("Task complete: car 1 painted blue, car 2 red, car 3 green; "
              "all carried down the line, arm back in its default pose.")
        self.pause("STEP 5: Press Enter to close the simulation.")
        env.close()


if __name__ == "__main__":
    sim = PaintRobotSimulation()
    sim.run()

    print("Paint robot simulation completed. You can now close the window.")
    plt.close("all")
    time.sleep(0.5)
