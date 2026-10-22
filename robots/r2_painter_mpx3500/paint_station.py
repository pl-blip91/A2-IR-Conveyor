import os
from math import pi
from types import SimpleNamespace

import numpy as np
import spatialgeometry as sg
from roboticstoolbox import DHLink, DHRobot, trapezoidal
from spatialmath import SE3

from shared import compat  # noqa: F401
from shared import line_config as cfg
from shared.car import PARK, make_mesh

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")


class PaintStation:
    # "cad": link dimensions measured from the supplied MPX3500 CAD model (the 3D meshes only line up with this one).
    # "user": the table supplied with the brief; the meshes don't fit it, so the robot is drawn as simple cylinders.
    DH_SOURCE = "cad"

    def __init__(self, env, profile, base=None):
        self.env = env
        self.profile = profile
        self.assets = MESH_DIR
        self.base_T = cfg.PAINT_BASE if base is None else base
        self.dt = cfg.DT

        # ---- robot kinematics (all lengths in metres) ----
        r = np.radians
        self.dh_tables = {
            # Measured from the manufacturer CAD (zero pose reproduces: L axis 0.795 m, U axis 2.059 m,
            # forearm axis 2.003 m, wrist centre 1.442 m, flange 1.578 m)
            "cad": dict(d=[0.795, 0.0, 0.0, 1.442, 0.0, 0.136], a=[0.0, 1.264, -0.056, 0.0, 0.0, 0.0]),
            # As supplied in the brief
            "user": dict(d=[0.550, 0.0, 0.0, 1.300, 0.0, 0.175], a=[0.180, 1.150, 0.250, 0.0, 0.0, 0.0]),
        }
        self.dh_alpha = [r(-90), 0.0, r(-90), r(90), r(-90), 0.0]
        self.dh_offset = [0.0, -pi / 2, 0.0, 0.0, 0.0, 0.0]
        # S/L from published data (L: -65/+115 deg); U/R/B/T are typical Motoman ranges
        self.qlim = [
            [r(-170), r(170)],   # S
            [r(-65), r(115)],    # L
            [r(-110), r(160)],   # U  (assumed)
            [r(-360), r(360)],   # R
            [r(-125), r(125)],   # B
            [r(-360), r(360)],   # T
        ]
        self.home_pose = np.zeros(6)  # default pose = CAD zero pose

        # ---- table R (the robot's table) ----
        self.table_size = cfg.PAINT_TABLE_SIZE
        self.table_top_h = cfg.PAINT_TABLE_TOP

        # ---- spray can(s) on the flange ----
        #   can_layout = (1, 1) -> single can ;  (3, 2) -> pack of 6 (3 x 2)
        self.can_layout = (1, 1)
        self.can_radius = 0.03
        self.can_body_len = 0.12
        self.nozzle_len = 0.02
        self.can_pitch = 0.07
        self.plate_thick = 0.01
        self.tool_len = self.plate_thick + self.can_body_len + self.nozzle_len

        # ---- spray pattern ----
        self.spray_standoff = 0.20  # nozzle -> car surface (must be 0.15-0.30 m)
        assert 0.15 <= self.spray_standoff <= 0.30, "Standoff outside 15-30 cm"
        self.sine_cycles = 3
        self.sine_amp = cfg.CAR_SIZE[1] / 2
        self.spray_speed = 0.10  # m/s along the path (medium pace)
        self.mark_every = 2  # a paint dot every N frames

        self.robot_has_meshes = False
        self.link_meshes = []  # (mesh, link frame index)
        self.tool_parts = []   # (geometry, local 4x4)
        self.rays = []         # one spray ray per car colour

        self.robot = self.build_robot()
        self.robot.q = self.home_pose
        self.robot.base = self.base_T
        if self.robot_has_meshes:
            self.build_robot_meshes(env)  # meshes follow the DH frames
        else:
            env.add(self.robot, readonly=False)
        self.build_table(env)
        self.build_tool(env)
        self.build_rays(env)
        self.draw()

    # ------------------------------------------------------------------ robot
    def build_robot(self):
        """DH model of the MPX3500 using the selected DH table."""
        t = self.dh_tables[self.DH_SOURCE]
        links = [DHLink(d=t["d"][i], a=t["a"][i], alpha=self.dh_alpha[i], offset=self.dh_offset[i], qlim=self.qlim[i])
                 for i in range(6)]
        robot = DHRobot(links, name="MPX3500")
        robot.base = self.base_T
        have_meshes = all(os.path.exists(os.path.join(self.assets, f"MPX3500_{n}.stl"))
                          for n in ("base", "S", "L", "U", "R", "B", "T"))
        if self.DH_SOURCE == "cad" and have_meshes:
            self.robot_has_meshes = True
            return robot
        print("MPX3500 meshes not found in ./meshes (or the brief's DH table is selected) - drawing cylinders instead.")
        from ir_support import CylindricalDHRobotPlot
        return CylindricalDHRobotPlot(robot, cylinder_radius=0.08, color="#3478f6").create_cylinders()

    def build_robot_meshes(self, env):
        """One mesh per link, already expressed in that link's own DH frame."""
        spec = [("base", 0, [0.25, 0.25, 0.28, 1.0]), ("S", 1, [0.88, 0.88, 0.90, 1.0]),
                ("L", 2, [0.92, 0.92, 0.94, 1.0]), ("U", 3, [0.92, 0.92, 0.94, 1.0]),
                ("R", 4, [0.75, 0.75, 0.78, 1.0]), ("B", 5, [0.85, 0.85, 0.88, 1.0]),
                ("T", 6, [0.55, 0.55, 0.60, 1.0])]
        for name, idx, colour in spec:
            m = make_mesh(os.path.join(self.assets, f"MPX3500_{name}.stl"), colour)
            env.add(m)
            self.link_meshes.append((m, idx))

    def link_frames(self, q):
        """World pose (4x4) of the base frame and of every link frame."""
        T = np.asarray(self.robot.base.A, dtype=float).copy()
        frames = [T]
        for link, qi in zip(self.robot.links, q):
            T = T @ link.A(qi).A
            frames.append(T)
        return frames

    def tip_position(self, q=None):
        """World position of the nozzle tip (end of the can)."""
        q = self.robot.q if q is None else q
        T = self.link_frames(q)[-1]
        return T[:3, 3] + T[:3, 2] * self.tool_len

    # ------------------------------------------------------------------ scene pieces
    def build_table(self, env):
        table = sg.Cuboid(scale=[self.table_size, self.table_size, self.table_top_h], color=[0.55, 0.4, 0.25, 1.0])
        table.T = SE3(self.base_T.t[0], self.base_T.t[1], self.table_top_h / 2).A
        env.add(table)
        self.table = table

    def build_tool(self, env):
        """Adapter plate + spray can(s) rigidly attached to the flange (tool frame = flange frame, z along the can)."""
        nx, ny = self.can_layout
        plate_w = (nx - 1) * self.can_pitch + 2 * self.can_radius + 0.02
        plate_d = (ny - 1) * self.can_pitch + 2 * self.can_radius + 0.02
        plate = sg.Cuboid(scale=[plate_w, plate_d, self.plate_thick], color=[0.3, 0.3, 0.3, 1.0])
        env.add(plate)
        self.tool_parts.append((plate, SE3(0, 0, self.plate_thick / 2).A))
        for ix in range(nx):
            for iy in range(ny):
                cx = (ix - (nx - 1) / 2) * self.can_pitch
                cy = (iy - (ny - 1) / 2) * self.can_pitch
                body = sg.Cylinder(radius=self.can_radius, length=self.can_body_len, color=[0.85, 0.85, 0.85, 1.0])
                env.add(body)
                self.tool_parts.append((body, SE3(cx, cy, self.plate_thick + self.can_body_len / 2).A))
                nozzle = sg.Cylinder(radius=0.008, length=self.nozzle_len, color=[0.1, 0.1, 0.1, 1.0])
                env.add(nozzle)
                self.tool_parts.append(
                    (nozzle, SE3(cx, cy, self.plate_thick + self.can_body_len + self.nozzle_len / 2).A))

    def build_rays(self, env):
        """One thin translucent cylinder per car colour (nozzle -> car); parked when the spray is off."""
        for name in cfg.CAR_ORDER:
            c = cfg.CAR_COLOURS[name]
            ray = sg.Cylinder(radius=0.004, length=self.spray_standoff, color=[c[0], c[1], c[2], 0.5])
            ray.T = PARK
            env.add(ray)
            self.rays.append(ray)

    def draw(self, spray_idx=None):
        """Place the link meshes, the spray can and the spray ray for the current joint angles."""
        frames = self.link_frames(self.robot.q)
        if self.robot_has_meshes:
            for mesh, idx in self.link_meshes:
                mesh.T = frames[idx]
        for geom, local in self.tool_parts:
            geom.T = frames[-1] @ local
        for ray in self.rays:
            ray.T = PARK
        if spray_idx is not None:
            tip = self.tip_position()
            self.rays[spray_idx].T = SE3(tip[0], tip[1], tip[2] - self.spray_standoff / 2).A

    def mark_pose(self, car_T, tip):
        """Pose of a wet-paint dot where the spray hits the car, lying on its (raw) surface."""
        p = np.linalg.inv(car_T) @ np.r_[tip, 1.0]
        lx, ly = self.profile.body_to_stl(p[0], p[1])
        h = float(self.profile.surface_height(lx, ly, smooth=False)[0])
        rz = max(h, 0.002) + 0.001
        return SE3(tip[0], tip[1], cfg.BELT_HEIGHT + rz)

    # ------------------------------------------------------------------ kinematics helpers
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

    def tip_to_flange_target(self, tip_xyz):
        """Full 6-DOF flange target so the nozzle tip is at tip_xyz with the can pointing straight down.

        Aarav's target was Rx(pi) in his own frame; his frame is the robot's frame, which here is turned to face the belt
        (base yaw = -side * 90 deg), so the same pose in the world carries that yaw as well.
        """
        flange_xyz = np.asarray(tip_xyz, dtype=float) + np.array([0.0, 0.0, self.tool_len])
        yaw = -cfg.PAINT_SIDE * pi / 2
        return SE3(flange_xyz[0], flange_xyz[1], flange_xyz[2]) * SE3.Rz(yaw) * SE3.Rx(pi)

    def solve_pose_ik(self, target, q_seed, label="", verbose=True):
        """Full-pose IK (position + 'can pointing down' orientation)."""
        sol = self.robot.ikine_LM(target, q0=np.asarray(q_seed, dtype=float))
        if getattr(sol, "success", True) and self.check_joint_limits(sol.q, self.robot, verbose=verbose):
            return sol.q
        raise RuntimeError(f"IK failed for {label}")

    def find_spray_start_pose(self, target):
        """A tool-down pose has several valid joint solutions (wrist flips etc.).

        Try a spread of seeds and keep the in-limit solution whose wrist roll joints (R and T) are closest to zero.
        """
        rng = np.random.default_rng(0)
        seeds = [np.array([0.0, 0.78, 0.07, 0.0, 0.72, 0.0])]
        for _ in range(60):
            seeds.append(np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.5, 1.2), rng.uniform(-0.8, 1.5),
                                   rng.uniform(-2.5, 2.5), rng.uniform(-1.8, 1.8), rng.uniform(-2.5, 2.5)]))
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

    def max_reach(self, samples=4000, seed=1):
        """Largest horizontal distance (m) of the nozzle tip from the S (base) axis over the whole joint range.

        Random joint samples, then the best one is polished by a few gradient-free coordinate steps.  Used at start-up
        to check that the guard fence round the robot is out of its reach (line_config.PAINT_REACH)."""
        lo, hi = np.asarray(self.robot.qlim[0], float), np.asarray(self.robot.qlim[1], float)
        rng = np.random.default_rng(seed)
        base = np.linalg.inv(np.asarray(self.robot.base.A, float))

        def reach(q):
            p = base @ np.r_[self.tip_position(q), 1.0]
            return float(np.hypot(p[0], p[1]))

        best_q = max((rng.uniform(lo, hi) for _ in range(samples)), key=reach)
        step = np.radians(5.0)
        while step > np.radians(0.01):
            improved = False
            for j in range(6):
                for d in (step, -step):
                    q = best_q.copy()
                    q[j] = np.clip(q[j] + d, lo[j], hi[j])
                    if reach(q) > reach(best_q) + 1e-9:
                        best_q, improved = q, True
            if not improved:
                step /= 2
        return reach(best_q)

    # ------------------------------------------------------------------ sine-wave spray path
    def generate_sine_pass(self, car_xy, forward=True, dense=800):
        """Dense nozzle-tip waypoints (world) for ONE pass over a car whose centre is at car_xy.

        Travels along the 15 cm length while oscillating across the 7.5 cm width.  The tip height follows the (smoothed)
        car surface, held `spray_standoff` above it.  The return pass runs the opposite way with an integer number of
        cycles, so its sine is in anti-phase with the forward pass and the two passes together fill in the surface.
        """
        half = cfg.CAR_SIZE[0] / 2
        y0, y1 = (-half, half) if forward else (half, -half)
        local_y = np.linspace(y0, y1, dense)  # STL frame: along the car
        phase = 2 * pi * self.sine_cycles * (local_y - y0) / (y1 - y0)
        local_x = self.sine_amp * np.sin(phase)  # STL frame: across the car
        z = cfg.BELT_HEIGHT + self.profile.surface_height(local_x, local_y, smooth=True) + self.spray_standoff
        # STL frame -> world for a car lying along the belt: world x = cx + ly, world y = cy - lx
        return np.column_stack([car_xy[0] + local_y, car_xy[1] - local_x, z])

    def resample_constant_speed(self, pts):
        """Re-time a dense path to the medium spray speed using a trapezoidal velocity profile over its arc length."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc[-1]
        n_steps = int(np.ceil(total / (self.spray_speed * self.dt)))
        s = np.asarray(trapezoidal(0, 1, n_steps).q).flatten()
        targets = s * total
        out = np.column_stack([np.interp(targets, arc, pts[:, k]) for k in range(3)])
        return out, total

    def path_to_joints(self, tip_path, q_seed, label=""):
        """Sequential IK along the path, each solve seeded by the previous."""
        Q = np.empty((len(tip_path), self.robot.n))
        q_prev = np.asarray(q_seed, dtype=float)
        for i, p in enumerate(tip_path):
            q_prev = self.solve_pose_ik(self.tip_to_flange_target(p), q_prev, label=f"{label} waypoint {i}")
            Q[i, :] = q_prev
        max_jump = np.max(np.abs(np.diff(Q, axis=0)))
        print(f"[{label}] {len(Q)} waypoints, max joint step {np.degrees(max_jump):.2f} deg/frame")
        if max_jump > np.radians(10):
            print(f"WARNING: large joint jump in {label} (possible IK branch flip)")
        return Q

    def plan(self, car_xy=(cfg.PAINT_X, 0.0)):
        """Plan the spray passes once (identical for every car, since every car stops at the same spot)."""
        print("Planning the MPX3500 sine-wave spray passes ...")
        fwd_tip, len_f = self.resample_constant_speed(self.generate_sine_pass(car_xy, forward=True))
        bwd_tip, _len_b = self.resample_constant_speed(self.generate_sine_pass(car_xy, forward=False))
        surf = cfg.BELT_HEIGHT + self.profile.surface_height(-(fwd_tip[:, 1] - car_xy[1]), fwd_tip[:, 0] - car_xy[0],
                                                             smooth=False)
        gap = fwd_tip[:, 2] - surf
        print(f"Pass length {len_f:.2f} m each, nozzle-to-surface distance {gap.min() * 100:.1f}-{gap.max() * 100:.1f} cm")
        q_start = self.find_spray_start_pose(self.tip_to_flange_target(fwd_tip[0]))
        Q_fwd = self.path_to_joints(fwd_tip, q_start, label="forward pass")
        Q_bwd = self.path_to_joints(bwd_tip, Q_fwd[-1], label="return pass")
        return SimpleNamespace(q_start=q_start, Q_fwd=Q_fwd, Q_bwd=Q_bwd, tip_fwd=fwd_tip, tip_bwd=bwd_tip,
                               q_home=self.home_pose.copy())
