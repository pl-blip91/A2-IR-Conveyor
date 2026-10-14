# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import swift
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Cuboid, Cylinder
from roboticstoolbox import DHLink, DHRobot, jtraj, trapezoidal
from ir_support import CylindricalDHRobotPlot
import spatialgeometry as sg
from math import pi

# Compatibility shim
for _cls in (sg.Cuboid, sg.Cylinder, sg.Sphere, sg.Arrow):
    if not hasattr(_cls, "_update_pyb"):
        _cls._update_pyb = lambda self: None


class PaintRobotSimulation:
    """
    Industrial paint-robot cell.

    StudentBot6 (the arm from the TurtleBot, DH parameters from student ID) is
    mounted on Table R. A conveyor runs past the front of the arm, carrying a
    15 x 7.5 x 5 cm block (the 'toy car'). The block stops in front of the arm,
    the arm traces a sine wave over the top face and back again with a spray
    can (or 3x2 can pack) fixed to its end effector, then the arm returns home
    while the conveyor carries the block off down the line.

    World frame: +X = direction the robot faces, belt runs along Y, +Z up.
    """

    def __init__(self):
        print("StudentBot6 Industrial Paint Cell")

        # ------------------------------------------------------------------
        # Simulation timing
        # ------------------------------------------------------------------
        self.dt = 0.05                 # seconds per Swift frame

        # ------------------------------------------------------------------
        # Table R (robot table)
        # ------------------------------------------------------------------
        self.table_pos = np.array([0.0, 0.0, 0.0])
        self.table_size = 0.50         # square footprint (m)
        self.table_top_h = 0.50

        # Robot base sits on the table, rotated 180 deg about Z so the arm's
        # natural reach direction (negative a2/a3 => folds towards -X at q=0)
        # points towards +X, i.e. at the conveyor.
        self.robot_base_pos = np.array([0.0, 0.0, self.table_top_h])
        self.robot_base_T = SE3(self.robot_base_pos) * SE3.Rz(pi)

        # Default (home) pose
        self.safe_pose = np.array([0.0, -pi/2, pi/2, 0.0, 0.0, 0.0])

        # IK seed for the spray posture. A tool-down pose has several valid
        # joint solutions; this seed picks the natural one (small wrist
        # angles, elbow bent forward) so the whole path stays on one branch.
        self.spray_seed = np.array([-0.2, 0.42, -1.7, -0.28, -1.57, -1.78])

        # ------------------------------------------------------------------
        # Conveyor belt (runs along Y, perpendicular to the robot's facing)
        # ------------------------------------------------------------------
        self.belt_x = 0.40             # belt centre line (in front of robot)
        self.belt_len = 2.0
        self.belt_w = 0.22
        self.belt_top_h = 0.30
        self.belt_thick = 0.04
        self.belt_y_min = -self.belt_len / 2
        self.belt_speed = 0.25         # m/s
        self.stripe_pitch = 0.20       # moving stripes so belt motion is visible
        self.stripes = []

        # ------------------------------------------------------------------
        # Object (toy car simplified): 15 x 7.5 x 5 cm, lying on its 15x7.5 face
        # Long side (15 cm) runs along the belt (Y).
        # ------------------------------------------------------------------
        self.obj_len = 0.15            # along Y
        self.obj_wid = 0.075           # along X
        self.obj_hgt = 0.05            # along Z
        self.obj_z = self.belt_top_h + self.obj_hgt / 2
        self.obj_top_z = self.belt_top_h + self.obj_hgt
        self.obj_start_y = -0.85       # end of the conveyor (entry)
        self.obj_station_y = 0.0       # directly in front of the robot
        self.obj_end_y = 0.85          # end of the line (exit)
        self.obj_y = self.obj_start_y

        # ------------------------------------------------------------------
        # Spray can(s) attached to the end effector
        #   can_layout = (1, 1) -> single can
        #   can_layout = (3, 2) -> pack of 6 (3 x 2)
        # ------------------------------------------------------------------
        self.can_layout = (1, 1)
        self.can_radius = 0.03
        self.can_body_len = 0.12
        self.nozzle_len = 0.02
        self.can_pitch = 0.07
        self.plate_thick = 0.01
        # Flange -> nozzle tip distance along the tool z axis
        self.tool_len = self.plate_thick + self.can_body_len + self.nozzle_len

        # ------------------------------------------------------------------
        # Spray pattern
        # ------------------------------------------------------------------
        self.spray_standoff = 0.20     # nozzle tip -> top surface (must be 0.15-0.30)
        assert 0.15 <= self.spray_standoff <= 0.30, "Standoff outside 15-30 cm"
        self.sine_cycles = 3
        self.sine_amp = self.obj_wid / 2        # covers full width of the top face
        self.spray_speed = 0.10                 # m/s along the path (medium pace)
        self.mark_every = 3                     # paint dot every N frames

        # Runtime handles
        self.robot = None
        self.tool_parts = []           # list of (geometry, local 4x4)
        self.obj = None
        self.spray_ray = None
        self.paint_marks = []          # list of (geometry, rel_x, rel_y)

    # ----------------------------------------------------------------------
    # Robot
    # ----------------------------------------------------------------------
    def build_studentbot6(self):
        d1, a2, a3, d4, d5, d6 = 0.1450, -0.2700, -0.2500, 0.0800, 0.0780, 0.0760
        links = [
            DHLink(d=d1, a=0, alpha=pi/2, qlim=[-2*pi, 2*pi]),
            DHLink(d=0, a=a2, alpha=0, qlim=[-pi, pi]),
            DHLink(d=0, a=a3, alpha=0, qlim=[-pi, pi]),
            DHLink(d=d4, a=0, alpha=pi/2, qlim=[-2*pi, 2*pi]),
            DHLink(d=d5, a=0, alpha=-pi/2, qlim=[-2*pi, 2*pi]),
            DHLink(d=d6, a=0, alpha=0, qlim=[-2*pi, 2*pi]),
        ]
        robot = DHRobot(links, name='StudentBot6')
        cyl_viz = CylindricalDHRobotPlot(
            robot, cylinder_radius=0.035, color="#3478f6"
        )
        return cyl_viz.create_cylinders()

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
                scale=[0.02, self.belt_len, 0.06],
                color=[0.6, 0.6, 0.65, 1.0]
            )
            rail.T = SE3(
                self.belt_x + side * (self.belt_w / 2 + 0.01),
                0.0,
                self.belt_top_h - 0.03 + 0.005
            ).A
            env.add(rail)

        # Conveyor: legs
        leg_h = self.belt_top_h - self.belt_thick
        for ly in (-0.9, 0.9):
            for side in (-1, 1):
                leg = Cuboid(
                    scale=[0.04, 0.04, leg_h],
                    color=[0.6, 0.6, 0.65, 1.0]
                )
                leg.T = SE3(
                    self.belt_x + side * (self.belt_w / 2 + 0.01),
                    ly, leg_h / 2
                ).A
                env.add(leg)

        # Moving stripes (purely visual: show the belt is running)
        n_stripes = int(self.belt_len / self.stripe_pitch)
        for i in range(n_stripes):
            stripe = Cuboid(
                scale=[self.belt_w * 0.9, 0.02, 0.002],
                color=[0.35, 0.35, 0.35, 1.0]
            )
            env.add(stripe)
            self.stripes.append((stripe, i * self.stripe_pitch))

        # Object (toy car stand-in)
        obj = Cuboid(
            scale=[self.obj_wid, self.obj_len, self.obj_hgt],
            color=[0.2, 0.5, 0.9, 1.0]
        )
        env.add(obj)
        self.obj = obj

        # Spray "ray" (thin cylinder from nozzle to surface, hidden when off)
        ray = Cylinder(
            radius=0.004, length=self.spray_standoff,
            color=[1.0, 0.2, 0.2, 0.5]
        )
        ray.T = SE3(0, 0, -5.0).A
        env.add(ray)
        self.spray_ray = ray

        self.set_object_y(self.obj_start_y)

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
                    color=[0.85, 0.1, 0.1, 1.0]
                )
                env.add(body)
                self.tool_parts.append((
                    body,
                    SE3(cx, cy,
                        self.plate_thick + self.can_body_len / 2).A
                ))

                nozzle = Cylinder(
                    radius=0.008, length=self.nozzle_len,
                    color=[0.9, 0.9, 0.9, 1.0]
                )
                env.add(nozzle)
                self.tool_parts.append((
                    nozzle,
                    SE3(cx, cy,
                        self.plate_thick + self.can_body_len
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
    # Frame updates (belt / tool / spray)
    # ----------------------------------------------------------------------
    def set_object_y(self, y):
        """Move the object (plus its paint and the belt stripes) to belt position y."""
        self.obj_y = y
        self.obj.T = SE3(self.belt_x, y, self.obj_z).A

        # Paint dots ride on the object
        for mark, rx, ry in self.paint_marks:
            mark.T = SE3(
                self.belt_x + rx, y + ry, self.obj_top_z + 0.0005
            ).A

        # Stripes scroll with the belt
        shift = y - self.obj_start_y
        for stripe, offset in self.stripes:
            sy = self.belt_y_min + ((offset + shift) % self.belt_len)
            stripe.T = SE3(self.belt_x, sy, self.belt_top_h + 0.001).A

    def tip_position(self, q=None):
        """World position of the nozzle tip (centre of the tool)."""
        T = self.robot.fkine(self.robot.q if q is None else q).A
        return T[:3, 3] + T[:3, 2] * self.tool_len

    def update_tool(self):
        T = self.robot.fkine(self.robot.q).A
        for geom, local in self.tool_parts:
            geom.T = T @ local

    def update_spray(self, spraying, add_mark=False, env=None):
        if not spraying:
            self.spray_ray.T = SE3(0, 0, -5.0).A
            return
        tip = self.tip_position()
        ray_len = max(tip[2] - self.obj_top_z, 0.01)
        self.spray_ray.T = SE3(tip[0], tip[1], tip[2] - ray_len / 2).A

        if add_mark and env is not None:
            rx = tip[0] - self.belt_x
            ry = tip[1] - self.obj_y
            mark = Cuboid(
                scale=[0.007, 0.007, 0.001],
                color=[1.0, 0.1, 0.1, 1.0]
            )
            mark.T = SE3(tip[0], tip[1], self.obj_top_z + 0.0005).A
            env.add(mark)
            self.paint_marks.append((mark, rx, ry))

    def frame(self, env, q=None, obj_y=None, spraying=False, add_mark=False):
        """Advance the simulation by one Swift frame."""
        if q is not None:
            self.robot.q = q
        if obj_y is not None:
            self.set_object_y(obj_y)
        self.update_tool()
        self.update_spray(spraying, add_mark, env)
        env.step(self.dt)

    # ----------------------------------------------------------------------
    # Kinematics helpers
    # ----------------------------------------------------------------------
    def check_joint_limits(self, Q, robot):
        success = True
        Q = np.atleast_2d(Q)
        qlim = robot.qlim
        for i, q in enumerate(Q):
            lower_violation = q < qlim[0]
            upper_violation = q > qlim[1]
            if np.any(lower_violation) or np.any(upper_violation):
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
        Full 6-DOF target for the flange so the nozzle tip is at tip_xyz with the
        can pointing straight down (flange z axis = world -Z).
        """
        flange_xyz = np.asarray(tip_xyz, dtype=float) + np.array(
            [0.0, 0.0, self.tool_len]
        )
        return SE3(flange_xyz[0], flange_xyz[1], flange_xyz[2]) * SE3.Rx(pi)

    def solve_pose_ik(self, target, q_seed, label="", extra_seeds=()):
        """Full-pose IK (position + 'can pointing down' orientation)."""
        seeds = [np.asarray(q_seed, dtype=float)] + list(extra_seeds)
        for seed in seeds:
            sol = self.robot.ikine_LM(target, q0=seed)
            if getattr(sol, "success", True):
                q = sol.q
                if self.check_joint_limits(q, self.robot):
                    return q
        raise RuntimeError(f"IK failed for {label}")

    # ----------------------------------------------------------------------
    # Sine-wave spray path
    # ----------------------------------------------------------------------
    def generate_sine_pass(self, forward=True, dense=800):
        """
        Dense nozzle-tip waypoints for ONE pass over the top face, in world
        coordinates, for an object sitting at the station.

        Travels along the 15 cm length (Y) while oscillating across the 7.5 cm
        width (X). The return pass runs the opposite way with an integer number
        of cycles, which automatically puts its sine in anti-phase with the
        forward pass, so together the two passes fill in the surface.
        """
        half = self.obj_len / 2
        y0, y1 = (-half, half) if forward else (half, -half)
        local_y = np.linspace(y0, y1, dense)
        phase = 2 * pi * self.sine_cycles * (local_y - y0) / (y1 - y0)
        local_x = self.sine_amp * np.sin(phase)

        pts = np.column_stack([
            self.belt_x + local_x,
            self.obj_station_y + local_y,
            np.full(dense, self.obj_top_z + self.spray_standoff)
        ])
        return pts

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
              f"{np.degrees(max_jump):.1f} deg/frame")
        if max_jump > np.radians(20):
            print(f"WARNING: large joint jump in {label} (possible IK branch flip)")
        return Q

    # ----------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------
    def run(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.6, -1.4, 1.1], [0.35, 0.0, 0.35])
        self.create_minimal_axes(
            env, scale=1.0, radius=0.005,
            head_scale=0.03, head_radius=0.15
        )

        # Robot on Table R
        self.robot = self.build_studentbot6()
        self.robot.q = self.safe_pose
        self.robot.base = self.robot_base_T
        env.add(self.robot, readonly=False)

        self.build_scene(env)
        self.build_tool(env)
        self.frame(env, q=self.safe_pose, obj_y=self.obj_start_y)

        # -------- STEP 1: default position, object at end of conveyor --------
        self.demo_forward_kinematics(
            self.robot, self.safe_pose, label="StudentBot6 (default pose)"
        )
        input("STEP 1: Scene built. Arm in default pose, object at the "
              "start of the conveyor. Press Enter to start the belt.\n")

        # -------- STEP 2: belt carries the object in front of the arm --------
        dist = self.obj_station_y - self.obj_start_y
        n_belt = int(np.ceil(dist / (self.belt_speed * self.dt)))
        s_belt = np.asarray(trapezoidal(0, 1, n_belt).q).flatten()
        print("STEP 2: conveyor moving object to the spray station...")
        for i in range(n_belt):
            y = self.obj_start_y + s_belt[i] * dist
            self.frame(env, obj_y=y)
        self.set_object_y(self.obj_station_y)

        # Pre-compute the spray path and its joint solutions
        print("Planning sine-wave spray passes...")
        fwd_tip, len_f = self.resample_constant_speed(
            self.generate_sine_pass(forward=True))
        bwd_tip, len_b = self.resample_constant_speed(
            self.generate_sine_pass(forward=False))
        print(f"Pass length: forward {len_f:.2f} m, return {len_b:.2f} m, "
              f"standoff {self.spray_standoff*100:.0f} cm")

        q_start = self.solve_pose_ik(
            self.tip_to_flange_target(fwd_tip[0]), self.spray_seed,
            label="spray start"
        )
        Q_fwd = self.path_to_joints(fwd_tip, q_start, label="forward pass")
        Q_bwd = self.path_to_joints(bwd_tip, Q_fwd[-1], label="return pass")
        self.demo_forward_kinematics(
            self.robot, q_start, label="StudentBot6 (spray start)"
        )

        input("STEP 2 complete: object is in front of the arm. "
              "Press Enter to start spraying.\n")

        # -------- STEP 3: move to start, sine forward, sine back --------
        print("STEP 3: moving to the spray start point...")
        for q in jtraj(self.safe_pose, q_start, 50).q:
            self.frame(env, q=q)

        print("STEP 3: spraying forward pass...")
        for i, q in enumerate(Q_fwd):
            self.frame(env, q=q, spraying=True,
                       add_mark=(i % self.mark_every == 0))

        print("STEP 3: spraying return pass...")
        for i, q in enumerate(Q_bwd):
            self.frame(env, q=q, spraying=True,
                       add_mark=(i % self.mark_every == 0))

        # -------- STEP 4: arm returns home WHILE belt carries object away --------
        input("Spraying complete. Press Enter to return the arm and "
              "restart the conveyor.\n")
        print("STEP 4: arm returning to default pose, conveyor moving...")
        Q_home = jtraj(Q_bwd[-1], self.safe_pose, 60).q
        dist = self.obj_end_y - self.obj_station_y
        n_belt = int(np.ceil(dist / (self.belt_speed * self.dt)))
        s_belt = np.asarray(trapezoidal(0, 1, n_belt).q).flatten()

        for i in range(max(len(Q_home), n_belt)):
            q = Q_home[i] if i < len(Q_home) else self.safe_pose
            y = self.obj_station_y + s_belt[min(i, n_belt - 1)] * dist
            self.frame(env, q=q, obj_y=y)
        self.set_object_y(self.obj_end_y)

        # -------- STEP 5: end --------
        print("Task complete: object painted with a sine-wave pattern and "
              "carried to the end of the line; arm back in default pose.")
        input("STEP 5: Press Enter to close the simulation.\n")
        env.close()


if __name__ == "__main__":
    sim = PaintRobotSimulation()
    sim.run()

    print("Paint robot simulation completed. You can now close the window.")
    plt.close("all")
    time.sleep(0.5)
