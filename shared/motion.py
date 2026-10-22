from itertools import combinations
from math import pi

import numpy as np
from ir_support import RectangularPrism, line_plane_intersection
from roboticstoolbox import jtraj, trapezoidal
from scipy import linalg
from spatialgeometry import Cuboid
from spatialmath.base import transl

from shared import line_config as cfg
from shared.scene import PARK

DT = cfg.DT
LOOKAHEAD = 24  # frames of the planned path checked for the obstacle
RRT_TRIES = 40  # random waypoints tried when re-planning (Lab 5 Q2/3 primitive RRT)
RRT_SPREAD = np.deg2rad(60)  # random waypoints are drawn within this of the colliding waypoint
SELF_SAMPLES = 7  # points sampled along each link segment for the self-collision test
SELF_WRIST_FROM = 3  # segment index where the wrist starts (base column, upper arm, forearm are segments 0, 1, 2)


# =============================================================================
# Obstacle + collision checking (Lab 5 Q2/3: RectangularPrism + line_plane_intersection + is_collision)
# =============================================================================
class CubeObstacle:
    """A cube that can be dropped into the planned path (Lab 5 Q2/3: Cuboid + RectangularPrism(...).get_data())."""

    def __init__(self, env, size, margin, colour=(1.0, 0.45, 0.0, 1.0)):
        self.size, self.margin, self.active = size, margin, False
        self.vertices = self.faces = self.face_normals = None
        self.centre = None
        self.shape = Cuboid(scale=[size, size, size], color=list(colour))
        env.add(self.shape)
        self.remove()

    def place(self, centre):
        big = self.size + 2 * self.margin  # collision prism = cube + safety clearance
        self.vertices, self.faces, self.face_normals = RectangularPrism(big, big, big, center=list(centre)).get_data()
        self.shape.T = transl(centre)
        self.centre = np.asarray(centre, float)
        self.active = True

    def remove(self):
        self.active = False
        self.centre = None
        self.shape.T = PARK


def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    w = intersect_p - triangle_verts[0, :]
    wu, wv = np.dot(w, u), np.dot(w, v)
    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:
        return False
    return True


def get_link_poses(robot, q=None, tip=0.0):
    """Lab 5 get_link_poses (fkine_all) plus one extra point: the end-effector (tool included) pushed on by `tip`."""
    q = robot.q if q is None else q
    tr = list(robot.fkine_all(q).A)
    tr.append(robot.fkine(q).A @ transl(0, 0, tip))
    return tr


def is_collision(robot, q_matrix, faces, vertex, face_normals, tip=0.0):
    """Lab 5 Q2/3 is_collision: True as soon as any link of any pose in q_matrix crosses a face of the obstacle prism."""
    for q in q_matrix:
        tr = get_link_poses(robot, q, tip)
        for i in range(len(tr) - 1):
            if np.linalg.norm(tr[i + 1][:3, 3] - tr[i][:3, 3]) < 1e-9:  # zero-length link (coincident frames)
                continue
            for j, face in enumerate(faces):
                vert_on_plane = vertex[face][0]
                intersect_p, check = line_plane_intersection(face_normals[j], vert_on_plane, tr[i][:3, 3], tr[i + 1][:3, 3])
                triangle_list = np.array(list(combinations(face, 3)), dtype=int)
                if check == 1:
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertex[triangle]):
                            return True
    return False


def fine_interpolation(q1, q2, max_step_radians):
    """Lab 5 Q2/3: keep calling jtraj until all step sizes are smaller than a given max step size."""
    steps = 2
    while np.any(max_step_radians < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
        steps += 1
    return jtraj(q1, q2, steps).q


def interpolate_waypoints_radians(waypoint_radians, max_step_radians):
    """Lab 5 Q2/3: given a set of waypoints, finely interpolate them."""
    q_matrix = []
    for i in range(np.size(waypoint_radians, 0) - 1):
        for q in fine_interpolation(waypoint_radians[i], waypoint_radians[i + 1], max_step_radians):
            q_matrix.append(q)
    return q_matrix


def segment_points(p0, p1, n=SELF_SAMPLES):
    """n points evenly spaced along the line from p0 to p1 (the link is a line through consecutive joint frames, Lab 5)."""
    s = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1 - s) * np.asarray(p0, float) + s * np.asarray(p1, float)


def self_collision(robot, q, tip, radii):
    """Lab 5 get_link_poses gives the joint positions, so segment k runs from joint k to joint k+1 (the last segment is the
    tool). Two segments that are NOT neighbours collide when some sampled points are closer than the sum of their radii
    (the same 'distance below a radius' idea as Lab 6 Q2). Neighbouring segments always touch, so they are skipped."""
    tr = get_link_poses(robot, q, tip)
    pts = [t[:3, 3] for t in tr]
    segs = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    samples = [segment_points(a, b) for a, b in segs]
    for i in range(min(len(segs), SELF_WRIST_FROM)):  # segments from SELF_WRIST_FROM on are the wrist + gripper: they sit
        for j in range(i + 2, len(segs)):  # a few cm apart by design, so wrist-vs-wrist pairs are not tested
            if np.linalg.norm(segs[i][1] - segs[i][0]) < 1e-9 or np.linalg.norm(segs[j][1] - segs[j][0]) < 1e-9:
                continue  # zero-length link (coincident frames)
            d = np.linalg.norm(samples[i][:, None, :] - samples[j][None, :, :], axis=2).min()
            if d < radii[i] + radii[j]:
                return True
    return False


# =============================================================================
# Arm: one robot + what the planner / jog need to know about it
# =============================================================================
class Arm:
    """A robot with its own obstacle, tip length (for the collision points), singularity threshold, joint step and status.
    `self_radii` (optional) switches on the self-collision check: one clearance radius per link segment."""

    def __init__(self, name, robot, obstacle, tip, epsilon, step, self_radii=None):
        self.name, self.robot, self.obstacle = name, robot, obstacle
        self.tip, self.epsilon, self.step = tip, epsilon, step
        self.self_radii = self_radii
        self.safe_q = None  # upright pose used as a via-point when a direct joint move would hit the arm itself
        self.blocked, self.note, self.replans = False, "idle", 0
        self.jog_R = None  # tool orientation held during a Cartesian jog

    @property
    def q(self):
        return np.asarray(self.robot.q, float)

    def hits(self, q_list):
        o = self.obstacle
        return o.active and len(q_list) > 0 and is_collision(self.robot, q_list, o.faces, o.vertices, o.face_normals, self.tip)

    def self_hits(self, q_list):
        """True if the arm would pass through ITSELF in any pose of q_list (only arms with self_radii are checked)."""
        return self.self_radii is not None and any(self_collision(self.robot, q, self.tip, self.self_radii) for q in q_list)

    def unsafe(self, q_list):
        """Used by the planner: a path is unsafe if it hits the obstacle OR the arm itself."""
        return self.hits(q_list) or self.self_hits(q_list)

    def jog_blocked(self, q_new):
        """Why a jog to q_new must be refused (None if it may go).  Fixed arms also keep every joint above the floor."""
        if self.hits([q_new]):
            return "obstacle in the way"
        if self.self_hits([q_new]):
            return "the arm would pass through itself"
        if min(T[2, 3] for T in get_link_poses(self.robot, q_new, self.tip)) < 0.02:
            return "the arm would go through the floor"
        return None

    def tcp(self):
        """World position of the tool point the Cartesian jog moves (robot.fkine, so robot.tool is included)."""
        return np.asarray(self.robot.fkine(self.robot.q).t, float)

    def manipulability(self, q=None):
        J = self.robot.jacob0(self.q if q is None else q)
        return float(np.sqrt(max(linalg.det(J @ J.T), 0.0)))


# =============================================================================
# Motion generators (one yield = one frame)
# =============================================================================
def wait(seconds):
    for _ in range(int(round(seconds / DT))):
        yield


def apply(arm, q):
    """Set robot.q = q, unless that pose would hit the obstacle: then hold the arm (one frame at a time) until the
    way is clear (Lab 5 Q2/3 is_collision on the NEXT pose). Returns True if it had to wait."""
    waited = False
    while arm.hits([q]):
        arm.blocked = waited = True
        yield
    arm.blocked = False
    arm.robot.q = q
    yield
    return waited


def replan(arm, q_start, q_goal):
    """Straight joint path is blocked (obstacle or the arm hitting itself): first try going via the arm's safe upright
    pose, then insert random waypoints (not in collision) until every segment is clear (Lab 5 Q2/3 primitive RRT)."""
    qlim = arm.robot.qlim
    q_waypoints = [np.asarray(q_start, float), np.asarray(q_goal, float)]
    if arm.safe_q is not None:  # known good upright pose: start -> safe pose -> goal
        via = [q_waypoints[0], np.asarray(arm.safe_q, float), q_waypoints[1]]
        segs = [interpolate_waypoints_radians([via[k], via[k + 1]], arm.step) for k in range(2)]
        if not any(arm.unsafe(seg[::2]) for seg in segs):
            return segs[0] + segs[1]
    for _ in range(RRT_TRIES):
        path, bad = [], None
        for i in range(len(q_waypoints) - 1):
            seg = interpolate_waypoints_radians([q_waypoints[i], q_waypoints[i + 1]], arm.step)
            if arm.unsafe(seg):
                bad = i
                break
            path.extend(seg)
        if bad is None:
            return path
        q_rand = q_waypoints[bad]
        for _ in range(50):  # randomly pick a pose that is not in collision
            q_rand = np.clip(q_waypoints[bad] + (2 * np.random.rand(len(q_rand)) - 1) * RRT_SPREAD, qlim[0], qlim[1])
            if not arm.unsafe([q_rand]):
                break
        q_waypoints.insert(bad + 1, q_rand)
    return None


def plan_move(arm, q_goal, note=None):
    """Joint move that looks ahead every frame; if the obstacle appears in the path it RE-PLANS around it. If no way
    round is found the arm is held (apply) and stops before contact."""
    if note:
        arm.note = note
    q_goal = np.asarray(q_goal, float)
    path = interpolate_waypoints_radians([arm.q, q_goal], arm.step)
    if arm.self_hits(path[::2]):  # the straight joint path would push the arm through itself: plan another way
        new = replan(arm, arm.q.copy(), q_goal)
        if new is None:
            raise RuntimeError(f"{arm.name}: no joint path that keeps the arm from passing through itself")
        path = new
        arm.replans += 1
        arm.note = "RE-PLANNED to avoid self-collision"
    i, tries = 0, 0
    while i < len(path):
        if not arm.obstacle.active:
            tries = 0
        if arm.obstacle.active and tries < 3 and arm.hits(path[i:i + LOOKAHEAD:3]):
            tries += 1
            new = replan(arm, arm.q.copy(), q_goal)
            if new is not None:
                path, i = new, 0
                arm.replans += 1
                arm.note = "RE-PLANNED around the obstacle"
                continue
        yield from apply(arm, path[i])
        i += 1


def rmrc(arm, p_end, speed, Rd):
    """Lab 9 Q1 RMRC (with the Lab 6 Q3 trapezoidal blend and the Lab 7 / 9 DLS): straight line to p_end at `speed`
    m/s, holding the tool orientation Rd."""
    robot, qlim = arm.robot, arm.robot.qlim.T
    p_start = np.asarray(robot.fkine(robot.q).t, float)
    p_end = np.asarray(p_end, float)
    steps = max(int(np.linalg.norm(p_end - p_start) / (speed * DT)), 2)
    s = trapezoidal(0, 1, steps).q
    x = np.zeros([3, steps])
    for i in range(steps):
        x[:, i] = (1 - s[i]) * p_start + s[i] * p_end
    W = np.diag([1, 1, 1, 0.1, 0.1, 0.1])  # weighting matrix for the velocity vector (Lab 9)
    i = 0
    while i < steps - 1:
        T = robot.fkine(robot.q).A
        delta_x = x[:, i + 1] - T[:3, 3]
        Ra = T[:3, :3]
        Rdot = (1 / DT) * (Rd - Ra)
        S = Rdot @ Ra.T
        linear_velocity = (1 / DT) * delta_x
        angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
        xdot = W @ np.r_[linear_velocity, angular_velocity]
        J = robot.jacob0(robot.q)
        m = np.sqrt(max(linalg.det(J @ J.T), 0.0))  # measure of manipulability (Lab 7 / 9); max() avoids NaN
        m_lambda = (1 - m / arm.epsilon) * 0.05 if m < arm.epsilon else 0  # DLS only near a singularity
        inv_j = linalg.inv(J.T @ J + m_lambda * np.eye(6)) @ J.T
        qdot = inv_j @ xdot
        for j in range(6):  # joint limits: stop the motor (Lab 9)
            if robot.q[j] + DT * qdot[j] < qlim[j, 0] or robot.q[j] + DT * qdot[j] > qlim[j, 1]:
                qdot[j] = 0
        waited = yield from apply(arm, robot.q + DT * qdot)
        if not waited:  # if the arm had to wait, redo this step from where it stopped
            i += 1


def ik(robot, T, seeds):
    """Lab 3 Q1 ikine_LM from several seeds, first solution inside the joint limits (Lab 4 Q2/3)."""
    qlim = robot.qlim
    for q0 in seeds:
        sol = robot.ikine_LM(T, q0=q0)
        if sol.success and np.all(sol.q >= qlim[0]) and np.all(sol.q <= qlim[1]):
            return sol.q
    return None


def ik_safe(arm, T, seeds):
    """Like ik(), but looks at ALL the seeds and returns the best solution that (a) is inside the joint limits, (b) does
    not put the arm through itself and (c) needs the smallest joint travel. Every solution is first wrapped by whole
    turns to the nearest equivalent of the current q, so a joint never takes the long way round."""
    robot, qlim = arm.robot, arm.robot.qlim
    ref = arm.q
    best, best_cost = None, None
    for q0 in seeds:
        sol = robot.ikine_LM(T, q0=q0)
        if not sol.success:
            continue
        q = ref + (sol.q - ref + pi) % (2 * pi) - pi
        if np.any(q < qlim[0]) or np.any(q > qlim[1]):
            q = np.asarray(sol.q, float)
        if np.any(q < qlim[0]) or np.any(q > qlim[1]) or arm.self_hits([q]):
            continue
        cost = float(np.abs(q - ref).max())
        if best is None or cost < best_cost:
            best, best_cost = q, cost
    return best


def wait_for_sensor(arm, sensor, timeout, what="a car"):
    """The arm does nothing until its sensor reports `what` (sensor.detected). If nothing is seen after `timeout`
    seconds of running time the job faults instead of waiting forever."""
    arm.note = f"waiting for its sensor to detect {what}"
    for _ in range(int(timeout / DT)):
        if sensor.detected:
            rng = getattr(sensor, "range", None)
            arm.note = f"sensor: {what} detected" + (f" ({rng:.3f} m)" if rng is not None else "")
            return
        yield
    raise RuntimeError(f"{arm.name}: its sensor never detected {what}")


# =============================================================================
# Teach-pendant jogging: ONE frame of a hold-to-move jog (called every frame while the button is held)
# =============================================================================
def jog_joint_step(arm, j, sign, speed, dt=DT):
    """Turn joint j (0-based) by sign * speed * dt (rad), clipped to its limit.  Returns None if it moved, or the
    reason it did not (joint limit, obstacle, self-collision, floor)."""
    q = arm.q
    lo, hi = arm.robot.qlim
    goal = q.copy()
    goal[j] = np.clip(q[j] + sign * speed * dt, lo[j], hi[j])
    if abs(goal[j] - q[j]) < 1e-9:
        return f"J{j + 1} is at its {'upper' if sign > 0 else 'lower'} joint limit"
    why = arm.jog_blocked(goal)
    if why and not arm.jog_blocked(q):  # a pose that is already blocked may still be jogged (to get out of it)
        return why
    arm.robot.q = goal
    return None


def jog_cart_step(arm, axis, sign, speed, dt=DT):
    """Move the tool point sign * speed * dt along world x / y / z (axis 0 / 1 / 2) with one RMRC step (Lab 9 Q1, DLS
    near a singularity), holding the orientation the tool had when the jog started.  Returns None or the reason."""
    robot = arm.robot
    q = arm.q
    T = robot.fkine(q).A
    if arm.jog_R is None:
        arm.jog_R = T[:3, :3].copy()
    Ra = T[:3, :3]
    S = ((arm.jog_R - Ra) / dt) @ Ra.T
    v = np.zeros(6)
    v[axis] = sign * speed
    v[3:] = 0.5 * np.array([S[2, 1], S[0, 2], S[1, 0]])  # gently pull the orientation back to the held one
    J = robot.jacob0(q)
    m = float(np.sqrt(max(linalg.det(J @ J.T), 0.0)))
    if m < 0.2 * arm.epsilon:
        return f"too close to a singularity (manipulability {m:.4f}) - joint jog out of it"
    lam = (1 - m / arm.epsilon) * 0.05 if m < arm.epsilon else 0.0
    qdot = linalg.inv(J.T @ J + lam * np.eye(len(q))) @ J.T @ v
    goal = q + dt * qdot
    lo, hi = robot.qlim
    bad = [k + 1 for k in range(len(q)) if goal[k] < lo[k] or goal[k] > hi[k]]
    if bad:
        return f"joint limit reached on J{bad}"
    why = arm.jog_blocked(goal)
    if why and not arm.jog_blocked(q):
        return why
    robot.q = goal
    return None
