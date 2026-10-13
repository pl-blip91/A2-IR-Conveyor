#Tests for the Yaskawa Motoman MH5 model.

import numpy as np
import pytest
from spatialmath import SE3
from spatialmath.base import transl, trotx, troty

pytest.importorskip("ir_support")
pytest.importorskip("roboticstoolbox")

from robots.yaskawa_mh5 import JOINT_LIMITS_DEG, YaskawaMH5  # noqa: E402

# Independent forward kinematics
URDF_JOINTS = [
    ((0.0, 0.0, 0.330), (0, 0, 1)),
    ((0.088, 0.0, 0.0), (0, 1, 0)),
    ((0.0, 0.0, 0.310), (0, -1, 0)),
    ((0.0, 0.0, 0.040), (-1, 0, 0)),
    ((0.305, 0.0, 0.0), (0, -1, 0)),
    ((0.080, 0.0, 0.0), (-1, 0, 0)),
]


def _axis_rotation(axis, angle):
    a = np.asarray(axis, float)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return T


def urdf_link_frames(q, base=None):
    """World pose of base_link, link_1_s ... link_6_t from the URDF joint chain."""
    T = np.eye(4) if base is None else np.array(base.A)
    frames = [T.copy()]
    for (origin, axis), qi in zip(URDF_JOINTS, q):
        T = T @ transl(*origin) @ _axis_rotation(axis, qi)
        frames.append(T.copy())
    return frames


def urdf_tool0(q):
    """URDF ``tool0`` frame: link_6_t with rpy = (180 deg, -90 deg, 0)."""
    return urdf_link_frames(q)[-1] @ troty(-np.pi / 2) @ trotx(np.pi)


@pytest.fixture(scope="module")
def arm():
    return YaskawaMH5()


def test_has_six_joints_and_limits_match_urdf(arm):
    assert arm.n == 6
    expected = np.radians(JOINT_LIMITS_DEG)
    assert np.allclose(arm.qlim.T, expected)
    assert JOINT_LIMITS_DEG == ((-170, 170), (-65, 150), (-70, 190), (-190, 190), (-135, 135), (-360, 360))


def test_zero_pose_is_the_urdf_l_shape(arm):
    T = arm.fkine(np.zeros(6))
    assert np.allclose(T.t, [0.473, 0.0, 0.680])
    # tool0 z points forward (out of the flange)
    assert np.allclose(T.R[:, 2], [1, 0, 0])


def test_dh_forward_kinematics_matches_urdf_chain(arm):
    rng = np.random.default_rng(0)
    lo, hi = arm.qlim_array
    for _ in range(100):
        q = rng.uniform(lo, hi)
        assert np.allclose(arm.fkine(q).A, urdf_tool0(q), atol=1e-9)


def test_tool_offset_moves_the_tcp_along_the_approach_axis():
    arm = YaskawaMH5(tool_offset=0.10)
    q = np.radians([10, 20, 30, 40, -50, 60])
    flange = YaskawaMH5().fkine(q)
    tcp = arm.fkine(q)
    assert np.allclose(tcp.t, flange.t + 0.10 * flange.R[:, 2])


def test_meshes_follow_the_urdf_link_frames_with_a_rotated_base():
    base = SE3(0.3, -0.2, 0.5) * SE3.Rz(0.7)
    arm = YaskawaMH5(base=base)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, 6)
        arm.q = q  # the picture must update immediately, not one assignment late
        for mesh, frame in zip(arm.links_3d, urdf_link_frames(q, base)):
            assert np.allclose(np.asarray(mesh.T), frame, atol=1e-9)


def test_jacobian_is_in_the_world_frame_for_a_rotated_base():
    arm = YaskawaMH5(tool_offset=0.1, base=SE3(0.5, 0.2, 0.15) * SE3.Rz(-np.pi / 2))
    q = np.radians([5, 20, 40, 10, -90, 30])
    dq = 1e-6 * np.array([1, -2, 3, -1, 2, 1.0])
    J = arm.jacob0(q)
    T0, T1 = arm.fkine(q), arm.fkine(q + dq)
    assert np.allclose(J[:3] @ dq, T1.t - T0.t, atol=1e-9)
    # Rotation part: world-frame angular velocity of the tool
    dR = T1.R @ T0.R.T
    omega = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
    assert np.allclose(J[3:] @ dq, omega, atol=1e-9)


def test_kinematics_only_copy_has_same_fkine(arm):
    k = arm.kinematics_only()
    q = np.radians([10, 20, 30, 40, -50, 60])
    assert np.allclose(k.fkine(q).A, arm.fkine(q).A)


def test_adds_to_swift(headless_env):
    arm = YaskawaMH5(base=SE3(1, 0, 0))
    arm.add_to_env(headless_env)
    headless_env.step(0.05)
