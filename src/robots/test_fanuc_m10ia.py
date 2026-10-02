import numpy as np
import pytest

from src.robots.fanuc_m10ia import forward_kinematics, flange_position


def test_pose_is_valid_transform():
    T = forward_kinematics([10, -20, 30, 40, -50, 60])
    R = T[:3, :3]
    assert T.shape == (4, 4)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_position_within_max_reach():
    # Sum of all link lengths is an upper bound on reach
    max_reach = 450 + 150 + 600 + 200 + 640 + 100
    pos = flange_position([30, 45, -30, 90, 20, 0])
    assert np.linalg.norm(pos) <= max_reach


def test_wrong_number_of_joints_raises():
    with pytest.raises(ValueError):
        forward_kinematics([0, 0, 0])
