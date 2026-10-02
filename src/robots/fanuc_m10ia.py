"""FANUC M-10iA forward kinematics using a DH table.

Link offsets (mm), from ros-industrial/fanuc (m10ia_macro.xacro):
    base -> shoulder : 450
    shoulder offset  : 150
    upper arm        : 600
    elbow offset     : 200
    forearm          : 640
    wrist -> flange  : 100

NOTE: The DH table below is a STARTING POINT. Each student must derive and
document their own DH parameters, then cross-check against the published
table (Jan et al., 2013) and the URDF. Adjust signs/offsets if your
derivation differs.
"""

import math

import numpy as np

# Link offsets in millimetres
D1 = 450.0   # base to shoulder
A1 = 150.0   # shoulder offset
A2 = 600.0   # upper arm
A3 = 200.0   # elbow offset
D4 = 640.0   # forearm
D6 = 100.0   # wrist to flange

# Classic DH table: (theta_offset, d, a, alpha) for each of the 6 joints
DH_TABLE = [
    (0.0,            D1, A1, -math.pi / 2),
    (-math.pi / 2,   0.0, A2, 0.0),
    (0.0,            0.0, A3, -math.pi / 2),
    (0.0,            D4, 0.0, math.pi / 2),
    (0.0,            0.0, 0.0, -math.pi / 2),
    (0.0,            D6, 0.0, 0.0),
]


def dh_transform(theta, d, a, alpha):
    """Return the 4x4 homogeneous transform for one classic DH row."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def forward_kinematics(joint_angles_deg):
    """Compute the flange pose from six joint angles (degrees).

    Returns a 4x4 matrix: rotation in [:3, :3], position (mm) in [:3, 3].
    """
    if len(joint_angles_deg) != 6:
        raise ValueError("Expected 6 joint angles")

    T = np.eye(4)
    for q_deg, (offset, d, a, alpha) in zip(joint_angles_deg, DH_TABLE):
        theta = math.radians(q_deg) + offset
        T = T @ dh_transform(theta, d, a, alpha)
    return T


def flange_position(joint_angles_deg):
    """Return just the (x, y, z) flange position in mm."""
    return forward_kinematics(joint_angles_deg)[:3, 3]


if __name__ == "__main__":
    pose = forward_kinematics([0, 0, 0, 0, 0, 0])
    np.set_printoptions(precision=2, suppress=True)
    print("Flange pose at home position:")
    print(pose)
