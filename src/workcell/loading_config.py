import math

import numpy as np

from . import config as cfg

STATION = cfg.station("load")
SIDE = STATION.side  # +1: the robot works from the +y side of the belt

# --- Cars to load -----------------------------------------------------------
MAX_CARS = 3
SLOT_PITCH = 0.25  # spacing between neighbouring slots along x

# Belt slots: car 0 goes furthest downstream (the way the belt will carry it).
BELT_SLOT_XS = tuple(STATION.x + (1 - k) * SLOT_PITCH for k in range(MAX_CARS))

# --- Supply table (where the cars are picked from) ----------------------------
SUPPLY_SHIFT = 1.8  # table slot k is this far upstream (-x) of belt slot k
TABLE_SLOT_XS = tuple(x - SUPPLY_SHIFT for x in BELT_SLOT_XS)
TABLE_HEIGHT = cfg.BELT_HEIGHT  # same height as the belt, so pick and place match
TABLE_TOP_THICKNESS = 0.04
TABLE_SIZE = (1.1, cfg.BELT_WIDTH)  # (x, y); the same width as the belt
TABLE_CENTRE_X = STATION.x - SUPPLY_SHIFT
TABLE_COLOUR = (0.60, 0.45, 0.30, 1.0)
LEG_SIZE = 0.04

# --- Base driving ---------------------------------------------------------------
WORK_Y = SIDE * cfg.DOCK_OFFSET_Y  # y of the base while picking or placing
STAGING_Y = SIDE * 0.90  # y of the lane the base turns in
WORK_YAW = -SIDE * math.pi / 2  # heading while working: arm +x points at the belt
BASE_SPEED_SCALE = 1.0

# --- Mounting the arm on the TurtleBot -------------------------------------------
MOUNT_PLATE_SIZE = (0.30, 0.28, 0.012)
MOUNT_PLATE_COLOUR = (0.35, 0.36, 0.40, 1.0)
MOUNT_HEIGHT = 0.138 + MOUNT_PLATE_SIZE[2]  # arm base above the floor (TurtleBot top + plate)

# --- Arm poses and motion -----------------------------------------------------------
# Compact pose used while driving. The gripper points straight down and the whole
# arm (and a carried car) stays within ~0.24 m of the arm base.
Q_TRAVEL = np.radians([0, -45, -35, 0, -100, 0])
Q_IK_SEED = np.radians([0, 20, 45, 0, -115, 0])  # near the pre-grasp pose

PRE_HEIGHT = 0.12  # approach/retreat distance above the grasp
GRASP_LIFT = 0.01  # grip centre this far above the car centre (fingertips clear the surface)
ARM_JOINT_SPEED = math.radians(45)  # rad/s, peak joint speed of joint-space moves
MIN_MOVE_TIME = 1.0  # s
RMRC_SPEED = 0.06  # m/s along the approach and retreat lines
GRIPPER_TIME = 0.8  # s to open or close

# RMRC safety limits
MANIPULABILITY_MIN = 0.002  # stop (fault) below this Yoshikawa manipulability
RMRC_DAMPING = 0.05  # max damping factor for damped least squares near singularities
MANIPULABILITY_DAMP = 0.02  # start damping below this
POSITION_TOLERANCE = 1e-3  # m: end of an RMRC move must be this close to its goal
