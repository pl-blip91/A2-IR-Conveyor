"""Constants for the loading station: slots, base driving, arm poses, vision, jogging and safety."""

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
# Same pose with the tool 0.08 m higher (same x, y and orientation). Used whenever a car is in
# the gripper, so the car clears the belt top (0.6 m) while the base docks beside the belt.
Q_CARRY = np.radians([0, -38.8, -17.1, 0, -111.7, 0])
Q_IK_SEED = np.radians([0, 20, 45, 0, -115, 0])  # near the pre-grasp pose

PRE_HEIGHT = 0.12  # approach/retreat distance above the grasp
PLACE_PRE_HEIGHT = 0.12  # same, for placing on the belt (higher so the swung-in car clears the belt edge)
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

# --- Visual servoing (simulated eye-in-hand camera) ---------------------------------
# Cars are dropped on the supply table by a person, so each one sits a little off its
# nominal slot. The arm finds it with image-based visual servoing before it descends.
PICK_JITTER_XY = 0.025  # m, max random offset of a car from its slot (each axis)
PICK_JITTER_YAW = math.radians(10)  # rad, max random rotation of a car about z
JITTER_SEED = 41013
CAM_FOCAL = 0.006  # m
CAM_PIXEL = 10e-6  # m per pixel
CAM_RES = 1024  # square image, principal point at the centre
IBVS_GAIN = 1.5  # 1/s
IBVS_TOL_PX = 0.7  # rms feature error (pixels) that counts as aligned
IBVS_MAX_TICKS = 600
IBVS_MAX_LIN = 0.10  # m/s
IBVS_MAX_ANG = 0.6  # rad/s

# --- Manual (teach/jog) control -------------------------------------------------------
JOG_JOINT_SPEED = math.radians(25)  # rad/s
JOG_CART_SPEED = 0.10  # m/s
MANUAL_MOVE_SPEED_SCALE = 1.0

# --- Safety layout (world frame, metres) ------------------------------------------------
# A guarded cell surrounds the part of the line the mobile MH5 works in. The only way in
# is a gate in the +y fence, which is covered by a light curtain.
ZONE_X = (-1.7, 1.7)  # extent of the guarded cell along the belt
ZONE_Y = (0.25, 1.6)  # from the belt edge out to the fence
FENCE_HEIGHT = 1.2
FENCE_THICKNESS = 0.03
GATE_X = (-0.5, 0.5)  # opening in the +y fence
CURTAIN_HEIGHT = 1.1
PERSON_RADIUS = 0.22
PERSON_SPEED = 0.8  # m/s
PERSON_OUTSIDE = (0.0, 2.7)  # where the person starts and returns to
PERSON_INSIDE = (0.0, 1.0)  # where the person walks to when entering the cell

# --- Collision checking -----------------------------------------------------------------
OBSTACLE_RADIUS = 0.05
COLLISION_MARGIN = 0.025  # extra clearance kept around obstacles
LOOKAHEAD_TICKS = 40  # how much of the planned joint path is checked ahead of the arm
BASE_LOOKAHEAD = 0.6  # m of the base's planned drive that is checked ahead
BASE_CLEARANCE_RADIUS = 0.27  # base plus the stowed arm and a carried car
REPLAN_TRIES = 300
REPLAN_SEED = 7
