"""Builds the initial sketch of the Toy Car Production workcell in Swift.

Just boxes: one for the conveyor, one pad per station where a robot will
stand, and a few toy cars on the belt. Nothing moves.
"""

import spatialgeometry as sg
from spatialmath import SE3

from . import config as cfg


def _box(env, size, centre, colour):
    shape = sg.Cuboid(scale=list(size), pose=SE3(*centre), color=list(colour))
    env.add(shape)
    return shape


def build_scene(env) -> dict:
    """Add the belt, robot pads and cars to ``env``; return the shapes."""
    belt = _box(
        env,
        (cfg.BELT_LENGTH, cfg.BELT_WIDTH, cfg.BELT_HEIGHT),
        (cfg.BELT_LENGTH / 2, 0, cfg.BELT_HEIGHT / 2),
        (0.15, 0.15, 0.17, 1.0),
    )

    pads = {
        st.key: _box(
            env, cfg.PAD_SIZE, (st.x, st.side * cfg.PAD_OFFSET_Y, cfg.PAD_HEIGHT / 2), cfg.PAD_COLOUR
        )
        for st in cfg.STATIONS
    }

    cars = [
        _box(
            env,
            cfg.CAR_SIZE,
            (x, 0, cfg.BELT_HEIGHT + cfg.CAR_SIZE[2] / 2),
            cfg.CAR_COLOURS[colour] if colour else cfg.UNPAINTED,
        )
        for x, colour in cfg.CARS
    ]

    if hasattr(env, "set_camera_pose"):
        env.set_camera_pose([cfg.BELT_LENGTH / 2, -5.0, 3.0], [cfg.BELT_LENGTH / 2, 0.0, 0.5])
    return {"belt": belt, "pads": pads, "cars": cars}
