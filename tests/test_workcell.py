"""Tests for the Toy Car Production line."""

import numpy as np
import pytest

pytest.importorskip("spatialgeometry")

from workcell import config as cfg  # noqa: E402


def _headless_swift():
    swift = pytest.importorskip("swift")
    env = swift.Swift()
    env.launch(headless=True)
    env._send_socket = lambda *args, **kwargs: "0"  # no browser in the tests
    return env


def test_stations_are_ordered_and_on_the_belt():
    xs = [s.x for s in cfg.STATIONS]
    assert xs == sorted(xs)
    assert all(0 < x < cfg.BELT_LENGTH for x in xs)
    assert [s.key for s in cfg.STATIONS] == ["load", "paint", "label", "sort"]


def test_robot_bases_face_the_belt():
    for s in cfg.STATIONS:
        forward = s.robot_base.R[:, 0]
        assert forward[1] * s.side < 0
        assert abs(forward[0]) < 1e-9
        assert s.robot_base.t[2] == pytest.approx(cfg.PAD_HEIGHT)


def test_pads_clear_the_belt_and_each_other():
    half = cfg.PAD_SIZE[1] / 2
    assert cfg.PAD_OFFSET_Y - half > cfg.BELT_WIDTH / 2
    for a, b in zip(cfg.STATIONS, cfg.STATIONS[1:]):
        assert b.x - a.x > cfg.PAD_SIZE[0]


def test_cars_sit_on_the_belt():
    for x, _ in cfg.CARS:
        assert 0 < x < cfg.BELT_LENGTH


def test_scene_builds_and_steps_in_swift():
    from workcell.scene import build_scene

    env = _headless_swift()
    scene = build_scene(env)
    env.step(0.05)
    assert set(scene["pads"]) == {"load", "paint", "label", "sort"}
    assert len(scene["cars"]) == len(cfg.CARS)
    for car, (x, _) in zip(scene["cars"], cfg.CARS):
        assert np.allclose(car.T[:3, 3], [x, 0, cfg.BELT_HEIGHT + cfg.CAR_SIZE[2] / 2])


def test_robot_can_be_mounted_on_a_pad():
    pytest.importorskip("ir_support")
    from ir_support.robots import UR3e
    from workcell.scene import build_scene

    env = _headless_swift()
    build_scene(env)
    robot = UR3e(base=cfg.station("sort").robot_base)
    robot.add_to_env(env)
    env.step(0.05)
