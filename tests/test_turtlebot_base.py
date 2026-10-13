#Tests for the TurtleBot differential-drive base.

import math

import numpy as np
import pytest

pytest.importorskip("ir_support_extra_robots")

from robots.turtlebot_base import (  # noqa: E402
    DriveLimits,
    TurtleBotBase,
    trapezoid_profile,
    wrap_angle,
)

DT = 0.05


def _run(gen):
    n = 0
    for _ in gen:
        n += 1
    return n


def test_profile_reaches_distance_and_stops():
    s, v = trapezoid_profile(1.0, 0.26, 0.25, DT)
    assert s[-1] == pytest.approx(1.0)
    assert v[-1] == 0.0
    assert np.all(np.diff(s) >= -1e-12)
    assert v.max() <= 0.26 + 1e-9


def test_profile_cruises_for_long_moves_and_is_triangular_for_short_ones():
    _, v_long = trapezoid_profile(2.0, 0.26, 0.25, DT)
    assert v_long.max() == pytest.approx(0.26)
    _, v_short = trapezoid_profile(0.02, 0.26, 0.25, DT)
    assert v_short.max() < 0.26


def test_profile_respects_acceleration_limit():
    _, v = trapezoid_profile(1.0, 0.26, 0.25, DT)
    assert np.abs(np.diff(np.r_[0.0, v])).max() / DT <= 0.25 * 1.05


def test_zero_distance_profile_is_trivial():
    s, v = trapezoid_profile(0.0, 0.26, 0.25, DT)
    assert s[-1] == 0 and v[-1] == 0


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_angle(0.3) == pytest.approx(0.3)


def test_drive_forward_and_reverse_follow_the_heading():
    bot = TurtleBotBase(1.0, 2.0, -math.pi / 2)
    _run(bot.drive(0.5, DT))
    assert (bot.x, bot.y) == pytest.approx((1.0, 1.5))
    _run(bot.drive(-0.25, DT))
    assert (bot.x, bot.y) == pytest.approx((1.0, 1.75))
    assert bot.yaw == pytest.approx(-math.pi / 2)


def test_turn_takes_the_short_way_and_stays_on_the_spot():
    bot = TurtleBotBase(0.5, 0.5, -math.pi / 2)
    yaws = []
    for _ in bot.turn_to(math.pi, DT):
        yaws.append(bot.yaw)
        assert (bot.x, bot.y) == (0.5, 0.5)
    assert bot.yaw == pytest.approx(math.pi, abs=1e-9) or bot.yaw == pytest.approx(-math.pi, abs=1e-9)
    # shortest way from -90 deg to 180 deg is -90 deg, so it never passes through 0 deg
    assert all(y < 0 or y > 3.0 or abs(y - math.pi) < 1e-9 for y in yaws)


def test_speeds_stay_within_limits():
    limits = DriveLimits(max_lin=0.2, max_ang=0.8)
    bot = TurtleBotBase(limits=limits)
    peak = 0.0
    for _ in bot.drive(2.0, DT):
        peak = max(peak, abs(bot.cmd_v))
    assert peak == pytest.approx(0.2)
    peak = 0.0
    for _ in bot.turn_to(math.pi / 2, DT):
        peak = max(peak, abs(bot.cmd_w))
    assert peak <= 0.8 + 1e-9
    assert bot.cmd_v == 0 and bot.cmd_w == 0


def test_speed_scale_shortens_the_move():
    slow = _run(TurtleBotBase().drive(1.0, DT, 1.0))
    fast = _run(TurtleBotBase().drive(1.0, DT, 2.0))
    assert fast < slow


def test_wheel_speeds():
    left, right = TurtleBotBase.wheel_speeds(0.2, 0.0)
    assert left == pytest.approx(right)
    left, right = TurtleBotBase.wheel_speeds(0.0, 1.0)
    assert left == pytest.approx(-right) and right > 0


def test_pose_matches_xy_yaw_and_the_model_follows(headless_env):
    bot = TurtleBotBase(0.4, -0.2, 0.3)
    bot.add_to_env(headless_env)
    bot.set_pose(1.0, 2.0, 1.2)
    assert np.allclose(bot.pose.t, [1.0, 2.0, 0.0])
    assert bot.pose.rpy()[2] == pytest.approx(1.2)
    assert np.allclose(np.asarray(bot.model.links_3d[1].T), bot.pose.A)
    headless_env.step(0.05)
