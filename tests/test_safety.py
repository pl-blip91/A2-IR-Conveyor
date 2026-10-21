"""Tests for the safety behaviour of the loading station.

Covers the light curtain / unsafe zone, collision prediction with re-planning, the
e-stop interplay and the safety equipment models.
"""

import math

import numpy as np
import pytest
import spatialgeometry as sg

pytest.importorskip("ir_support")
pytest.importorskip("ir_support_extra_robots")

from safety.collision import (  # noqa: E402
    CollisionChecker,
    Obstacle,
    plan_around,
    segment_point_distance,
)
from safety.zones import LightCurtain, Person, SafetyInputs, UnsafeZone  # noqa: E402
from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation, RunState  # noqa: E402
from workcell.scene import build_scene  # noqa: E402


@pytest.fixture
def station(headless_env):
    build_scene(headless_env)
    return LoadingStation(headless_env, n_cars=2, speed_scale=4.0)


def _run_until(station, predicate, max_ticks=6000):
    for _ in range(max_ticks):
        station.tick()
        if predicate():
            return True
    return False


def _snapshot(st):
    return (st.base.x, st.base.y, st.base.yaw, np.array(st.arm.q), st.gripper.opening)


def _frozen(a, b):
    return all(np.allclose(x, y) for x, y in zip(a, b))


# ------------------------------------------------------------------ pure geometry
def test_light_curtain_is_broken_only_when_a_person_crosses_the_gate_line():
    curtain = LightCurtain()
    assert not curtain.broken_by(Person(x=0.0, y=2.7))
    assert curtain.broken_by(Person(x=0.0, y=lc.ZONE_Y[1]))
    assert not curtain.broken_by(Person(x=2.5, y=lc.ZONE_Y[1]))  # far to the side of the gate


def test_unsafe_zone_covers_the_cell_but_not_the_outside():
    zone = UnsafeZone()
    assert zone.occupied_by(Person(x=0.0, y=1.0))
    assert not zone.occupied_by(Person(x=0.0, y=2.7))
    assert not zone.occupied_by(Person(x=3.5, y=1.0))


def test_person_walks_at_constant_speed_and_stops_at_the_target():
    p = Person(x=0.0, y=2.7)
    p.walk_to(0.0, 1.0)
    for _ in range(10):
        p.update(0.5)
    assert (p.x, p.y) == pytest.approx((0.0, 1.0))
    assert not p.moving


def test_safety_inputs_follow_the_person_and_the_forced_flag():
    s = SafetyInputs(Person(x=0.0, y=2.7))
    assert not s.update(0.05)
    s.forced_curtain = True
    assert s.update(0.05) and s.curtain_broken
    s.forced_curtain = False
    assert not s.update(0.05)


def test_path_check_finds_an_obstacle_on_the_tool_path(station):
    ob = Obstacle(radius=0.08)
    chk = CollisionChecker(station.checker.envelope, lambda: np.asarray(station.arm.base.A), [ob], [], 0.04)
    q = lc.Q_TRAVEL
    tip = chk.envelope.points(q, np.asarray(station.arm.base.A))[-1]
    assert chk.config_hit(q) is None
    ob.place(tip)
    assert chk.config_hit(q) == "obstacle"
    ob.place(tip + [1.0, 0, 0])
    assert chk.config_hit(q) is None


def test_static_keepouts_stop_the_arm_being_sent_into_the_belt(station):
    # Joint 2 forward and joint 3 down, with the base parked by the belt, swings the forearm into it.
    station.base.set_pose(lc.STATION.x, lc.WORK_Y, lc.WORK_YAW)
    station._sync()
    hit = station.checker.config_hit(np.radians([0, 100, 100, 0, 0, 0]))
    assert hit in ("workcell structure", "obstacle")


def test_plan_around_returns_empty_when_direct_and_none_when_impossible():
    lo, hi = -np.ones(6), np.ones(6)
    a, b = np.zeros(6), np.full(6, 0.5)
    assert plan_around(a, b, lambda x, y: True, lo, hi) == []
    assert plan_around(a, b, lambda x, y: False, lo, hi, tries=10) is None


# --------------------------------------------------------- light curtain / zone stop
def test_person_entering_stops_the_station_and_it_needs_reset_then_resume(station):
    station.start()
    assert _run_until(station, lambda: station.phase.endswith("picking"))
    station.person_walk_in()
    assert _run_until(station, lambda: station.run_state is RunState.PROTECTIVE)
    assert station.inputs.curtain_broken or station.inputs.zone_occupied
    frozen = _snapshot(station)
    for _ in range(40):
        station.tick()
    assert _frozen(frozen, _snapshot(station))

    assert not station.reset()  # person still in the zone
    assert not station.resume()
    station.person_walk_out()
    assert _run_until(station, lambda: not station.inputs.tripped)
    assert station.run_state is RunState.PROTECTIVE  # leaving alone does not restart
    for _ in range(20):
        station.tick()
    assert station.run_state is RunState.PROTECTIVE

    assert station.reset()
    assert station.run_state is RunState.STOPPED
    assert station.resume()
    assert station.run(max_ticks=20000) is RunState.DONE
    assert station.placed == 2


def test_forced_light_curtain_signal_stops_motion_within_one_tick(station):
    station.start()
    for _ in range(30):
        station.tick()
    station.trip_light_curtain(True)
    station.tick()
    assert station.run_state is RunState.PROTECTIVE
    assert "SAFETY STOP" in station.phase


def test_start_is_refused_while_the_zone_is_occupied(station):
    station.trip_light_curtain(True)
    station.tick()  # the idle station also latches a protective stop
    assert station.run_state is RunState.PROTECTIVE
    with pytest.raises(RuntimeError):
        station.start()


def test_estop_pressed_during_a_protective_stop_needs_both_inputs_clear(station):
    station.start()
    for _ in range(30):
        station.tick()
    station.trip_light_curtain(True)
    station.tick()
    station.emergency_stop()
    assert station.run_state is RunState.ESTOPPED
    station.trip_light_curtain(False)
    station.tick()
    assert not station.reset()  # e-stop still pressed
    station.release_estop()
    assert station.reset() and station.resume()
    assert station.run_state is RunState.RUNNING


def test_beacon_and_beam_show_the_state(station):
    eq = station.equipment
    station.tick()
    assert eq.lamps["green"].state is True and eq.lamps["red"].state is False
    station.emergency_stop()
    station.tick()
    assert eq.lamps["red"].state is True and eq.lamps["green"].state is False
    assert eq.estop_button.state is True  # button shown pressed
    station.release_estop()
    station.reset()
    station.trip_light_curtain(True)
    station.tick()
    assert eq.beam.state is False  # red beam shown


# ---------------------------------------------------- collision prediction / avoidance
def _gap_to_obstacle(station):
    pts = station.checker.envelope.points(station.arm.q, np.asarray(station.arm.base.A))
    d = segment_point_distance(pts[:-1], pts[1:], station.obstacle.centre)
    return float((d - station.checker.envelope.radii).min()) - station.obstacle.radius


@pytest.mark.parametrize("goal_deg", [[60, 10, 20, 0, -80, 0], [-60, 10, 20, 0, -80, 0], [90, -20, 0, 0, -90, 0]])
def test_obstacle_dropped_in_a_joint_move_is_avoided_by_replanning(headless_env, goal_deg):
    build_scene(headless_env)
    st = LoadingStation(headless_env, n_cars=1, speed_scale=1.0)
    goal = np.radians(goal_deg)
    assert st.goto_q(goal)
    for _ in range(4):
        st.tick()
    assert st.drop_obstacle_in_path() is not None
    closest = 1e9
    for _ in range(3000):
        if st._manual is None:
            break
        st.tick()
        closest = min(closest, _gap_to_obstacle(st))
    assert st.replans >= 1
    assert closest > 0.0  # never touched the obstacle's envelope
    assert np.allclose(st.arm.q, goal, atol=1e-6)  # and still got where it was going


def test_obstacle_in_the_base_path_is_steered_round_and_the_job_completes(station):
    station.start()
    assert _run_until(station, lambda: station._motion is not None and station._motion["kind"] == "base"
                      and station.base.cmd_v != 0 and abs(math.sin(station.base.yaw)) < 0.1)
    assert station.drop_obstacle_in_path() is not None
    ob = station.obstacle.centre[:2]
    closest = 1e9
    for _ in range(30000):
        busy = station.tick()
        closest = min(closest, float(np.hypot(station.base.x - ob[0], station.base.y - ob[1])))
        if not busy:
            break
    assert station.run_state is RunState.DONE, station.fault
    assert station.replans >= 1
    assert closest > lc.BASE_CLEARANCE_RADIUS + lc.OBSTACLE_RADIUS  # footprint never reached it
    assert station.placed == 2


def test_obstacle_with_no_way_round_holds_the_arm_until_removed(station):
    station.start()
    assert _run_until(station, lambda: station.phase.endswith("picking"))
    # Put the obstacle right on the car being approached: the goal itself is blocked.
    car = np.asarray(station.cars[0].T)[:3, 3]
    station.place_obstacle(car + [0, 0, 0.12])
    assert _run_until(station, lambda: station.run_state is RunState.BLOCKED, 3000)
    assert station.block_reason
    frozen = _snapshot(station)
    for _ in range(60):
        station.tick()
    assert station.run_state is RunState.BLOCKED
    assert _frozen(frozen, _snapshot(station))
    station.remove_obstacle()
    assert station.run(max_ticks=20000) is RunState.DONE
    assert station.placed == 2


def test_obstacle_too_close_to_steer_round_stops_the_base_until_cleared(station):
    station.start()
    assert _run_until(station, lambda: station._motion is not None and station._motion["kind"] == "base"
                      and station.base.cmd_v != 0 and abs(math.sin(station.base.yaw)) < 0.1)
    # Right in front of the base, inside the distance needed to turn aside.
    heading = np.array([math.cos(station.base.yaw), math.sin(station.base.yaw)])
    xy = np.array([station.base.x, station.base.y]) + heading * 0.25
    station.place_obstacle([xy[0], xy[1], lc.OBSTACLE_RADIUS])
    assert _run_until(station, lambda: station.run_state is RunState.BLOCKED, 2000)
    pose = (station.base.x, station.base.y)
    for _ in range(40):
        station.tick()
    assert (station.base.x, station.base.y) == pytest.approx(pose)
    station.remove_obstacle()
    assert station.run(max_ticks=30000) is RunState.DONE


def test_estop_while_blocked_then_recovery_returns_to_blocked_or_running(station):
    station.start()
    assert _run_until(station, lambda: station.phase.endswith("picking"))
    station.place_obstacle(np.asarray(station.cars[0].T)[:3, 3] + [0, 0, 0.12])
    assert _run_until(station, lambda: station.run_state is RunState.BLOCKED, 3000)
    station.emergency_stop()
    station.release_estop()
    assert station.reset() and station.resume()
    assert station.run_state is RunState.BLOCKED
    station.remove_obstacle()
    assert station.run(max_ticks=20000) is RunState.DONE


# ------------------------------------------------------------------- equipment models
def test_safety_equipment_surrounds_the_cell_with_one_gated_entrance(station):
    shapes = station.equipment.shapes
    assert len(shapes) > 20
    x0, x1 = lc.ZONE_X
    y0, y1 = lc.ZONE_Y
    g0, g1 = lc.GATE_X
    # No fence panel blocks the gate opening on the +y side.
    for s in shapes:
        if not isinstance(s, sg.Cuboid):
            continue
        sx, sy, sz = s.T[:3, 3]
        w, d, h = s.scale
        if abs(sy - y1) < 1e-6 and h == pytest.approx(lc.FENCE_HEIGHT) and w > 0.5:
            assert not (sx - w / 2 < g0 + 1e-6 and sx + w / 2 > g1 - 1e-6)
    # The mobile base's whole travel range sits inside the fence.
    assert x0 < min(lc.TABLE_SLOT_XS) and max(lc.BELT_SLOT_XS) < x1
    assert y0 < lc.WORK_Y < y1 and lc.STAGING_Y < y1
