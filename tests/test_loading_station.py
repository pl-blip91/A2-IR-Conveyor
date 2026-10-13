#Tests for the loading station: MH5 on a TurtleBot loading toy cars.

import math
import struct
from pathlib import Path

import numpy as np
import pytest
from spatialmath import SE3

pytest.importorskip("ir_support")
pytest.importorskip("ir_support_extra_robots")

from robots.parallel_gripper import TCP_OFFSET  # noqa: E402
from robots.turtlebot_base import FOOTPRINT_RADIUS  # noqa: E402
from robots.yaskawa_mh5 import MESH_DIR, YaskawaMH5, _MESH_FILES  # noqa: E402
from workcell import config as cfg  # noqa: E402
from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import (  # noqa: E402
    LoadingStation,
    RunState,
    StationFault,
    build_loading_area,
)
from workcell.scene import build_scene  # noqa: E402


@pytest.fixture
def station(headless_env):
    build_scene(headless_env)
    return LoadingStation(headless_env, n_cars=3, speed_scale=4.0)


def _read_stl(name):
    data = (Path(MESH_DIR) / f"{name}.stl").read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    v = rec["v"].reshape(-1, 3).astype(float)
    return np.c_[v, np.ones(len(v))]


# ----------------------------------------------------------------------- layout
def test_slots_are_on_the_belt_and_the_table_is_upstream_of_the_belt():
    assert len(lc.BELT_SLOT_XS) == len(lc.TABLE_SLOT_XS) == lc.MAX_CARS
    half_car = cfg.CAR_SIZE[0] / 2
    for x in lc.BELT_SLOT_XS:
        assert half_car <= x <= cfg.BELT_LENGTH - half_car
    table_end = lc.TABLE_CENTRE_X + lc.TABLE_SIZE[0] / 2
    assert table_end < 0  # clear of the belt
    for x in lc.TABLE_SLOT_XS:
        assert abs(x - lc.TABLE_CENTRE_X) + half_car < lc.TABLE_SIZE[0] / 2


def test_neighbouring_slots_do_not_overlap_and_clear_the_placeholder_cars():
    pitch = np.diff(sorted(lc.BELT_SLOT_XS))
    assert np.all(pitch >= cfg.CAR_SIZE[0] + 0.02)
    for x_slot in lc.BELT_SLOT_XS:
        for x_other, _ in cfg.CARS:
            assert abs(x_slot - x_other) >= cfg.CAR_SIZE[0] + 0.05


def test_table_and_belt_have_the_same_height_so_pick_and_place_reach_match():
    assert lc.TABLE_HEIGHT == cfg.BELT_HEIGHT


def test_staging_lane_is_clear_of_the_belt_and_table_for_turning():
    clearance = abs(lc.STAGING_Y) - FOOTPRINT_RADIUS
    assert clearance > cfg.BELT_WIDTH / 2


def test_load_station_has_a_dock_instead_of_a_pad(headless_env):
    scene = build_scene(headless_env)
    assert cfg.station("load").mobile
    assert scene["pads"]["load"].scale[2] < 0.01  # a floor marking, not a pillar
    assert np.allclose(cfg.station("load").robot_base.t, [cfg.station("load").x, cfg.DOCK_OFFSET_Y, 0.0])
    for key in ("paint", "label", "sort"):
        assert scene["pads"][key].scale[2] == pytest.approx(cfg.PAD_HEIGHT)


def test_loading_area_builds_cars_on_the_table(headless_env):
    area = build_loading_area(headless_env, 3)
    assert len(area["cars"]) == 3
    for k, car in enumerate(area["cars"]):
        t = np.asarray(car.T)[:3, 3]
        assert t == pytest.approx([lc.TABLE_SLOT_XS[k], 0, lc.TABLE_HEIGHT + cfg.CAR_SIZE[2] / 2])
    with pytest.raises(ValueError):
        build_loading_area(headless_env, 4)


# ----------------------------------------------------------- reach and clearance
def test_grasp_poses_are_reachable_with_margin(station):
    kin = station.arm.kinematics_only()
    lo, hi = station.arm.qlim_array
    for k in range(3):
        for car in (station.pick_pose(k), station.place_pose(k)):
            grasp = station.grasp_tcp(car)
            pre = SE3(0, 0, lc.PRE_HEIGHT) * grasp
            for goal in (grasp, pre):
                # Goal expressed in the arm frame for a base parked at this slot.
                base = SE3(car.t[0], lc.WORK_Y, lc.MOUNT_HEIGHT) * SE3.Rz(lc.WORK_YAW)
                sol = kin.ikine_LM(base.inv() * goal, q0=lc.Q_IK_SEED)
                assert sol.success
                assert np.all(sol.q > lo + math.radians(5)), "within 5 deg of a lower limit"
                assert np.all(sol.q < hi - math.radians(5)), "within 5 deg of an upper limit"


def test_travel_pose_keeps_the_arm_and_a_carried_car_over_the_base():
    arm = YaskawaMH5(tool_offset=TCP_OFFSET)
    arm.q = lc.Q_TRAVEL
    radius, lowest = 0.0, 1e9
    for i, name in enumerate(_MESH_FILES):
        pts = (np.asarray(arm.links_3d[i].T) @ _read_stl(name).T).T[:, :3]
        radius = max(radius, np.hypot(pts[:, 0], pts[:, 1]).max())
        lowest = min(lowest, pts[:, 2].min())
    tcp = arm.fkine(lc.Q_TRAVEL)
    assert tcp.R[:, 2] == pytest.approx([0, 0, -1], abs=1e-6)  # gripper points down
    car_radius = math.hypot(abs(tcp.t[0]) + cfg.CAR_SIZE[1] / 2, cfg.CAR_SIZE[0] / 2)
    # Stays within ~2.5 cm of the TurtleBot footprint circle, and well inside the
    # distance to the belt edge from the dock (checked below).
    assert radius < FOOTPRINT_RADIUS + 0.03
    assert car_radius < FOOTPRINT_RADIUS + 0.03
    assert lowest >= -1e-4  # nothing goes below the mounting plane (STL float32 noise allowed)
    dock_to_belt_edge = cfg.DOCK_OFFSET_Y - cfg.BELT_WIDTH / 2
    assert max(radius, car_radius) < dock_to_belt_edge  # parking with the car never touches the belt


def test_arm_joint_move_to_pregrasp_stays_above_the_mounting_plane(station):
    # Park the base at the first table slot, as the real sequence does before picking.
    station.base.set_pose(lc.TABLE_SLOT_XS[0], lc.WORK_Y, lc.WORK_YAW)
    station._sync()
    q_pre = station._ik(SE3(0, 0, lc.PRE_HEIGHT) * station.grasp_tcp(station.pick_pose(0)))
    arm = station.arm
    arm.q = lc.Q_TRAVEL
    vertices = [_read_stl(n) for n in _MESH_FILES]
    for s in np.linspace(0, 1, 15):
        arm.q = (1 - s) * lc.Q_TRAVEL + s * q_pre
        for i in range(1, 7):
            pts = (np.asarray(arm.links_3d[i].T) @ vertices[i].T).T[:, 2]
            assert pts.min() > lc.MOUNT_HEIGHT - 1e-3


# ------------------------------------------------------------------ whole job
def test_full_sequence_loads_every_car_onto_its_belt_slot(station):
    state = station.run(max_ticks=20000)
    assert state is RunState.DONE, station.fault
    assert station.placed == 3
    for k, car in enumerate(station.cars):
        assert np.allclose(np.asarray(car.T), station.place_pose(k).A, atol=1e-6)
    # Back at the dock with the arm stowed and the gripper empty.
    assert (station.base.x, station.base.y) == pytest.approx((cfg.station("load").x, lc.WORK_Y))
    assert station.base.yaw == pytest.approx(lc.WORK_YAW)
    assert np.allclose(station.arm.q, lc.Q_TRAVEL, atol=1e-6)
    assert station.status()["holding_car"] is False


def test_base_never_enters_the_belt_or_table_footprint_while_driving(station):
    belt_edge = cfg.BELT_WIDTH / 2
    table_edge = cfg.BELT_WIDTH / 2
    station.start()
    min_y = 1e9
    while station.tick():
        if station.base.cmd_v or station.base.cmd_w:
            min_y = min(min_y, station.base.y - FOOTPRINT_RADIUS)
    assert min_y >= min(belt_edge, table_edge) - 1e-9
    # Rotations (cmd_w) only happen in the staging lane.


def test_turns_only_happen_in_the_staging_lane(station):
    station.start()
    while station.tick():
        if abs(station.base.cmd_w) > 0:
            assert station.base.y == pytest.approx(lc.STAGING_Y)


def test_car_is_carried_with_the_gripper(station):
    seen_carry = False
    station.start()
    car = station.cars[0]
    while station.tick():
        if station.phase.endswith("driving to conveyor"):
            tcp = station.arm.fkine(station.arm.q).t
            assert np.linalg.norm(np.asarray(car.T)[:3, 3] - tcp) <= lc.GRASP_LIFT + 1e-3
            seen_carry = True
        if station.placed:
            break
    assert seen_carry


def test_on_car_placed_callback_is_called(headless_env):
    build_scene(headless_env)
    placed = []
    st = LoadingStation(headless_env, n_cars=1, speed_scale=4.0, on_car_placed=lambda k, c: placed.append(k))
    assert st.run(max_ticks=20000) is RunState.DONE
    assert placed == [0]


# ------------------------------------------------------------------------ RMRC
def test_rmrc_descent_is_a_straight_line_with_constant_orientation(station):
    grasp = station.grasp_tcp(station.pick_pose(0))
    # Park the base where the first car is, as the real sequence does.
    station.base.set_pose(lc.TABLE_SLOT_XS[0], lc.WORK_Y, lc.WORK_YAW)
    station._sync()
    pre = SE3(0, 0, lc.PRE_HEIGHT) * grasp
    station.arm.q = station._ik(pre)
    station._sync()
    start = station.arm.fkine(station.arm.q)
    direction = (grasp.t - start.t) / np.linalg.norm(grasp.t - start.t)

    worst_line, worst_rot = 0.0, 0.0
    for _ in station._rmrc(grasp):
        now = station.arm.fkine(station.arm.q)
        off = (now.t - start.t) - np.dot(now.t - start.t, direction) * direction
        worst_line = max(worst_line, np.linalg.norm(off))
        worst_rot = max(worst_rot, np.linalg.norm(now.R - start.R))
    assert worst_line < 1e-3
    assert worst_rot < 1e-3
    assert np.allclose(station.arm.fkine(station.arm.q).t, grasp.t, atol=1e-3)
    assert station.manipulability > lc.MANIPULABILITY_MIN


def test_rmrc_faults_instead_of_driving_through_a_joint_limit(station):
    station.arm.q = lc.Q_TRAVEL
    far = SE3(2.0, 0.0, 1.5) * SE3.Rx(math.pi)  # well outside the reach
    with pytest.raises(StationFault):
        for _ in station._rmrc(far):
            pass


def test_unreachable_target_is_reported(station):
    with pytest.raises(StationFault):
        station._ik(SE3(3.0, 3.0, 1.0) * SE3.Rx(math.pi))


# --------------------------------------------------------------------- e-stop
def _snapshot(st):
    return (st.base.x, st.base.y, st.base.yaw, np.array(st.arm.q), st.gripper.opening)


def test_estop_halts_latches_and_needs_reset_then_resume(station):
    station.start()
    for _ in range(200):
        station.tick()
    assert station.run_state is RunState.RUNNING

    station.emergency_stop()
    assert station.run_state is RunState.ESTOPPED
    frozen = _snapshot(station)
    for _ in range(40):
        station.tick()
    assert all(np.allclose(a, b) for a, b in zip(frozen, _snapshot(station)))
    assert station.base.cmd_v == 0 and station.base.cmd_w == 0

    # Releasing the button alone must not restart anything.
    station.release_estop()
    for _ in range(40):
        station.tick()
    assert station.run_state is RunState.ESTOPPED
    assert not station.resume()  # cannot resume without a reset

    assert station.reset()
    assert station.run_state is RunState.STOPPED
    for _ in range(40):
        station.tick()
    assert station.run_state is RunState.STOPPED  # reset alone does not restart either
    assert all(np.allclose(a, b) for a, b in zip(frozen, _snapshot(station)))

    assert station.resume()
    assert station.run(max_ticks=20000) is RunState.DONE
    assert station.placed == 3


def test_reset_is_refused_while_the_stop_is_still_pressed(station):
    station.start()
    for _ in range(50):
        station.tick()
    station.emergency_stop()
    assert not station.reset()
    assert station.run_state is RunState.ESTOPPED
    station.release_estop()
    assert station.reset()


def test_cannot_start_twice(station):
    station.start()
    with pytest.raises(RuntimeError):
        station.start()


# ------------------------------------------------------------------------- demo
def test_demo_runs_to_completion(headless_env):
    from loading_demo import run_demo

    lines = []
    assert run_demo(headless_env, cars=1, speed=4.0, log=lines.append) is RunState.DONE
    assert any("finished: DONE" in line for line in lines)


def test_demo_estop_script_stops_then_recovers(headless_env):
    from loading_demo import run_demo

    lines = []
    assert run_demo(headless_env, cars=2, speed=4.0, estop_test=True, log=lines.append) is RunState.DONE
    text = "\n".join(lines)
    for expected in ("estop -> ESTOPPED", "release -> ESTOPPED", "reset -> STOPPED", "resume -> RUNNING"):
        assert expected in text


def test_phase_is_restored_after_resume_and_start_is_refused_while_pressed(station):
    station.emergency_stop()
    with pytest.raises(RuntimeError):
        station.start()
    station.release_estop()
    station.start()
    for _ in range(100):
        station.tick()
    phase = station.phase
    station.emergency_stop()
    assert station.phase == "EMERGENCY STOP"
    station.release_estop()
    station.reset()
    station.resume()
    assert station.phase == phase
