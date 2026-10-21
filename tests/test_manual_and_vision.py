"""Tests for the teach/jog interface, go-to-joint-state, re-arming and visual servoing."""

import math

import numpy as np
import pytest
from spatialmath import SE3

pytest.importorskip("ir_support")
pytest.importorskip("ir_support_extra_robots")

from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation, RunState  # noqa: E402
from workcell.scene import build_scene  # noqa: E402
from workcell.vision import Camera, CarServo, camera_twist_to_tool_twist, car_top_corners, limit_twist  # noqa: E402


@pytest.fixture
def station(headless_env):
    build_scene(headless_env)
    return LoadingStation(headless_env, n_cars=2, speed_scale=4.0)


def _ticks(st, n):
    for _ in range(n):
        st.tick()


# ----------------------------------------------------------------------- jogging
def test_joint_jog_moves_only_the_selected_joint_at_the_jog_speed(station):
    q0 = np.array(station.arm.q)
    assert station.set_jog("joint", 2, +1)
    _ticks(station, 20)
    station.stop_jog()
    q1 = np.array(station.arm.q)
    expected = lc.JOG_JOINT_SPEED * station.dt * 20
    assert q1[2] - q0[2] == pytest.approx(expected, rel=1e-6)
    assert np.allclose(np.delete(q1, 2), np.delete(q0, 2))
    _ticks(station, 10)
    assert np.allclose(station.arm.q, q1)  # releasing the button stops the motion


def test_joint_jog_stops_at_the_joint_limit(station):
    lo, hi = station.arm.qlim_array
    station.set_jog("joint", 0, +1)
    _ticks(station, 400)
    assert station.arm.q[0] <= hi[0] + 1e-9
    assert "limit" in station.message


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_cartesian_jog_moves_the_tool_along_one_world_axis(station, axis, sign):
    t0 = station.arm.fkine(station.arm.q)
    assert station.set_jog("cart", axis, sign)
    _ticks(station, 10)
    station.stop_jog()
    t1 = station.arm.fkine(station.arm.q)
    moved = t1.t - t0.t
    expected = sign * lc.JOG_CART_SPEED * station.dt * 10
    assert moved[axis] == pytest.approx(expected, rel=0.15)  # the stowed pose is close to singular
    others = np.delete(moved, axis)
    assert np.all(np.abs(others) < 3e-3)
    assert np.linalg.norm(t1.R - t0.R) < 5e-3  # orientation is held while translating


def test_jog_is_refused_while_a_job_is_running_or_stopped_mid_job(station):
    station.start()
    _ticks(station, 30)
    assert not station.set_jog("joint", 0, 1)
    station.emergency_stop()
    station.release_estop()
    station.reset()
    assert station.run_state is RunState.STOPPED
    assert not station.set_jog("joint", 0, 1)  # would make the paused trajectory jump on resume


def test_jog_is_refused_while_estopped_or_in_a_protective_stop(station):
    station.emergency_stop()
    assert not station.set_jog("joint", 0, 1)
    station.release_estop()
    station.reset()
    assert station.resume()
    station.trip_light_curtain(True)
    station.tick()
    assert not station.set_jog("cart", 0, 1)


def test_estop_cancels_a_jog_in_progress(station):
    station.set_jog("joint", 1, +1)
    _ticks(station, 5)
    q = np.array(station.arm.q)
    station.emergency_stop()
    _ticks(station, 20)
    assert np.allclose(station.arm.q, q)
    station.release_estop()
    station.reset()
    station.resume()
    _ticks(station, 20)
    assert np.allclose(station.arm.q, q)  # the held button did not come back by itself


def test_jog_will_not_drive_the_arm_into_the_obstacle(station):
    tip = station.checker.envelope.points(station.arm.q, np.asarray(station.arm.base.A))[-1]
    station.place_obstacle(tip + [0.0, -0.10, -0.17])  # below and a little to the side of the gripper
    q = np.array(station.arm.q)
    station.set_jog("cart", 2, -1)  # jog down
    _ticks(station, 60)
    assert station.checker.config_hit(station.arm.q) is None
    assert "blocked" in station.message
    assert station.arm.fkine(station.arm.q).t[2] < station.arm.fkine(q).t[2] + 1e-9


# --------------------------------------------------------------- go to joint state
def test_goto_q_reaches_the_commanded_joint_state(station):
    goal = np.radians([20, 10, 25, 10, -70, 15])
    assert station.goto_q(goal)
    _ticks(station, 400)
    assert np.allclose(station.arm.q, goal, atol=1e-6)
    assert station.run_state is RunState.IDLE


def test_goto_q_refuses_out_of_limit_or_colliding_targets(station):
    assert not station.goto_q(np.radians([200, 0, 0, 0, 0, 0]))
    assert "limit" in station.message
    tip = station.checker.envelope.points(lc.Q_TRAVEL, np.asarray(station.arm.base.A))[-1]
    station.place_obstacle(tip)
    assert not station.goto_q(lc.Q_TRAVEL)
    assert "hit" in station.message


def test_goto_q_and_home_work_after_a_job(station):
    assert station.run(max_ticks=30000) is RunState.DONE
    goal = np.radians([30, 0, 30, 0, -60, 0])
    assert station.goto_q(goal)
    _ticks(station, 400)
    assert np.allclose(station.arm.q, goal, atol=1e-6)
    assert station.go_home()
    _ticks(station, 400)
    assert np.allclose(station.arm.q, lc.Q_TRAVEL, atol=1e-6)


def test_goto_q_stops_on_estop_and_does_not_continue_after_resume(station):
    goal = np.radians([60, 10, 20, 0, -80, 0])
    assert station.goto_q(goal)
    _ticks(station, 5)
    station.emergency_stop()
    q = np.array(station.arm.q)
    _ticks(station, 30)
    assert np.allclose(station.arm.q, q)
    station.release_estop()
    station.reset()
    station.resume()
    _ticks(station, 100)
    assert np.allclose(station.arm.q, q)  # the interrupted move was cancelled, not resumed


# ------------------------------------------------------------------------- re-arm
def test_rearm_restores_a_finished_cell_so_it_can_run_again(station):
    assert station.run(max_ticks=30000) is RunState.DONE
    assert station.rearm()
    assert station.run_state is RunState.IDLE and station.placed == 0
    for k, car in enumerate(station.cars):
        assert np.allclose(np.asarray(car.T)[:3, 3][2], station.pick_pose(k).t[2])
        assert abs(np.asarray(car.T)[0, 3] - station.pick_pose(k).t[0]) < lc.PICK_JITTER_XY + 1e-9
    assert station.run(max_ticks=30000) is RunState.DONE
    assert station.placed == 2


def test_rearm_is_refused_while_running_or_estopped(station):
    station.start()
    _ticks(station, 10)
    assert not station.rearm()
    station.emergency_stop()
    assert not station.rearm()


def test_a_fault_is_recoverable_by_rearm(station):
    station.start()
    station.run_state = RunState.FAULT  # as if a StationFault had been raised
    station.fault = "test"
    assert station.rearm()
    assert station.fault is None and station.run_state is RunState.IDLE


# -------------------------------------------------------------------- estop at idle
def test_estop_in_idle_latches_and_needs_reset_then_resume(station):
    station.emergency_stop()
    assert station.run_state is RunState.ESTOPPED
    station.release_estop()
    _ticks(station, 10)
    assert station.run_state is RunState.ESTOPPED
    assert station.reset() and station.resume()
    assert station.run_state is RunState.IDLE


# ---------------------------------------------------------------- visual servoing
def test_camera_projects_a_point_on_the_axis_to_the_image_centre():
    cam = Camera()
    uv, z = cam.project(SE3(0, 0, 0), np.array([[0.0, 0.0, 0.3]]))
    assert uv[0] == pytest.approx([cam.res / 2, cam.res / 2])
    assert z[0] == pytest.approx(0.3)


def test_ibvs_converges_from_an_offset_and_rotated_start():
    cam = Camera()
    car = SE3(0.0, 0.0, 0.03)
    goal = SE3(0, 0, 0.2) * SE3.Rx(math.pi)  # camera 0.2 m above the car frame, looking down
    servo = CarServo(cam, goal)
    # Start 3 cm and 2 cm off, rotated 10 degrees. The car frame is at z = 0.03 in the world.
    cam_pose = SE3(0.03, -0.02, 0.23) * SE3.Rx(math.pi) * SE3.Rz(math.radians(10))
    dt = 0.05
    for _ in range(600):
        v, err = servo.twist(cam_pose, car)
        if err < 0.3:
            break
        v = limit_twist(v)
        cam_pose = cam_pose * SE3.Delta(v * dt)
    assert err < 0.3
    assert np.linalg.norm(cam_pose.t[:2]) < 1e-3
    assert cam_pose.t[2] == pytest.approx(0.23, abs=2e-3)


def test_camera_twist_to_tool_twist_adds_the_lever_arm():
    pose = SE3()
    v = np.array([0, 0, 0, 0, 0, 1.0])  # pure rotation about the camera z axis
    tool_point = np.array([0.1, 0.0, 0.0])
    out = camera_twist_to_tool_twist(v, pose, tool_point)
    assert out[:3] == pytest.approx([0.0, 0.1, 0.0])  # w x r
    assert out[3:] == pytest.approx([0, 0, 1.0])


def test_corners_are_the_car_top_face():
    c = car_top_corners()
    assert c.shape == (4, 3)
    assert np.allclose(c[:, 2], 0.03)


def test_pick_succeeds_for_every_car_despite_the_random_offsets(station):
    offsets = [np.linalg.norm((np.asarray(c.T)[:3, 3] - station.pick_pose(k).t)[:2]) for k, c in enumerate(station.cars)]
    assert max(offsets) > 0.005  # the cars really are off their slots
    assert station.run(max_ticks=30000) is RunState.DONE, station.fault
    assert station.placed == 2
    assert station.vision_error_px < lc.IBVS_TOL_PX
    assert any("visual servoing converged" in e for e in station.events)


def test_pick_fails_cleanly_if_the_car_is_far_outside_the_camera_correction(headless_env):
    build_scene(headless_env)
    st = LoadingStation(headless_env, n_cars=1, speed_scale=4.0, jitter=False)
    car = st.cars[0]
    T = np.array(car.T)
    T[0, 3] += 0.40  # way off: out of the camera's view
    car.T = T
    assert st.run(max_ticks=30000) is RunState.FAULT
    assert st.fault
