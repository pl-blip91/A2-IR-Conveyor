"""Tests for the teach pendant window. Skipped when there is no display (or no Tk)."""

import numpy as np
import pytest

pytest.importorskip("ir_support")
pytest.importorskip("ir_support_extra_robots")
tk = pytest.importorskip("tkinter")

from gui.app import STATE_COLOURS, TeachPendant  # noqa: E402
from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation, RunState  # noqa: E402


class _Env:
    """The smallest stand-in for Swift that the station needs."""

    def add(self, obj):
        pass

    def step(self, dt):
        pass


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.geometry('+0+0')
    station = LoadingStation(_Env(), n_cars=2, speed_scale=4.0)
    pendant = TeachPendant(root, station)
    root.update()  # map the window so synthetic mouse events are delivered
    yield pendant
    pendant._running = False
    root.destroy()


def _tick(app, n=1):
    for _ in range(n):
        app.station.tick()
    app.refresh()
    app.root.update_idletasks()


def _enabled(widget):
    return not widget.instate(["disabled"]) if hasattr(widget, "instate") else str(widget["state"]) != "disabled"


def test_every_state_has_a_banner_colour():
    assert {s.value for s in RunState} <= set(STATE_COLOURS)


def test_start_button_runs_the_job_and_banner_follows(app):
    assert app.banner.cget("text") == "IDLE"
    assert _enabled(app.btn_start)
    app.btn_start.invoke()
    _tick(app, 5)
    assert app.banner.cget("text") == "RUNNING"
    assert not _enabled(app.btn_start)
    assert app.status_vars["phase"].get().startswith("car 1/2")


def test_estop_button_stops_and_shows_a_red_banner_then_requires_release_reset_resume(app):
    app.btn_start.invoke()
    _tick(app, 20)
    app.btn_estop.invoke()
    _tick(app)
    assert app.banner.cget("text").startswith("ESTOPPED")
    assert app.banner.cget("bg") == STATE_COLOURS["ESTOPPED"][0]
    assert app.status_vars["estop"].get() == "PRESSED"
    assert not _enabled(app.btn_reset) and not _enabled(app.btn_resume)  # reset needs the release first
    q = np.array(app.station.arm.q)
    _tick(app, 20)
    assert np.allclose(app.station.arm.q, q)

    app.btn_release.invoke()
    _tick(app)
    assert app.banner.cget("text").startswith("ESTOPPED")  # releasing alone changes nothing
    assert _enabled(app.btn_reset) and not _enabled(app.btn_resume)
    app.btn_reset.invoke()
    _tick(app)
    assert app.banner.cget("text") == "STOPPED"
    assert _enabled(app.btn_resume)
    _tick(app, 10)
    assert app.banner.cget("text") == "STOPPED"
    app.btn_resume.invoke()
    _tick(app)
    assert app.banner.cget("text") == "RUNNING"


def test_holding_a_joint_jog_button_moves_the_joint_and_releasing_stops_it(app):
    j = 0
    plus = app.jog_buttons[2 * j + 1]
    q0 = app.station.arm.q[j]
    plus.event_generate("<ButtonPress-1>")
    _tick(app, 10)
    moved = app.station.arm.q[j] - q0
    assert moved == pytest.approx(lc.JOG_JOINT_SPEED * app.station.dt * 10, rel=1e-6)
    plus.event_generate("<ButtonRelease-1>")
    _tick(app, 10)
    assert app.station.arm.q[j] - q0 == pytest.approx(moved)


def test_leaving_the_button_while_held_stops_the_jog(app):
    minus = app.jog_buttons[0]
    minus.event_generate("<ButtonPress-1>")
    _tick(app, 3)
    minus.event_generate("<Leave>")
    q = np.array(app.station.arm.q)
    _tick(app, 10)
    assert np.allclose(app.station.arm.q, q)


def test_cartesian_jog_buttons_move_the_tool(app):
    z_plus = app.jog_buttons[12 + 2 * 2 + 1]  # after 12 joint buttons: X-, X+, Y-, Y+, Z-, Z+
    z0 = app.station.arm.fkine(app.station.arm.q).t[2]
    z_plus.event_generate("<ButtonPress-1>")
    _tick(app, 10)
    z_plus.event_generate("<ButtonRelease-1>")
    assert app.station.arm.fkine(app.station.arm.q).t[2] > z0 + 0.03


def test_jog_buttons_are_disabled_while_the_job_runs(app):
    assert all(str(b["state"]) == "normal" for b in app.jog_buttons)
    app.btn_start.invoke()
    _tick(app, 3)
    assert all(str(b["state"]) == "disabled" for b in app.jog_buttons)


def test_go_to_joint_state_from_the_entries(app):
    goal = [20, 10, 25, 10, -70, 15]
    for e, v in zip(app.goto_entries, goal):
        e.delete(0, "end")
        e.insert(0, str(v))
    app.btn_go.invoke()
    _tick(app, 400)
    assert np.allclose(np.degrees(app.station.arm.q), goal, atol=1e-6)


def test_bad_goto_input_is_reported_not_raised(app):
    app.goto_entries[0].delete(0, "end")
    app.goto_entries[0].insert(0, "abc")
    app.btn_go.invoke()
    _tick(app)
    assert "enter six numbers" in app.status_vars["message"].get()


def test_light_curtain_checkbox_triggers_a_protective_stop(app):
    app.btn_start.invoke()
    _tick(app, 10)
    app.curtain_var.set(True)
    app.root.update()
    # Checkbutton command is bound to the variable; call it as a click would.
    app.station.trip_light_curtain(app.curtain_var.get())
    _tick(app, 2)
    assert app.banner.cget("text").startswith("PROTECTIVE")
    assert app.status_vars["curtain"].get() == "BROKEN"


def test_drop_obstacle_button_with_nothing_moving_explains_itself(app):
    app._drop_obstacle()
    _tick(app)
    assert "nothing is moving" in app.status_vars["message"].get()


def test_event_log_shows_station_events(app):
    app.btn_estop.invoke()
    _tick(app)
    text = app.log.get("1.0", "end")
    assert "EMERGENCY STOP pressed" in text
