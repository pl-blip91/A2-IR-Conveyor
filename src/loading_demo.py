"""Scripted demo of the loading station: a Yaskawa MH5 on a TurtleBot loads toy cars onto the belt.

    To Run:
    python src/loading_demo.py                       # load 3 cars at real-time speed
    python src/loading_demo.py --scenario curtain    # a person walks into the cell mid-job
    python src/loading_demo.py --scenario obstacle   # an obstacle is dropped in the base and arm paths
    python src/loading_demo.py --scenario estop      # e-stop press, release, reset, resume
    python src/loading_demo.py --scenario all        # all of the above in one run

"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets `import workcell` work

import swift  # noqa: E402

from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation, RunState  # noqa: E402
from workcell.scene import build_scene  # noqa: E402

# Scripted e-stop test, in simulated seconds after the start of the run.
ESTOP_AT, RELEASE_AT, RESET_AT, RESUME_AT = 20.0, 24.0, 26.0, 28.0
SCENARIOS = ("estop", "curtain", "obstacle", "all")


def _driving_along_the_lane(st: LoadingStation) -> bool:
    return (st.run_state is RunState.RUNNING and st.base.cmd_v != 0 and abs(math.sin(st.base.yaw)) < 0.1
            and st._motion is not None and st._motion.get("kind") == "base")


def _try_reset(st: LoadingStation) -> None:
    ok = st.reset()
    st._log("reset " + ("accepted" if ok else "REFUSED (a stop input is still active)"))


def _scenario_steps(name: str) -> list:
    """Ordered (label, condition, action, delay) steps.

    Each step fires once, after the one before it, once its condition has held for
    delay simulated seconds (so each state is visible for a moment).
    """
    obstacle = [
        ("obstacle dropped in the base's path", _driving_along_the_lane, lambda st: st.drop_obstacle_in_path(), 0.0),
        ("obstacle dropped in the arm's tool path",
         lambda st: st.run_state is RunState.RUNNING and st.phase.endswith("placing")
         and st._motion is not None and st._motion.get("kind") == "tool",
         lambda st: st.drop_obstacle_in_path(), 0.0),
        ("obstacle removed", lambda st: st.run_state is RunState.BLOCKED,
         lambda st: st.remove_obstacle(), 3.0),
    ]
    curtain = [
        ("person walks into the cell", lambda st: st.run_state is RunState.RUNNING and st.phase.endswith("picking"),
         lambda st: st.person_walk_in(), 0.0),
        ("reset tried while the person is inside", lambda st: st.run_state is RunState.PROTECTIVE,
         _try_reset, 3.0),
        ("person leaves", lambda st: st.run_state is RunState.PROTECTIVE, lambda st: st.person_walk_out(), 1.0),
        ("reset", lambda st: st.run_state is RunState.PROTECTIVE and st.inputs_clear, _try_reset, 2.0),
        ("resume", lambda st: st.run_state is RunState.STOPPED, lambda st: st.resume(), 2.0),
    ]
    estop = [
        ("e-stop pressed", lambda st: st.run_state is RunState.RUNNING and st.phase.endswith("placing"),
         lambda st: st.emergency_stop(), 0.0),
        ("e-stop released", lambda st: st.run_state is RunState.ESTOPPED, lambda st: st.release_estop(), 3.0),
        ("reset", lambda st: st.run_state is RunState.ESTOPPED and st.inputs_clear, _try_reset, 2.0),
        ("resume", lambda st: st.run_state is RunState.STOPPED, lambda st: st.resume(), 2.0),
    ]
    return {"obstacle": obstacle, "curtain": curtain, "estop": estop, "all": obstacle + curtain + estop}[name]


def run_demo(env, cars: int = lc.MAX_CARS, speed: float = 1.0, estop_test: bool = False,
             log=print, scenario: str | None = None) -> RunState:
    """Build the scene, run the loading job and return the final state."""
    build_scene(env)
    station = LoadingStation(env, n_cars=cars, speed_scale=speed)
    if hasattr(env, "set_camera_pose"):
        env.set_camera_pose([1.6, -2.8, 2.2], [-0.1, 0.5, 0.5])

    steps = list(_scenario_steps(scenario)) if scenario else []
    station.start()
    last_phase, fired = None, set()
    seen, true_since = 0, None
    while station.tick():
        if station.phase != last_phase:
            last_phase = station.phase
            log(f"[{station.sim_time:6.1f}s] {station.run_state.value:10s} {station.phase}")
        events = list(station.events)
        for line in events[-(len(events) - seen):] if len(events) > seen else []:
            log(f"           {line}")
        seen = len(events)
        if estop_test:  # one action per threshold, so this never blocks or loops on its own
            t = station.sim_time
            for name, when, action in (
                ("estop", ESTOP_AT, station.emergency_stop),
                ("release", RELEASE_AT, station.release_estop),
                ("reset", RESET_AT, station.reset),
                ("resume", RESUME_AT, station.resume),
            ):
                if t >= when and name not in fired:
                    fired.add(name)
                    action()
                    log(f"[{t:6.1f}s] e-stop test: {name} -> {station.run_state.value}")
        if steps:
            label, condition, action, delay = steps[0]
            if not condition(station):
                true_since = None
            else:
                true_since = station.sim_time if true_since is None else true_since
                if station.sim_time - true_since >= delay:
                    steps.pop(0)
                    true_since = None
                    action(station)
                    log(f"[{station.sim_time:6.1f}s] scenario: {label} -> {station.run_state.value}")
    log(f"finished: {station.run_state.value}, {station.placed} car(s) placed, "
        f"{station.sim_time:.0f} s of simulated time")
    if station.fault:
        log(f"fault: {station.fault}")
    return station.run_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cars", type=int, default=lc.MAX_CARS, choices=range(1, lc.MAX_CARS + 1))
    parser.add_argument("--speed", type=float, default=1.0, help="speed multiplier (default 1)")
    parser.add_argument("--estop-test", action="store_true", help="script an e-stop press/release/reset/resume at fixed times")
    parser.add_argument("--scenario", choices=SCENARIOS, help="script a safety scenario during the job")
    args = parser.parse_args()

    env = swift.Swift()
    env.launch(realtime=True)
    try:
        run_demo(env, args.cars, args.speed, args.estop_test, scenario=args.scenario)
        env.hold()  # keep the browser tab open; Ctrl+C to quit
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
