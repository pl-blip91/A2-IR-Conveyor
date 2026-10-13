"""Demo of the loading station: a Yaskawa MH5 on a TurtleBot loads toy cars onto the belt.

    To Run:
    python src/loading_demo.py                  # load 3 cars at real-time speed

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets `import workcell` work

import swift  # noqa: E402

from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation, RunState  # noqa: E402
from workcell.scene import build_scene  # noqa: E402

# Scripted e-stop test, in simulated seconds after the start of the run.
ESTOP_AT, RELEASE_AT, RESET_AT, RESUME_AT = 20.0, 24.0, 26.0, 28.0


def run_demo(env, cars: int = lc.MAX_CARS, speed: float = 1.0, estop_test: bool = False,
             log=print) -> RunState:
    """Build the scene, run the loading job and return the final state."""
    build_scene(env)
    station = LoadingStation(env, n_cars=cars, speed_scale=speed)
    if hasattr(env, "set_camera_pose"):
        env.set_camera_pose([1.2, -2.6, 1.9], [-0.1, 0.4, 0.5])

    station.start()
    last_phase, steps = None, set()
    while station.tick():
        if station.phase != last_phase:
            last_phase = station.phase
            log(f"[{station.sim_time:6.1f}s] {station.run_state.value:8s} {station.phase}")
        if estop_test:  # one action per threshold, so this never blocks or loops on its own
            t = station.sim_time
            for name, when, action in (
                ("estop", ESTOP_AT, station.emergency_stop),
                ("release", RELEASE_AT, station.release_estop),
                ("reset", RESET_AT, station.reset),
                ("resume", RESUME_AT, station.resume),
            ):
                if t >= when and name not in steps:
                    steps.add(name)
                    action()
                    log(f"[{t:6.1f}s] e-stop test: {name} -> {station.run_state.value}")
    log(f"finished: {station.run_state.value}, {station.placed} car(s) placed, "
        f"{station.sim_time:.0f} s of simulated time")
    if station.fault:
        log(f"fault: {station.fault}")
    return station.run_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cars", type=int, default=lc.MAX_CARS, choices=range(1, lc.MAX_CARS + 1))
    parser.add_argument("--speed", type=float, default=1.0, help="speed multiplier (default 1)")
    parser.add_argument("--estop-test", action="store_true", help="script an e-stop press/release/reset/resume")
    args = parser.parse_args()

    env = swift.Swift()
    env.launch(realtime=True)
    try:
        run_demo(env, args.cars, args.speed, args.estop_test)
        env.hold()  # keep the browser tab open; Ctrl+C to quit
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
