"""Run the loading station in Swift with the teach pendant.

    To Run:
    python src/main.py                 # 3 cars, normal speed
    python src/main.py --cars 2 --speed 2 --no-jitter

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets `import workcell` work

import swift  # noqa: E402

from workcell import loading_config as lc  # noqa: E402
from workcell.loading_station import LoadingStation  # noqa: E402
from workcell.scene import build_scene  # noqa: E402

CAMERA_POSE = ([1.6, -2.8, 2.2], [-0.1, 0.5, 0.5])  # looks at the loading cell from the belt side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cars", type=int, default=lc.MAX_CARS, choices=range(1, lc.MAX_CARS + 1))
    parser.add_argument("--speed", type=float, default=1.0, help="speed multiplier (default 1)")
    parser.add_argument("--no-jitter", action="store_true", help="put the cars exactly on their slots")
    args = parser.parse_args()

    try:
        from gui.app import run_gui
    except ImportError as err:  # tkinter is missing from some Python installs
        raise SystemExit(f"The teach pendant needs tkinter ({err}). Install a Python with Tk support.")

    env = swift.Swift()
    env.launch(realtime=True)
    build_scene(env)
    station = LoadingStation(env, n_cars=args.cars, speed_scale=args.speed, jitter=not args.no_jitter)
    if hasattr(env, "set_camera_pose"):
        env.set_camera_pose(*CAMERA_POSE)
    env.step(0.05)
    try:
        run_gui(station)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
