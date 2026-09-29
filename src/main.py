"""Show the Toy Car Production workcell sketch in Swift.

    To Run:
    python src/main.py

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets `import workcell` work

import swift  # noqa: E402

from workcell.scene import build_scene  # noqa: E402


def main() -> None:
    env = swift.Swift()
    env.launch(realtime=True)
    build_scene(env)
    env.step(0.05)
    try:
        env.hold()  # keep the browser tab open; Ctrl+C to quit
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
