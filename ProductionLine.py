"""
Toy Car Production Line - integrated Swift workcell.  Run:   python ProductionLine.py

    Robot 1  LOADING   Yaskawa MH5 on a TurtleBot3 (Lachlan)  robots/r1_loader_mh5      loads the cars on the belt
    Robot 2  PAINTING  Yaskawa MPX3500 (Aarav)                robots/r2_painter_mpx3500  sprays them blue, red, green
    Robot 3  STAMPING  FANUC M-10iA (Pauras)                  robots/r3_stamper_fanuc    stamps the 'IR' logo on each roof
    Robot 4  SORTING   UR3e + RG2 (Pauras)                    robots/r4_sorter_ur3e      boxes each car by its colour

"""

import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import swift  # noqa: E402

from integration.production_cell import ToyCarCell  # noqa: E402


def console_listener(cell):
    while True:
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if not line:  # end of input (terminal closed / no stdin)
            return
        cell.cmds.append(("console", line.strip()))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    gui = "--no-gui" not in argv
    root = None
    if gui:
        # macOS: the Tk window must exist BEFORE pygame (the gamepad e-stop) starts SDL, otherwise SDL takes over the
        # Cocoa application object and Tk aborts with "-[SDLApplication macOSVersion]: unrecognised selector".
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # shown once the scene is built
    env = swift.Swift()
    env.launch(realtime=True)
    cell = ToyCarCell(env, factory="--no-factory" not in argv, gamepad="--no-gamepad" not in argv)
    threading.Thread(target=console_listener, args=(cell,), daemon=True).start()
    print("Terminal: final mh5|mpx|fanuc|ur d1 d2 d3 d4 d5 d6  sets a robot's final joint state (degrees). "
          "Close the pendant window or Ctrl+C to quit.")
    try:
        if not gui:
            cell.start_production()
            cell.run()
        else:
            from integration.teach_pendant import run_gui
            run_gui(cell, root)
    except KeyboardInterrupt:
        pass
    finally:
        cell.hw.close()
        env.close()


if __name__ == "__main__":
    main()
