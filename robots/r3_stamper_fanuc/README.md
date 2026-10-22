# Robot 3 - Stamping: FANUC M-10iA (Pauras)

Everything in this folder is the stamping robot. It waits for its laser gate to see a car, moves to a hover pose above the
flattest part of the roof, presses the "IR" die straight down with RMRC, dwells, lifts straight up and goes home.

| File | What to show |
|---|---|
| `fanuc_m10ia.py` | DH model from the manufacturer drawing, URDF joint limits and axis signs, mesh offsets from `fkine_all` |
| `station.py` | `StampingStation.run()` (the robot's part of production), `stamp_cycle()`, `StampTool`, `letter_strokes()` |
| `meshes/` | FANUC STL files |

Home pose: `Q_HOME` has J1 = +90 deg, i.e. the base joint is turned 90 deg to the robot's left (anticlockwise seen from
above), so the arm rests along the belt instead of out over it.
