# Robot 4 - Sorting: UR3e with an OnRobot RG2 gripper (Pauras)

Everything in this folder is the sorting robot. It waits for its laser gate to see a car at the end of the belt, picks it
with RMRC, carries it over the compartment of its own colour (wrist turned so the car lies along it) and places it.

| File | What to show |
|---|---|
| `ur3e.py` | UR3e DH model and mesh offsets, RG2 gripper meshes with sliding fingers |
| `station.py` | `SortingStation.run()` (the robot's part of production), `sort_cycle()`, `arm_goto()` with `ik_safe()` (never a pose through itself), the three-compartment storage box |
| `meshes/` | UR3e and RG2 STL files |

Self-collision: this is the only robot with link radii (`UR_SELF_RADII`), so `shared/motion.py` checks every planned
pose and every jog step against the arm passing through itself.
