# Robot 2 - Painting: Yaskawa Motoman MPX3500 (Aarav)

Everything in this folder is the painting robot. It waits for its laser gate to see a car, sprays it with a forward and
a return sine-wave pass that follows the car's surface 20 cm above it, then goes home while the belt brings the next car.

| File | What to show |
|---|---|
| `paint_station.py` | DH model from the CAD (`dh_tables`), link meshes, spray can tool, `generate_sine_pass()` path following the height map, `resample_constant_speed()`, `path_to_joints()` sequential IK, `find_spray_start_pose()`, `max_reach()` |
| `station.py` | `PaintingStation.run()` (the robot's part of production), `paint_cycle()` / `paint_pass()` (paint dots on the car), `home_and_advance()` (arm homes while the belt moves) |
| `meshes/` | MPX3500 STL files |

Guard fence: the enclosure round this robot is an octagon whose sides are 3.29 m from the S axis in every direction.
`max_reach()` measures the full horizontal reach over the whole joint range (2.993 m with the spray can) and the line
prints the check at start-up, so the arm cannot reach the fence even fully stretched out.
