# Robot 1 - Loading: Yaskawa MH5 on a TurtleBot3 Waffle (Lachlan)

Everything in this folder is the loading robot. It drives to the supply table, visually servos onto each car, picks it,
drives to the belt and puts it in its slot, where the conveyor's slot photo-eye sees it.

| File | What to show |
|---|---|
| `station.py` | `LoaderStation.run()` (the robot's part of production), `LoaderArm.hits()` / `jog_blocked()` (how the pendant's jog and go-to use the capsule collision model) |
| `loading_station.py` | `sequence()` the whole job, `_travel_to()` / `_drive()` / `_detour_plan()` base driving and steering round obstacles, `_visual_servo()` IBVS, `_arm_joint_move()` look-ahead + re-planning, `_rmrc()` / `_resolved_rate()` RMRC with damping and singularity / joint-limit faults |
| `yaskawa_mh5.py` | DH model and STL meshes of the MH5 |
| `turtlebot_base.py` | differential-drive base with trapezoidal speed profiles |
| `vision.py` | simulated eye-in-hand camera and image-based visual servoing |
| `collision.py` | capsule envelope of the arm, spherical obstacle, static keep-outs, `plan_around()` via-point planner |
| `parallel_gripper.py` | two-finger gripper |
| `loading_config.py` | every number the station uses |

