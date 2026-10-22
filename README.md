# Toy Car Production

Lab Assignment 2, 41013 Industrial Robotics.

## Overview

Toy Car Production is a four-station miniature manufacturing line simulated in Swift using Python:

1. **Loading:** an arm loads toy cars onto a conveyor.
2. **Spray painting:** a second arm simulates spray painting.
3. **Logo/label:** a third arm applies a logo or label.
4. **Sorting:** a fourth arm sorts finished cars by colour into storage boxes.

The simulation coordinates conveyor movement, visual classification and safe transfers between stations.

## Team

| Member | Personally created robot model |
|--------|--------------------------------|
| Lachlan | Yaskawa Motoman MH5 |
| Aarav | Yaskawa Motoman MPX3500 |
| Pauras | FANUC M-10iA |

Model assignments are provisional until confirmed with a tutor. Each personally assessed model must be a 6-DoF (or higher) industrial arm that does not already exist in the 41013 model packs or Peter Corke's Robotics Toolbox for Python, and is not a Universal Robots model.

A UR3/UR3e (supplied in `ir_support`) will be used for sorting and as the representative physical demonstration.

**Real robot:** UR3e

## Robot model sources

Each student builds their own parameter-based (DH) model in Python and documents where the parameters came from. Sources to be confirmed as each model is built.

### Yaskawa Motoman MPX3500 (Aarav)

- **Cross-check:** joint origins and limits in the ROS-Industrial URDF macro file (motoman_mpx3500_support/urdf/mpx3500_macro.xacro) inside the ros-industrial/motoman repository.
- **Meshes:** STL visual and collision meshes (base_link and link_1 to link_6) are located in the motoman_mpx3500_support/meshes/mpx3500/ directory of the same repository. They are licensed under Apache-2.0 / BSD-3-Clause. Keep the license attribution if any meshes are copied into your repository.
### Yaskawa Motoman MH5 (Lachlan)

- **Kinematics:** joint origins, axes and limits in `motoman_mh5_support/urdf/mh5_macro.xacro` in [ros-industrial/motoman](https://github.com/ros-industrial/motoman). Link offsets are 330 mm (base to shoulder), 88 mm (shoulder offset), 310 mm (upper arm), 40 mm (elbow offset), 305 mm (forearm) and 80 mm (wrist to flange). Derive the DH table from these and cross-check against the basic specifications in the [Yaskawa MOTOMAN-MH5 instruction manual](https://www.manualslib.com/manual/1335625/Yaskawa-Motoman-Mh5.html).
- **Meshes:** STL visual and collision meshes (`base_link` and `link_1_s` to `link_6_t`) in `motoman_mh5_support/meshes/mh5/` of the same repository, licensed Apache-2.0 / BSD-3-Clause. Keep the licence attribution if any meshes are copied into this repo.

### FANUC M-10iA (Pauras)

- **Kinematics:** joint origins, axes and limits in `fanuc_m10ia_support/urdf/m10ia_macro.xacro` in [ros-industrial/fanuc](https://github.com/ros-industrial/fanuc). Link offsets are 450 mm (base to shoulder), 150 mm (shoulder offset), 600 mm (upper arm), 200 mm (elbow offset), 640 mm (forearm) and 100 mm (wrist to flange). Derive the DH table from these. A published DH table for the M-10iA appears in Jan et al. (2013), "Smartphone Based Control Architecture of Teaching Pendant for Industrial Manipulators" ([table on ResearchGate](https://www.researchgate.net/figure/Fanuc-M-10iA-model-joint-angles-TABLE-I-THE-DH-PARAMETERS-FOR-M-10IA-ROBOT_fig4_262202562)), and can be used as a cross-check.
- **Meshes:** STL visual and collision meshes (`base_link` and `link_1` to `link_6`) in `fanuc_m10ia_support/meshes/m10ia/` of the same repository, licensed Apache-2.0 / BSD-3-Clause. Keep the licence attribution if any meshes are copied into this repo.
- **Variant:** the M-10iA has several variants (e.g. /7L, /12). The repository's generic `m10ia` model appears to match the /12 (see [issue #286](https://github.com/ros-industrial/fanuc/issues/286)), but this is not confirmed, so check the variant against FANUC documentation before relying on it.

The ROS-Industrial packages are community supported, and the URDFs give link offsets rather than a ready-made DH table, so each student derives and documents their own DH parameters from the sources above.

## Setup

From the repo root, create and activate a virtual environment, then install the requirements.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.12 is recommended (3.10 to 3.12 are supported). `ir-support-full` installs the core `ir-support` package plus the extra robot and parts model packs. Package names use hyphens in pip, but imports use underscores:

| Install package | Python import |
|-----------------|---------------|
| `ir-support` | `ir_support` |
| `ir-support-extra-robots` | `ir_support_extra_robots` |
| `ir-support-extra-parts` | `ir_support_extra_parts` |

Check the Subject resources page on Canvas for the official 41013 setup instructions.

Do not commit `.venv/`, installed toolboxes, caches or third-party packages.

## Running the workcell

With your environment active, from the repo root:

```bash
python src/main.py
```

A browser tab opens showing the Swift scene. Press Ctrl+C in the terminal to close it.

This is an initial sketch made of plain boxes. Nothing moves yet and there are no robots. It has:

- **A conveyor:** along +x.
- **Four robot pads:** one box per station, alternating sides of the belt. The loading and sorting pads are at the two ends of the belt.

  | Station | Planned robot |
  |---------|---------------|
  | Loading | Yaskawa Motoman MH5 (Lachlan) |
  | Spray painting | Yaskawa Motoman MPX3500 (Aarav) |
  | Logo / label | FANUC M-10iA (Pauras) |
  | Sorting | UR3e (group) |



### Known issues with the Swift environment

- `ir-support` pins `swift-sim<2`, and Swift 1.1.0 crashes in `env.step()` with the newer `spatialgeometry` that pip installs. `src/workcell/compat.py` adds a small patch, applied automatically when `workcell` is imported. Import `workcell` before stepping any scene that you build yourself.
- Do not call `env.remove(...)`: after a removal Swift 1.1.0's `env.step()` fails. Move or hide objects instead.
