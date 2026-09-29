# Toy Car Production

Lab Assignment 2 (Pick-and-Place DoGoodBot), 41013 Industrial Robotics.

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
| Pauras | Stäubli TX60 |
| Aarav | FANUC M-10iA |

Model assignments are provisional until confirmed with a tutor. Each personally assessed model must be a 6-DoF (or higher) industrial arm that does not already exist in the 41013 model packs or Peter Corke's Robotics Toolbox for Python, and is not a Universal Robots model.

A UR3/UR3e (supplied in `ir_support`) will be used for sorting and as the representative physical demonstration.

**Real robot:** to be decided (UR3/UR3e proposed).

## Robot model sources

Each student builds their own parameter-based (DH) model in Python and documents where the parameters came from. Sources to be confirmed as each model is built.

### Stäubli TX60 (Pauras)

- **Kinematics (DH):** nominal modified-DH parameters are published in Table 2 of [Kinematics Parameter Calibration of Serial Industrial Robots Based on Partial Pose Measurement](https://www.mdpi.com/2227-7390/11/23/4802) (MDPI Mathematics, open access). Link lengths are 290 mm, 20 mm (elbow offset), 310 mm and 70 mm.
- **Cross-check:** joint origins and limits in the ROS-Industrial URDF (`staubli_tx60_support/urdf`) in [ros-industrial/staubli_experimental](https://github.com/ros-industrial/staubli_experimental). The URDF also gives a 375 mm base-to-shoulder height that the paper's table does not include.
- **Meshes:** STL visual and collision meshes (`base_link` and `link_1` to `link_6`) in `staubli_tx60_support/meshes/tx60/` of the same repository, licensed Apache-2.0. Keep the licence attribution if any meshes are copied into this repo.

### Yaskawa Motoman MH5 (Lachlan)

- **Kinematics:** joint origins, axes and limits in `motoman_mh5_support/urdf/mh5_macro.xacro` in [ros-industrial/motoman](https://github.com/ros-industrial/motoman). Link offsets are 330 mm (base to shoulder), 88 mm (shoulder offset), 310 mm (upper arm), 40 mm (elbow offset), 305 mm (forearm) and 80 mm (wrist to flange). Derive the DH table from these and cross-check against the basic specifications in the [Yaskawa MOTOMAN-MH5 instruction manual](https://www.manualslib.com/manual/1335625/Yaskawa-Motoman-Mh5.html).
- **Meshes:** STL visual and collision meshes (`base_link` and `link_1_s` to `link_6_t`) in `motoman_mh5_support/meshes/mh5/` of the same repository, licensed Apache-2.0 / BSD-3-Clause. Keep the licence attribution if any meshes are copied into this repo.

### FANUC M-10iA (Aarav)

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

## Repository layout

```
src/
  workcell/      shared simulation, conveyor, scene and safety equipment
  robots/        one module per student-created robot model
  gui/           teach/jog interface and e-stop controls
  safety/        e-stop state machine, light curtain, collision checks
docs/            risk assessment, SWMS and notes
tests/           unit tests
```

## Code standard

- PEP 8 style, `snake_case` for functions and variables, `PascalCase` for classes.
- Docstrings on all public classes and functions.
- Each student's work lives in their own robot module so contributions are identifiable.
- Commit regularly with clear messages, from your own account.
