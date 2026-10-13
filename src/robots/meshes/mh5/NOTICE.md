# Yaskawa Motoman MH5 meshes

The `*.stl` files in this folder are the visual meshes of the
`motoman_mh5_support` package from
[ros-industrial/motoman](https://github.com/ros-industrial/motoman)
(`motoman_mh5_support/meshes/mh5/visual/`, branch `kinetic-devel`),
copied unmodified. The package declares its licence as BSD
(`motoman_mh5_support/package.xml`).

Mapping used by `src/robots/yaskawa_mh5.py` (each mesh is expressed in its
own URDF link frame, in metres):

| Robot link | Mesh file    |
|------------|--------------|
| 0 (base)   | base_link.stl |
| 1          | link_1_s.stl  |
| 2          | link_2_l.stl  |
| 3          | link_3_u.stl  |
| 4          | link_4_r.stl  |
| 5          | link_5_b.stl  |
| 6          | link_6_t.stl  |

Keep this notice with the meshes.
