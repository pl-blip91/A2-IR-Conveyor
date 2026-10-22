import spatialgeometry as sg


def apply_patches() -> None:
    noop = lambda self: None  # noqa: E731
    if hasattr(sg, "Shape") and not hasattr(sg.Shape, "_update_pyb"):
        sg.Shape._update_pyb = noop
    for name in ("Cuboid", "Cylinder", "Sphere", "Arrow", "Mesh"):
        cls = getattr(sg, name, None)
        if cls is not None and not hasattr(cls, "_update_pyb"):
            cls._update_pyb = noop


apply_patches()
