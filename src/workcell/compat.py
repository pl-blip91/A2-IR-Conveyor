"""Compatibility patches for the 41013 Python environment."""

import spatialgeometry as sg


def apply_patches() -> None:
    """Add a no-op ``Shape._update_pyb`` if the installed version lacks it."""
    if not hasattr(sg.Shape, "_update_pyb"):
        sg.Shape._update_pyb = lambda self: None


apply_patches()
