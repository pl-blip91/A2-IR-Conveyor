"""Compatibility patches for the 41013 Python environment.

Why this exists
---------------
``ir-support`` pins ``swift-sim<2``. Swift 1.1.0 calls ``Shape._update_pyb()``
on every shape that has ``collision=True`` each time ``env.step()`` runs, but
the ``spatialgeometry`` release that pip installs alongside it (1.4.x) no
longer has that method. Without the patch below, ``env.step()`` raises::

    AttributeError: 'Mesh' object has no attribute '_update_pyb'

The patch adds a no-op ``_update_pyb`` only when it is missing, so it does
nothing on an environment that does not need it.

Known Swift 1.1.0 limitation
----------------------------
Do not call ``env.remove(...)``: after a removal ``env.step()`` fails because
it iterates over the emptied slot. Hide or park objects instead (see
``toy_car.PARK_POSE``).
"""

import spatialgeometry as sg


def apply_patches() -> None:
    """Add a no-op ``Shape._update_pyb`` if the installed version lacks it."""
    if not hasattr(sg.Shape, "_update_pyb"):
        sg.Shape._update_pyb = lambda self: None


apply_patches()
