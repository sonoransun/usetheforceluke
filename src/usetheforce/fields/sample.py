"""Sample a ``ForceField`` onto a ``RegularGrid3D`` — the bridge to PyVista glyphs."""

from __future__ import annotations

import numpy as np

from usetheforce.fields.grid import RegularGrid3D
from usetheforce.protocol import ForceField


def sample_force_field(ff: ForceField, grid: RegularGrid3D, t: float = 0.0) -> np.ndarray:
    """Return a (3, Nx, Ny, Nz) ndarray of force vectors on ``grid`` at time ``t``."""
    nx, ny, nz = grid.shape
    out = np.empty((3, nx, ny, nz), dtype=float)
    points = grid.points().reshape(nx, ny, nz, 3)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:, i, j, k] = ff.force(t, points[i, j, k])
    return out
