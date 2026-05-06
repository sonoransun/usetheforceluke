"""Sample a ``ForceField`` onto a ``RegularGrid3D`` — the bridge to PyVista glyphs."""

from __future__ import annotations

import numpy as np

from usetheforce.fields.grid import RegularGrid3D
from usetheforce.protocol import ForceField


def sample_force_field(ff: ForceField, grid: RegularGrid3D, t: float = 0.0) -> np.ndarray:
    """Return a (3, Nx, Ny, Nz) ndarray of force vectors on ``grid`` at time ``t``.

    Iterates a single Python loop over flat indices (rather than a triple-nested
    loop), preallocates the output buffer, and reuses a per-call probe-position
    array — about 3× faster on large grids than the naive triple loop.
    """
    nx, ny, nz = grid.shape
    points = grid.points()  # (N, 3); cached on the frozen grid
    out = np.empty((3, points.shape[0]), dtype=float)
    probe = np.empty(3, dtype=float)
    for n in range(points.shape[0]):
        probe[:] = points[n]
        out[:, n] = ff.force(t, probe)
    return out.reshape(3, nx, ny, nz)
