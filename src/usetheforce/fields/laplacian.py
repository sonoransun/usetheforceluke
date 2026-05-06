"""Second-order central-difference 3D Laplacian on a ``RegularGrid3D``."""

from __future__ import annotations

import numpy as np

from usetheforce.fields.grid import RegularGrid3D


def laplacian_3d(field: np.ndarray, grid: RegularGrid3D) -> np.ndarray:
    """∇²φ on an interior stencil; boundary cells return 0 (Dirichlet-style)."""
    if field.shape != grid.shape:
        raise ValueError(f"field shape {field.shape} != grid shape {grid.shape}")
    dx, dy, dz = grid.spacing
    out = np.zeros_like(field, dtype=float)
    out[1:-1, 1:-1, 1:-1] = (
        (field[2:, 1:-1, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / dx**2
        + (field[1:-1, 2:, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / dy**2
        + (field[1:-1, 1:-1, 2:] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / dz**2
    )
    return out
