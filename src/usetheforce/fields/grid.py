"""Regular 3D grids in SI units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegularGrid3D:
    """Uniform 3D grid: origin (m), spacing (m), shape (Nx, Ny, Nz)."""

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.spacing):
            raise ValueError("spacing components must be positive")
        if any(n < 2 for n in self.shape):
            raise ValueError("shape components must be >= 2")

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        nx, ny, nz = self.shape
        return (
            ox + dx * np.arange(nx),
            oy + dy * np.arange(ny),
            oz + dz * np.arange(nz),
        )

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = self.axes()
        return np.meshgrid(x, y, z, indexing="ij")

    def points(self) -> np.ndarray:
        """Flat (N, 3) array of grid-point coordinates in ij ordering."""
        X, Y, Z = self.meshgrid()
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
