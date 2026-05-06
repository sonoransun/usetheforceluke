"""Scaled-geometry Casimir variant. SPECULATIVE.

Wraps the textbook parallel-plate formula with a dimensionless geometry factor
``g`` representing the (hypothetical) enhancement from a non-trivial cavity
geometry tuned for propulsion. ``g = 1`` recovers the textbook case.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce.casimir.parallel_plate import ParallelPlateCasimir


class ScaledCasimir:
    """Parallel-plate Casimir with a geometry-enhancement factor ``g``."""

    metadata: dict[str, Any]

    def __init__(
        self,
        area: Any,
        separation: Any,
        geometry_factor: float = 1.0,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        if geometry_factor <= 0:
            raise ValueError("geometry_factor must be positive")
        self._base = ParallelPlateCasimir(area=area, separation=separation, axis=axis)
        self._g = float(geometry_factor)
        self.metadata = {
            "avenue": "casimir",
            "model": f"scaled parallel-plate (g={self._g})",
            "speculative": True,
            "speculative_components": ["geometry_factor"],
            "citation": "speculative extension; geometry-factor ansatz only",
        }

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        return self._g * self._base.force(t, r)

    def potential(self, r: np.ndarray) -> float | None:
        base = self._base.potential(r)
        return None if base is None else self._g * base
