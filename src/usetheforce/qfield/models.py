"""Parametric ansatz for a shaped quantum field. SPECULATIVE.

Reference figure rather than physical model: a Gaussian-localized scalar
potential ``U(r) = -A exp(-|r-r0|^2 / (2 σ^2))`` whose gradient supplies a
conservative force. The ansatz exists so the framework can be exercised
end-to-end (trajectories, conservation tests, field visualization) without
claiming a physical mechanism.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ShapedFieldAnsatz:
    """Conservative Gaussian-well force field with parameters (A, σ, r₀)."""

    metadata: dict[str, Any]

    def __init__(
        self,
        amplitude: float,
        sigma: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._A = float(amplitude)
        self._sigma = float(sigma)
        self._r0 = np.asarray(center, dtype=float)
        self.metadata = {
            "avenue": "qfield",
            "model": "Gaussian-well ansatz",
            "speculative": True,
            "speculative_components": ["amplitude", "sigma", "center"],
            "citation": "parametric ansatz only; not a physical model",
        }

    def potential(self, r: np.ndarray) -> float:
        d = np.asarray(r, dtype=float) - self._r0
        return -self._A * float(np.exp(-np.dot(d, d) / (2.0 * self._sigma**2)))

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d = np.asarray(r, dtype=float) - self._r0
        # F = -∇U = -A * exp(-|d|²/(2σ²)) * (d/σ²)
        return -self._A * np.exp(-np.dot(d, d) / (2.0 * self._sigma**2)) * (d / self._sigma**2)
