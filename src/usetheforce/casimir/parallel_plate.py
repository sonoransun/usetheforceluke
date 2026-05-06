"""Parallel-plate Casimir force — F/A = -π² ℏ c / (240 a⁴).

Reference: Casimir, H.B.G. (1948), "On the attraction between two perfectly
conducting plates", Proc. K. Ned. Akad. Wet. 51: 793.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import c, hbar

from usetheforce.units import to_si


class ParallelPlateCasimir:
    """Attractive Casimir force between two perfectly-conducting parallel plates.

    The force acts along ``-axis`` on the plate at ``+separation/2`` and along
    ``+axis`` on the plate at ``-separation/2``. By default we report the force
    on the +axis plate; the sign is therefore negative along ``axis``.

    Parameters use pint Quantities at the boundary; internally everything is SI.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        area: Any,
        separation: Any,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self._area = float(to_si(area, "m**2"))
        self._sep = float(to_si(separation, "m"))
        if self._sep <= 0:
            raise ValueError("separation must be positive")
        if self._area <= 0:
            raise ValueError("area must be positive")
        self._axis = np.asarray(axis, dtype=float)
        self._axis /= np.linalg.norm(self._axis)
        self.metadata = {
            "avenue": "casimir",
            "model": "parallel-plate (Casimir 1948)",
            "speculative": False,
            "citation": "Casimir 1948, Proc. K. Ned. Akad. Wet. 51:793",
        }

    @property
    def pressure(self) -> float:
        """Casimir pressure in Pa (negative = attractive)."""
        return -(np.pi**2) * hbar * c / (240.0 * self._sep**4)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002 (steady-state)
        """Force in Newtons on the ``+axis`` plate."""
        magnitude = self.pressure * self._area
        return magnitude * self._axis

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        """Energy U = -π² ℏ c A / (720 a³). Independent of ``r`` (rigid plates)."""
        return -(np.pi**2) * hbar * c * self._area / (720.0 * self._sep**3)
