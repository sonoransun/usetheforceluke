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

    The model represents a rigid-cavity *body force* — a constant thrust
    between two fixed-geometry plates. ``force(t, r)`` returns the same vector
    everywhere in space (the plate separation is set at construction; probe
    position has no effect). ``potential(r)`` returns the cavity's
    Casimir energy (also constant in ``r``).

    **This is not a probe-position field.** ``F = -∇U`` holds trivially because
    both sides are zero in space — but ``total_energy`` over a Casimir trajectory
    is meaningless because the field exerts a constant body force regardless of
    position. The mission adapter flags ``applicable=False`` for free-flight
    propulsion accordingly; treat this model as the textbook anchor for
    benchmarking, not as a propulsion-suitable potential.

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
        # Cached static quantities — both depend only on (separation, area), set
        # at construction. Caching saves the per-call pow + multiply in the
        # hot ODE / snapshot loop.
        self._pressure: float = -(np.pi**2) * hbar * c / (240.0 * self._sep**4)
        self._potential: float = -(np.pi**2) * hbar * c * self._area / (720.0 * self._sep**3)
        self._force_vec: np.ndarray = self._pressure * self._area * self._axis
        self.metadata = {
            "avenue": "casimir",
            "model": "parallel-plate (Casimir 1948)",
            "speculative": False,
            "speculative_components": [],
            "applicable_for_trajectory": False,
            "citation": "Casimir 1948, Proc. K. Ned. Akad. Wet. 51:793",
        }

    @property
    def pressure(self) -> float:
        """Casimir pressure in Pa (negative = attractive). Cached."""
        return self._pressure

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Force in Newtons on the ``+axis`` plate. Constant in space and time."""
        return self._force_vec

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        """Cavity energy U = -π² ℏ c A / (720 a³). Constant in space (rigid plates)."""
        return self._potential
