"""Antimatter → counter-gravity conversion model. SPECULATIVE.

Parametric model: locally cancels a supplied background gravitational
acceleration with conversion-efficiency ε ∈ [0, 1]. ε=0 → no effect, ε=1 →
perfect cancellation. The "force" is the reaction on a body of mass ``m``.
This is *not* a derivation from any physical theory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class AntimatterCounterGravity:
    """Counter-gravity force = -ε · m · g(r), where g is a supplied background field."""

    metadata: dict[str, Any]

    def __init__(
        self,
        mass: float,
        efficiency: float,
        background_g: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("efficiency must lie in [0, 1]")
        if mass <= 0:
            raise ValueError("mass must be positive")
        self._m = float(mass)
        self._eps = float(efficiency)
        self._g = background_g
        self.metadata = {
            "avenue": "antimatter",
            "model": f"counter-gravity ansatz (ε={self._eps})",
            "speculative": True,
            "citation": "speculative parametric model; no underlying derivation",
        }

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        g = np.asarray(self._g(np.asarray(r, dtype=float)), dtype=float)
        return -self._eps * self._m * g

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        # General background g may be non-conservative or path-dependent;
        # we don't claim a potential without knowing the field.
        return None
