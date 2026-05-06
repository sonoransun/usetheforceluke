"""The ``ForceField`` protocol — the single shared seam between propulsion avenues.

Each avenue (Casimir, qfield, antimatter, …) produces an object satisfying this
protocol; ``trajectories.integrate`` consumes it. Adding a new avenue is one
class. Keep this contract minimal — every method here is paid for in every
implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ForceField(Protocol):
    """A propulsion-relevant force field over (t, r) in SI.

    ``force(t, r)`` returns Newtons (shape ``(3,)``). ``potential(r)`` is
    optional — when present the trajectory integrator can compute total energy
    and conservation tests apply. ``metadata`` declares the avenue name, a
    ``speculative`` flag, and a citation/source.
    """

    metadata: dict[str, Any]

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        """Force in Newtons at time ``t`` (s) and position ``r`` (m)."""
        ...

    def potential(self, r: np.ndarray) -> float | None:
        """Potential energy in Joules at ``r``, or ``None`` if not conservative/known."""
        ...
