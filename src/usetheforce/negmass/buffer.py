"""Negative-mass point source — repulsive Newtonian gravity. SPECULATIVE.

A single negative-mass element at ``source`` exerts a *repulsive* gravitational
force on a positive-mass probe::

    F(r) = +m_probe · G · |m_neg| / R² · r̂,   R = |r − source|

(Force points *away* from the source — the sign reversal vs. ordinary Newtonian
attraction is the entire content of the model.) This primitive is the building
block for "negative-mass buffer" scenarios: place one between a craft and a
gravitational sink (e.g. a black hole) and the buffer pushes the craft outward,
partially counteracting the sink's pull. The mission factory
``event_horizon_stationkeep_with_buffer`` uses exactly this construction via
``CompositeField``.

A pure point-source field is singular at the source; ``R_FLOOR_M`` from
``_yukawa.py`` guards against probe coincidence. For finite-extent or screened
behaviour, wrap a different speculative model (e.g. a Yukawa form with finite
``λ``) instead.

Speculative components: ``["m_negative_kg", "negative_mass_premise"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._negmass import G_NEWTON, validate_neg_mass
from usetheforce._yukawa import displacement


class NegativeMassPointSource:
    """Repulsive Newtonian gravity from a single negative-mass point.

    Parameters
    ----------
    m_negative_kg:
        Magnitude of the negative inertial mass (positive float — sign is part
        of the model premise, not the argument).
    probe_mass_kg:
        Positive-mass probe; force scales as ``m_probe``.
    source:
        Point-source position (m), shape ``(3,)``. Defaults to origin.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        m_negative_kg: float,
        probe_mass_kg: float,
        source: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        validate_neg_mass(m_negative_kg)
        if probe_mass_kg <= 0:
            raise ValueError("probe_mass_kg must be positive")
        source_arr = np.asarray(source, dtype=float)
        if source_arr.shape != (3,):
            raise ValueError("source must have shape (3,)")
        self._m_neg = float(m_negative_kg)
        self._mp = float(probe_mass_kg)
        self._source = source_arr
        self.metadata = {
            "avenue": "negmass",
            "model": (
                f"negative-mass point source (|m_neg|={self._m_neg:g} kg, "
                f"source={tuple(self._source)})"
            ),
            "speculative": True,
            "speculative_components": ["m_negative_kg", "negative_mass_premise"],
            "applicable_for_trajectory": True,
            "citation": "Newtonian gravity with sign-flipped source (Bondi 1957 premise)",
        }

    @property
    def m_negative_kg(self) -> float:
        return self._m_neg

    @property
    def source(self) -> np.ndarray:
        return self._source.copy()

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._source)
        magnitude = G_NEWTON * self._m_neg * self._mp / (R * R)
        # Repulsive: force points *along* d (away from source).
        return magnitude * (d / R)

    def potential(self, r: np.ndarray) -> float:
        """Repulsive Newtonian potential ``+G · |m_neg| · m_probe / R``.

        Sign-flipped vs. ordinary attractive gravity. Diverges at the source.
        """
        _, R = displacement(r, self._source)
        return G_NEWTON * self._m_neg * self._mp / R
