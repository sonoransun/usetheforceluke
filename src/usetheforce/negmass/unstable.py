"""Negative-total-mass binary — repulsive and short-lived (Loeb 2024).

A binary with negative total mass ``m1 + m2 < 0`` is gravitationally repulsive
in its centre-of-mass dynamics and admits no bound orbit. Loeb describes it as
"short-lived"; we make that quantitative by exposing a break-time

    t_break = π · sqrt(d³ / (G · |m1 + m2|))

(the natural dynamical timescale for an unbound binary of this scale). The
model returns a constant repulsive body force on the composite craft for
``0 ≤ t < t_break`` and raises ``ValueError`` at ``t ≥ t_break`` — same
singularity-guard idiom as ``HeavyElementLattice``'s ``softening`` check.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from usetheforce._negmass import G_NEWTON, validate_neg_mass, validate_separation


class NegativeTotalMassBinary:
    """Repulsive Bondi-style pair with negative total mass; disintegrates at ``t_break``."""

    metadata: dict[str, Any]

    def __init__(
        self,
        m_positive_kg: float,
        m_negative_kg: float,
        separation_m: float,
        craft_mass_kg: float,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        if m_positive_kg <= 0:
            raise ValueError("m_positive_kg must be positive")
        validate_neg_mass(m_negative_kg)
        if m_negative_kg <= m_positive_kg:
            raise ValueError(
                "negative-total-mass requires |m_negative_kg| > m_positive_kg "
                f"(got {m_negative_kg} vs {m_positive_kg})"
            )
        validate_separation(separation_m)
        if craft_mass_kg <= 0:
            raise ValueError("craft_mass_kg must be positive")
        axis_arr = np.asarray(axis, dtype=float)
        if axis_arr.shape != (3,):
            raise ValueError("axis must have shape (3,)")
        norm = float(np.linalg.norm(axis_arr))
        if norm == 0:
            raise ValueError("axis must be non-zero")
        self._axis = axis_arr / norm

        self._m_pos = float(m_positive_kg)
        self._m_neg = float(m_negative_kg)
        self._d = float(separation_m)
        self._mc = float(craft_mass_kg)
        self._M_net_abs = self._m_neg - self._m_pos  # > 0 by construction
        self._a = G_NEWTON * self._M_net_abs / (self._d * self._d)
        # Repulsive (anti-Bondi) thrust on craft: force points along -axis.
        self._force_vec: np.ndarray = -self._mc * self._a * self._axis
        # Natural dynamical timescale; raise once the binary has flown apart.
        self._t_break: float = math.pi * math.sqrt(
            self._d**3 / (G_NEWTON * self._M_net_abs)
        )
        self.metadata = {
            "avenue": "negmass",
            "model": (
                f"negative-total-mass binary (m+={self._m_pos:g}, |m-|={self._m_neg:g} kg, "
                f"t_break={self._t_break:g} s)"
            ),
            "speculative": True,
            "speculative_components": ["negative_mass_premise"],
            "applicable_for_trajectory": False,
            "citation": "Bondi 1957 unstable regime; Loeb 2024 'short-lived'",
            "t_break_s": self._t_break,
        }

    @property
    def t_break_s(self) -> float:
        return self._t_break

    def _check_alive(self, t: float) -> None:
        if t >= self._t_break:
            raise ValueError(
                f"NegativeTotalMassBinary has disintegrated: t={t} s ≥ t_break={self._t_break} s"
            )

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self._check_alive(float(t))
        return self._force_vec

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        # As with BondiRunawayPair: a position-independent body force has no
        # meaningful probe-position potential. Return None.
        return None
