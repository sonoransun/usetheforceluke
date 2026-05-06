"""Antimatter-annihilation graviton field. SPECULATIVE.

A localized antimatter annihilation source at ``r₀`` is modelled as emitting a
Yukawa-type graviton field::

    φ(r) = -g · Γ · exp(-R/λ) / R,    R = |r − r₀|

with annihilation rate ``Γ`` (events/s), graviton coupling ``g``, and screening
length ``λ`` (m). The force on a probe of mass ``m_probe`` is::

    F(r) = -m_probe · ∇φ(r)

Closed-form gradient: with ``d̂ = (r − r₀)/R``,

    dφ/dR = g · Γ · exp(-R/λ) · (R + λ) / (λ R²)
    F(r) = -m_probe · (dφ/dR) · d̂

In the λ → ∞ limit the Yukawa factor (R+λ)·exp(-R/λ)/λ → 1 and the field
reduces to a Newtonian-like 1/r² law.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._yukawa import displacement, yukawa_force_on_probe, yukawa_potential


class AntimatterGravitonField:
    """Yukawa graviton potential sourced by an annihilation hotspot."""

    metadata: dict[str, Any]

    def __init__(
        self,
        source: np.ndarray | tuple[float, float, float],
        gamma: float,
        coupling: float,
        screening: float,
        probe_mass: float,
    ) -> None:
        self._r0 = np.asarray(source, dtype=float)
        if self._r0.shape != (3,):
            raise ValueError("source must have shape (3,)")
        if gamma <= 0:
            raise ValueError("gamma (annihilation rate) must be positive")
        if coupling <= 0:
            raise ValueError("coupling must be positive")
        if screening <= 0:
            raise ValueError("screening (lambda) must be positive")
        if probe_mass <= 0:
            raise ValueError("probe_mass must be positive")
        self._gamma = float(gamma)
        self._g = float(coupling)
        self._lam = float(screening)
        self._mp = float(probe_mass)
        self.metadata = {
            "avenue": "antimatter",
            "model": f"Yukawa graviton (λ={self._lam:g} m)",
            "speculative": True,
            "speculative_components": ["gamma", "coupling", "screening"],
            "citation": "Yukawa-form ansatz for antimatter→graviton emission; not derived from EFT",
        }

    def potential(self, r: np.ndarray) -> float:
        _, R = displacement(r, self._r0)
        return self._mp * yukawa_potential(R, self._g, self._gamma, self._lam)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._r0)
        return yukawa_force_on_probe(d, R, self._g, self._gamma, self._lam, self._mp)
