"""Internal helpers shared by Yukawa-form ``ForceField`` implementations.

The two graviton models (``AntimatterGravitonField``, ``QGPGravitonField``)
both compute a probe-to-source displacement and a Yukawa potential::

    φ(R) = -g · Γ · exp(-R/λ) / R
    F_radial(R) = -m_probe · dφ/dR
                = -m_probe · g · Γ · exp(-R/λ) · (R + λ) / (λ R²)

Centralising these formulas keeps the two classes truly equivalent up to the
choice of ``Γ`` source.
"""

from __future__ import annotations

import numpy as np

# Smallest |R| we treat as physically meaningful before raising.
# Below this, the 1/R² term overflows; above, the float arithmetic is fine.
R_FLOOR_M: float = 1e-30


def displacement(r: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(d = r − r₀, R = |d|)``; raise if ``R`` falls below ``R_FLOOR_M``."""
    d = np.asarray(r, dtype=float) - source
    R = float(np.sqrt(np.dot(d, d)))
    if R < R_FLOOR_M:
        raise ValueError(
            f"probe within {R_FLOOR_M} m of source (R={R}); Yukawa potential is singular"
        )
    return d, R


def yukawa_potential(R: float, g: float, gamma: float, lam: float) -> float:
    """φ(R) = -g·Γ·exp(-R/λ)/R."""
    return -g * gamma * np.exp(-R / lam) / R


def yukawa_force_on_probe(
    d: np.ndarray, R: float, g: float, gamma: float, lam: float, mass: float
) -> np.ndarray:
    """F = -m·∇φ along (d/R), magnitude ``m·g·Γ·e^{-R/λ}·(R+λ)/(λR²)``, attractive."""
    dphi_dR = g * gamma * np.exp(-R / lam) * (R + lam) / (lam * R**2)
    return -mass * dphi_dR * (d / R)
