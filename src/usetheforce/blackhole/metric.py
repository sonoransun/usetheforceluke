"""Schwarzschild gravitational field outside the event horizon.

Newtonian-limit force on a probe of mass ``m_probe`` at distance ``R`` from a
point mass ``M`` centred at ``r_c``::

    F(r) = -G M m_probe / R² · r̂,    R = |r - r_c|

with optional general-relativistic correction for a *stationary* (hovering)
observer::

    F_hover(r) = F_Newtonian(r) · 1 / sqrt(1 - r_s / R),    r_s = 2 G M / c²

The GR factor diverges as ``R → r_s⁺`` — no real propulsion system can hover at
the horizon. ``potential()`` returns the Newtonian potential only; the GR
hover correction is non-conservative (it is the force the *engine* must apply,
not a -∇U term), so ``total_energy`` is only meaningful in the Newtonian-only
mode.

References
----------
- Schwarzschild (1916); Misner, Thorne, Wheeler *Gravitation* §31.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._schwarzschild import (
    G_NEWTON,
    displacement,
    gr_hover_factor,
    schwarzschild_radius,
)


class SchwarzschildGravity:
    """Newtonian / optional-GR Schwarzschild gravitational field on a probe.

    Parameters
    ----------
    mass_kg:
        Central body mass (kg). Sets ``r_s = 2 G M / c²``.
    probe_mass_kg:
        Probe mass (kg). Force is mass-weighted: ``F = -G M m_probe / R² · r̂``.
    center:
        Centre position of the gravitating mass (m), shape ``(3,)``.
    use_gr_hover_correction:
        When True, multiply the Newtonian force by ``1 / sqrt(1 - r_s/R)`` —
        the proper acceleration a stationary observer must apply to hover.
        Diverges at the horizon; only meaningful when ``R > r_s``.
    horizon_softening_m:
        Minimum gap above ``r_s`` at which ``force(t, r)`` and ``potential(r)``
        are willing to evaluate. A probe at ``R ≤ r_s + horizon_softening`` raises
        ``ValueError``. Default 0.0 (raise only inside or on the horizon itself).
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        mass_kg: float,
        probe_mass_kg: float,
        center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_gr_hover_correction: bool = False,
        horizon_softening_m: float = 0.0,
    ) -> None:
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if probe_mass_kg <= 0:
            raise ValueError("probe_mass_kg must be positive")
        if horizon_softening_m < 0:
            raise ValueError("horizon_softening_m must be non-negative")
        self._M = float(mass_kg)
        self._mp = float(probe_mass_kg)
        self._center = np.asarray(center, dtype=float)
        if self._center.shape != (3,):
            raise ValueError("center must have shape (3,)")
        self._use_gr = bool(use_gr_hover_correction)
        self._softening = float(horizon_softening_m)
        self._rs = schwarzschild_radius(self._M)
        self.metadata = {
            "avenue": "blackhole",
            "model": (
                f"Schwarzschild gravity (M={self._M:.3g} kg, r_s={self._rs:.3g} m"
                f"{', GR hover' if self._use_gr else ''})"
            ),
            "speculative": False,
            "speculative_components": [],
            "applicable_for_trajectory": True,
            "citation": "Schwarzschild (1916); MTW Gravitation §31",
            "schwarzschild_radius_m": self._rs,
            "use_gr_hover_correction": self._use_gr,
            "horizon_softening_m": self._softening,
        }

    @property
    def schwarzschild_radius_m(self) -> float:
        return self._rs

    @property
    def mass_kg(self) -> float:
        return self._M

    @property
    def probe_mass_kg(self) -> float:
        return self._mp

    def _check_outside_horizon(self, R: float) -> None:
        if self._rs + self._softening >= R:
            raise ValueError(
                f"probe at R={R} m is inside r_s + softening "
                f"(r_s={self._rs}, softening={self._softening}); Schwarzschild field is singular"
            )

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._center)
        self._check_outside_horizon(R)
        # Newtonian-limit magnitude (attractive toward centre).
        magnitude = G_NEWTON * self._M * self._mp / (R * R)
        if self._use_gr:
            magnitude *= gr_hover_factor(R, self._rs)
        return -magnitude * (d / R)

    def potential(self, r: np.ndarray) -> float:
        """Newtonian gravitational potential ``U = -G M m_probe / R``.

        Note: when ``use_gr_hover_correction=True`` the *force* is non-conservative
        (it is the engine thrust needed to hover, not -∇U). The Newtonian
        potential is still well-defined as a scalar; callers using
        ``total_energy`` should construct the field with ``use_gr_hover_correction=False``
        for meaningful conservation checks.
        """
        _, R = displacement(r, self._center)
        self._check_outside_horizon(R)
        return -G_NEWTON * self._M * self._mp / R
