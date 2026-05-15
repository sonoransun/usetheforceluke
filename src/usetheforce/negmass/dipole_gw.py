"""Dipole gravitational-radiation field from a WEP-violating negative-mass binary.

In standard GR the mass dipole moment ``D = Σ mᵢ rᵢ`` is conserved (it is the
total linear momentum divided by total mass), so gravitational radiation
appears at *quadrupole* order. If the equivalence principle fails for the
negative-mass component — Loeb's footnote — that conservation is broken and
*dipole* radiation appears at leading order. The angular pattern of the
radiation pressure on a distant probe is then ``∝ cos θ`` (the dipole pattern)
rather than the quadrupole's ``∝ sin² θ``.

This class models the radial force on a probe at displacement ``r`` from the
binary's centre of mass as

    F(r) = -m_probe · α · G · m_neg² · cos θ / (d² · R²) · r̂

with ``α ∈ [0, 1]`` the dimensionless WEP-violation amplitude (``α = 0``
recovers the null case — no dipole radiation) and ``θ`` the angle between
``r`` and the binary's axis. The angular dependence is the headline test: at
``θ = 0`` the force is maximal, at ``θ = π/2`` it vanishes exactly.

Speculative on speculative: this model requires both negative mass *and* a WEP
violation. ``metadata["speculative_components"]`` lists both, plus the
``wep_violation_amplitude`` knob.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._negmass import G_NEWTON, validate_neg_mass, validate_separation
from usetheforce._yukawa import displacement


class DipoleGravitonRadiator:
    """Dipole-pattern radiative force from a WEP-violating negative-mass binary."""

    metadata: dict[str, Any]

    def __init__(
        self,
        m_negative_kg: float,
        separation_m: float,
        wep_violation: float,
        probe_mass_kg: float,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        source: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        validate_neg_mass(m_negative_kg)
        validate_separation(separation_m)
        if not 0.0 <= wep_violation <= 1.0:
            raise ValueError("wep_violation must lie in [0, 1]")
        if probe_mass_kg <= 0:
            raise ValueError("probe_mass_kg must be positive")
        axis_arr = np.asarray(axis, dtype=float)
        if axis_arr.shape != (3,):
            raise ValueError("axis must have shape (3,)")
        norm = float(np.linalg.norm(axis_arr))
        if norm == 0:
            raise ValueError("axis must be non-zero")
        self._axis = axis_arr / norm
        source_arr = np.asarray(source, dtype=float)
        if source_arr.shape != (3,):
            raise ValueError("source must have shape (3,)")

        self._m_neg = float(m_negative_kg)
        self._d = float(separation_m)
        self._alpha = float(wep_violation)
        self._mp = float(probe_mass_kg)
        self._source = source_arr
        # Pre-compute the scalar prefactor: α · G · m_neg² / d² (units: m⁵ s⁻²).
        self._prefactor = self._alpha * G_NEWTON * self._m_neg**2 / (self._d * self._d)
        self.metadata = {
            "avenue": "negmass",
            "model": (
                f"dipole graviton radiator (|m_neg|={self._m_neg:g} kg, d={self._d:g} m, "
                f"α={self._alpha})"
            ),
            "speculative": True,
            "speculative_components": [
                "m_negative_kg",
                "negative_mass_premise",
                "wep_violation_amplitude",
            ],
            "applicable_for_trajectory": True,
            "citation": "Loeb 2024 Medium (WEP-violation footnote); dipole pattern ∝ cos θ",
            "wep_violation": self._alpha,
        }

    def _cos_theta(self, d_vec: np.ndarray, R: float) -> float:
        """Cosine of angle between probe displacement and binary axis."""
        return float(np.dot(d_vec, self._axis) / R)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._source)
        cos_theta = self._cos_theta(d, R)
        # Radial force on probe; dipole angular dependence cos θ.
        magnitude = self._mp * self._prefactor * cos_theta / (R * R)
        return -magnitude * (d / R)

    def potential(self, r: np.ndarray) -> float | None:
        """Dipole-radiation pressure is non-conservative (it transports
        energy radially outward from the source). Return ``None`` — a
        scalar probe potential would only mislead ``total_energy``."""
        del r
        return None
