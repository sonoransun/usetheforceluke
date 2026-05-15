"""Mixed-mass binary with positive total mass — anti-chirp gravitational radiation.

Loeb (2024) notes that a binary with positive total mass but one *negative*
component loses orbital energy through GW emission, but — unlike a normal
chirping binary whose orbit shrinks and whose GW frequency rises — the
anti-chirp binary's orbit *expands* and its GW frequency *falls* over time.
The signed Peters–Mathews chirp mass captures the sign reversal::

    M_c    = sign(m1·m2) · |m1·m2|^(3/5) / (m1+m2)^(1/5)
    df/dt  = (96/5) · π^(8/3) · (G M_c / c³)^(5/3) · f^(11/3)

For anti-chirp (one negative, total positive), ``m1·m2 < 0`` so ``M_c < 0`` and
``df/dt < 0``.

This class exposes the static Newtonian-equivalent gravitational field of the
binary's *total mass* with a Yukawa spatial cutoff (reusing the same
``_yukawa`` helpers as the antimatter graviton model), plus a ``df_dt`` method
that returns the (negative) chirp-rate at a given instantaneous GW frequency
``f``. The trajectory effect is dominated by the static field; the anti-chirp
signature is exposed in the metadata + method so analysts can extract it
without running the full ODE.

``applicable_for_trajectory`` is ``True`` — the spatial profile is a proper
probe field.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._negmass import G_NEWTON, chirp_mass, validate_neg_mass, validate_separation
from usetheforce._yukawa import displacement, yukawa_force_on_probe, yukawa_potential
from usetheforce.symbolic.negative_mass import anti_chirp_dfdt_lambdified


class AntiChirpBinary:
    """Yukawa-screened gravitational field of an anti-chirp binary.

    The binary has positive total mass ``m1 + m2 > 0`` with one component
    negative. Force on a probe of mass ``m_probe`` at displacement ``r`` from
    the binary's centre of mass::

        F(r) = -m_probe · G · M_total · exp(-R/λ) · (R + λ) / (λ R²) · r̂

    In the ``λ → ∞`` limit this recovers the Newtonian inverse-square law.

    The orbital decay rate is exposed via ``df_dt(f)``.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        m_positive_kg: float,
        m_negative_kg: float,
        separation_m: float,
        probe_mass_kg: float,
        source: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        screening_m: float = 1.0e12,
    ) -> None:
        if m_positive_kg <= 0:
            raise ValueError("m_positive_kg must be positive")
        validate_neg_mass(m_negative_kg)
        if m_positive_kg <= m_negative_kg:
            raise ValueError(
                "anti-chirp requires positive total mass: m_positive_kg > m_negative_kg"
            )
        validate_separation(separation_m)
        if probe_mass_kg <= 0:
            raise ValueError("probe_mass_kg must be positive")
        if screening_m <= 0:
            raise ValueError("screening_m must be positive")
        source_arr = np.asarray(source, dtype=float)
        if source_arr.shape != (3,):
            raise ValueError("source must have shape (3,)")

        self._m_pos = float(m_positive_kg)
        self._m_neg = float(m_negative_kg)
        self._d = float(separation_m)
        self._mp = float(probe_mass_kg)
        self._lam = float(screening_m)
        self._source = source_arr
        # Signed (negative) chirp mass — sign carries the anti-chirp diagnostic.
        self._M_c = chirp_mass(self._m_pos, -self._m_neg)
        self._M_total = self._m_pos - self._m_neg  # > 0 by construction
        self._dfdt_lambd = anti_chirp_dfdt_lambdified()
        self.metadata = {
            "avenue": "negmass",
            "model": (
                f"anti-chirp binary (m+={self._m_pos:g}, |m-|={self._m_neg:g} kg, "
                f"M_total={self._M_total:g} kg, signed M_c={self._M_c:g} kg)"
            ),
            "speculative": True,
            "speculative_components": ["m_negative_kg", "negative_mass_premise"],
            "applicable_for_trajectory": True,
            "citation": "Loeb 2024 Medium; Peters–Mathews 1963 with sign reversal",
            "signed_chirp_mass_kg": self._M_c,
            "total_mass_kg": self._M_total,
        }

    @property
    def signed_chirp_mass_kg(self) -> float:
        return self._M_c

    def df_dt(self, f_hz: float) -> float:
        """Peters–Mathews chirp rate at GW frequency ``f``; negative for anti-chirp."""
        if f_hz <= 0:
            raise ValueError("f_hz must be positive")
        from scipy.constants import c as C_LIGHT

        return float(self._dfdt_lambd(self._M_c, f_hz, G_NEWTON, C_LIGHT))

    def potential(self, r: np.ndarray) -> float:
        _, R = displacement(r, self._source)
        # Yukawa-screened gravitational potential of total mass; reuse the
        # _yukawa helper with coupling = G_NEWTON and "gamma" = M_total.
        return self._mp * yukawa_potential(R, G_NEWTON, self._M_total, self._lam)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._source)
        return yukawa_force_on_probe(d, R, G_NEWTON, self._M_total, self._lam, self._mp)
