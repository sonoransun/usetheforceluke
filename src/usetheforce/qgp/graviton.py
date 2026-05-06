"""QGP-sourced graviton field. Anchored ε(T) × speculative graviton coupling.

Re-implements the same Yukawa form as ``AntimatterGravitonField`` but draws
the emission rate Γ from a ``QuarkGluonPlasmaSource``. This keeps the avenue
peer-not-hierarchy: ``QGPGravitonField`` and ``AntimatterGravitonField`` are
independent classes that happen to share a potential form.

    φ(r) = -g · Γ · exp(-R/λ) / R,  R = |r − r₀|
    F(r) = -m_probe · ∇φ(r)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._yukawa import displacement, yukawa_force_on_probe, yukawa_potential
from usetheforce.qgp.source import QuarkGluonPlasmaSource


class QGPGravitonField:
    """Yukawa graviton field whose Γ is set by a ``QuarkGluonPlasmaSource``.

    The Yukawa potential ``φ(r) = -g·Γ·exp(-R/λ)/R`` and the associated
    ``F = -m·∇φ`` are *re-implemented* here (delegating to ``_yukawa``) rather
    than inherited from ``AntimatterGravitonField``: the two models share the
    same potential form but represent independent speculative mechanisms, so
    the package keeps them as peer classes.

    Construction:

    - ``source``: the QGP source providing ``graviton_emission_rate()``;
      Γ is read once at construction and cached for inner-loop speed.
    - ``screening_length`` (λ, m): Yukawa screening length. ``λ → ∞`` recovers
      the inverse-square limit.
    - ``coupling_g``: dimensionless graviton coupling.
    - ``probe_mass`` (kg): mass of the probe whose force is computed; lets the
      class return Newtons directly.

    The anchored Stefan–Boltzmann ε(T) lives in the ``source``; the speculative
    coupling lives here. ``metadata["speculative_components"]`` enumerates the
    speculative parameters.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        source: QuarkGluonPlasmaSource,
        source_position: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        screening_length: float = 1.0e6,
        coupling_g: float = 1.0,
        probe_mass: float = 1.0,
    ) -> None:
        self._r0 = np.asarray(source_position, dtype=float)
        if self._r0.shape != (3,):
            raise ValueError("source_position must have shape (3,)")
        if screening_length <= 0:
            raise ValueError("screening_length (lambda) must be positive")
        if coupling_g <= 0:
            raise ValueError("coupling_g must be positive")
        if probe_mass <= 0:
            raise ValueError("probe_mass must be positive")
        self._lam = float(screening_length)
        self._g = float(coupling_g)
        self._mp = float(probe_mass)
        # Cache Γ — source parameters are immutable post-construction; evaluating
        # Γ once keeps the ODE inner loop free of ε(T) recomputation.
        self._gamma = float(source.graviton_emission_rate())
        self._source = source
        self.metadata = {
            "avenue": "qgp",
            "model": (
                f"QGP graviton emission "
                f"(V={source.volume_m3:.3g} m³, T={source.temperature_K:.3g} K)"
            ),
            "speculative": True,
            "speculative_components": [
                "containment_efficiency",
                "graviton_yield",
                "graviton_energy_quantum_J",
                "screening",
                "coupling_g",
            ],
            "citation": (
                "QGP energy density: Bjorken 1983 / lattice QCD g_eff(T); "
                "graviton coupling: speculative ansatz"
            ),
            "source_metadata": dict(source.metadata),
            "annihilation_rate_gamma": self._gamma,
            "screening_lambda_m": self._lam,
            "coupling_g": self._g,
            "probe_mass_kg": self._mp,
        }

    @property
    def gamma(self) -> float:
        """Cached graviton emission rate (events/s) from the source."""
        return self._gamma

    @property
    def source(self) -> QuarkGluonPlasmaSource:
        return self._source

    def potential(self, r: np.ndarray) -> float:
        _, R = displacement(r, self._r0)
        return self._mp * yukawa_potential(R, self._g, self._gamma, self._lam)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, R = displacement(r, self._r0)
        return yukawa_force_on_probe(d, R, self._g, self._gamma, self._lam, self._mp)
