"""Quark–gluon plasma power-source model.

Energy density follows the Stefan–Boltzmann form with a temperature-dependent
effective degrees-of-freedom count:

    ε(T) = (π² / 30) · g_eff(T) · (k_B T)⁴ / (ℏ c)³

In the QGP limit (T ≫ T_c) with three light flavours::

    g_QGP = 16 + (7/8) · 12 · 3 = 47.5

In the hadron-resonance-gas limit (T ≪ T_c) the relevant degrees of freedom are
mostly massless pions::

    g_HRG ≈ 3

The deconfinement transition at physical quark masses is a smooth crossover
near T_c ≈ 155 MeV (lattice QCD). We interpolate between the two limits via
``tanh((T − T_c) / ΔT)`` with ΔT ≈ 20 MeV — phenomenological but in line with
the shape of lattice ε/T⁴ data.

The graviton coupling on top of this is *speculative*: a single dimensionless
``graviton_yield`` η times the source's power output, divided by a stated
graviton-energy-quantum, gives the emission rate Γ. Every consumer of this
class prints those two parameters so the speculative knob is visible.

References
----------
- Bjorken, J. D. (1983). "Highly relativistic nucleus-nucleus collisions".
  Phys. Rev. D 27 (1): 140–151.
- Borsanyi et al. (2014). "Full result for the QCD equation of state".
  Phys. Lett. B 730: 99–104. (Lattice QCD ε(T) crossover.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import scipy.constants as sc

from usetheforce.units import to_si

# Lattice-QCD-flavoured constants. These are *not* speculative — they are
# textbook QGP thermodynamics for three light flavours.
G_HRG: float = 3.0
G_QGP: float = 47.5
T_C_MEV: float = 155.0  # crossover temperature, lattice-QCD value
DELTA_T_MEV: float = 20.0  # crossover width

# Conversion: 1 MeV = (1e6 e) joules / k_B per Kelvin.
# T_c in K:
_MEV_TO_K: float = 1e6 * sc.e / sc.k


def g_effective(
    T_kelvin: float,
    *,
    T_c_mev: float = T_C_MEV,
    delta_T_mev: float = DELTA_T_MEV,
    g_hrg: float = G_HRG,
    g_qgp: float = G_QGP,
) -> float:
    """Lattice-QCD-flavoured effective degrees of freedom at temperature ``T`` (K).

    Smooth crossover from ``g_hrg`` (T ≪ T_c) to ``g_qgp`` (T ≫ T_c) via tanh,
    with width ``delta_T``. Defaults reproduce the standard 3-flavour QGP
    crossover at the lattice T_c ≈ 155 MeV; overrides allow sensitivity tests.
    """
    if T_kelvin <= 0:
        raise ValueError("temperature must be positive (Kelvin)")
    if delta_T_mev <= 0:
        raise ValueError("delta_T_mev must be positive")
    T_c = T_c_mev * _MEV_TO_K
    dT = delta_T_mev * _MEV_TO_K
    return (g_hrg + g_qgp) / 2.0 + (g_qgp - g_hrg) / 2.0 * math.tanh((T_kelvin - T_c) / dT)


@dataclass(slots=True)
class QuarkGluonPlasmaSource:
    """A speculative confined-QGP power source.

    The energy density is anchored: Stefan–Boltzmann
    ``ε(T) = (π²/30)·g_eff(T)·(k_BT)⁴/(ℏc)³`` with a lattice-QCD-flavoured
    crossover ``g_eff(T)`` from ``g_HRG ≈ 3`` to ``g_QGP ≈ 47.5`` near
    ``T_c ≈ 155 MeV``. Total energy is just ``ε × V``.

    Three parameters are *speculative* — they're listed in
    ``metadata["speculative_components"]`` and live nowhere else in the
    package's anchored physics:

    - ``containment_efficiency``: fraction of the total rest energy convertible
      to graviton flux per ``time_constant_s``.
    - ``graviton_yield`` (η): converts power flux into a graviton emission rate.
    - ``graviton_energy_quantum_j``: assumed energy per graviton emission event.

    Parameters at the boundary may be pint Quantities; they are converted to SI
    once and stored as bare floats for inner-loop speed.
    """

    volume: Any  # m³ (pint Quantity or float)
    temperature: Any  # K (pint Quantity or float)
    containment_efficiency: float = 1.0  # ∈ (0, 1]; speculative
    graviton_yield: float = 1.0  # η; speculative
    graviton_energy_quantum_j: float = 2.0e9  # speculative; ≈ Planck rest energy in J
    time_constant_s: float = 1.0  # rest-energy → power reference time

    _V: float = field(init=False, repr=False)
    _T: float = field(init=False, repr=False)
    metadata: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._V = float(to_si(self.volume, "m**3"))
        self._T = float(to_si(self.temperature, "K"))
        if self._V <= 0:
            raise ValueError("volume must be positive")
        if self._T <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.containment_efficiency <= 1.0:
            raise ValueError("containment_efficiency must be in (0, 1]")
        if self.graviton_yield < 0:
            raise ValueError("graviton_yield must be non-negative")
        if self.graviton_energy_quantum_j <= 0:
            raise ValueError("graviton_energy_quantum_j must be positive")
        if self.time_constant_s <= 0:
            raise ValueError("time_constant_s must be positive")
        self.metadata = {
            "T_K": self._T,
            "T_MeV": self._T / _MEV_TO_K,
            "V_m3": self._V,
            "g_HRG": G_HRG,
            "g_QGP": G_QGP,
            "T_c_MeV": T_C_MEV,
            "delta_T_MeV": DELTA_T_MEV,
            "containment_efficiency": self.containment_efficiency,
            "graviton_yield": self.graviton_yield,
            "graviton_energy_quantum_J": self.graviton_energy_quantum_j,
            "time_constant_s": self.time_constant_s,
            "speculative_components": [
                "containment_efficiency",
                "graviton_yield",
                "graviton_energy_quantum_J",
            ],
        }

    @property
    def temperature_K(self) -> float:  # noqa: N802 — physics convention
        return self._T

    @property
    def volume_m3(self) -> float:
        return self._V

    def energy_density(self) -> float:
        """ε(T) in J/m³, Stefan–Boltzmann form with lattice-flavoured g_eff(T)."""
        g = g_effective(self._T)
        kT = sc.k * self._T  # J
        return (math.pi**2 / 30.0) * g * (kT**4) / (sc.hbar * sc.c) ** 3

    def total_energy(self) -> float:
        """Total QGP rest energy in the confined volume (J)."""
        return self.energy_density() * self._V

    def power_output(self) -> float:
        """Speculative usable power (W) = η_containment · E_total / τ."""
        return self.containment_efficiency * self.total_energy() / self.time_constant_s

    def graviton_emission_rate(self) -> float:
        """Speculative graviton emission rate (events/s) = η_g · P / E_quantum."""
        return self.graviton_yield * self.power_output() / self.graviton_energy_quantum_j
