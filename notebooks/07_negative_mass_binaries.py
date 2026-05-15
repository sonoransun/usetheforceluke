"""Negative-mass binaries demo — Bondi runaway, anti-chirp, unstable, dipole-GW, buffer.

EXCEPTIONALLY SPECULATIVE. Each of the five negmass sub-models requires the
existence of negative inertial mass (Bondi 1957); two of them additionally
require an equivalence-principle violation or specific binary configuration.
This notebook surfaces the speculative parameters at the call site and
demonstrates the ``CompositeField`` seam by bolting a Bondi pair onto an
existing antimatter-graviton drive.

Headline numbers (printed below):

- ``BondiRunawayPair`` self-acceleration ``a = G·|m_neg|/d²`` — independent of
  payload mass. The ``bondi_runaway_cruise`` factory exposes the energy
  non-conservation that's built into the premise.
- The negative-mass *buffer* placed between a craft and a Schwarzschild
  black hole partially cancels the BH pull. ``event_horizon_stationkeep_with_buffer``
  exposes ``buffer_offset_ratio`` and ``augmented_shortfall_ratio``.

Run as a script (``python notebooks/07_negative_mass_binaries.py``) or convert
to a notebook with ``jupytext --to ipynb 07_negative_mass_binaries.py``.
"""

# %% Imports
from __future__ import annotations

import os

import numpy as np

from usetheforce import CompositeField
from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.missions import VEHICLES, bondi_runaway_cruise
from usetheforce.negmass import (
    AntiChirpBinary,
    BondiRunawayPair,
    DipoleGravitonRadiator,
    NegativeMassPointSource,
    NegativeTotalMassBinary,
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# %% Construct one of each sub-model with safe defaults and print the speculative leap.
print("=== Negmass sub-models (avenue = 'negmass'; all speculative=True) ===\n")
models = [
    (
        "BondiRunawayPair",
        BondiRunawayPair(m_negative_kg=1.0e20, separation_m=1.0, craft_mass_kg=1.0e6),
    ),
    (
        "AntiChirpBinary",
        AntiChirpBinary(
            m_positive_kg=10.0, m_negative_kg=1.0, separation_m=1.0e6, probe_mass_kg=100.0
        ),
    ),
    (
        "NegativeTotalMassBinary",
        NegativeTotalMassBinary(
            m_positive_kg=1.0, m_negative_kg=5.0, separation_m=1.0, craft_mass_kg=1.0e6
        ),
    ),
    (
        "DipoleGravitonRadiator",
        DipoleGravitonRadiator(
            m_negative_kg=1.0, separation_m=1.0, wep_violation=0.5, probe_mass_kg=1.0
        ),
    ),
    (
        "NegativeMassPointSource",
        NegativeMassPointSource(m_negative_kg=1.0e20, probe_mass_kg=1.0),
    ),
]
for name, m in models:
    sc = m.metadata["speculative_components"]
    applicable = m.metadata.get("applicable_for_trajectory", True)
    print(f"{name}")
    print(f"  speculative_components = {sc}")
    print(f"  applicable_for_trajectory = {applicable}")
    print(f"  citation: {m.metadata['citation']}")
    print()


# %% Bondi runaway cruise — the energy-non-conservation triple.
city = VEHICLES["city_ship"]
print(f"=== bondi_runaway_cruise({city.key}) ===")
result = bondi_runaway_cruise(
    vehicle=city,
    duration_s=100.0,
    m_negative_kg=1.0e20,
    separation_m=1.0,
    n_eval=50,
)
tm = result.target_metric
print(f"  self_acceleration_mps2     = {tm['self_acceleration_mps2']:.6g}")
print(f"  terminal_velocity_mps      = {tm['terminal_velocity_mps']:.6g}")
print(f"  terminal_velocity / c      = {tm['terminal_velocity_fraction_c']:.6g}")
print(f"  energy_non_conservation_J  = {tm['energy_non_conservation_J']:.6g}")
print()


# %% CompositeField — bolt a Bondi appendage onto an antimatter-graviton drive.
print("=== CompositeField(antimatter_graviton, bondi_pair) on city_ship ===")
graviton = AntimatterGravitonField(
    source=(0.0, 0.0, 0.0),
    gamma=1.0,
    coupling=1.0,
    screening=1.0e6,
    probe_mass=city.mass_kg,
)
bondi = BondiRunawayPair(
    m_negative_kg=1.0e20, separation_m=1.0, craft_mass_kg=city.mass_kg
)
augmented = CompositeField(graviton, bondi)
print(f"  composite metadata['avenue']               = {augmented.metadata['avenue']}")
print(
    f"  composite metadata['speculative']          = {augmented.metadata['speculative']}"
)
print(
    f"  composite metadata['applicable_for_trajectory'] = "
    f"{augmented.metadata['applicable_for_trajectory']}"
)
print(
    f"  composite metadata['speculative_components']: "
    f"{augmented.metadata['speculative_components']}"
)
# Sample the composite force at a far probe.
probe = np.array([1.0e6, 0.0, 0.0])
f_g = graviton.force(0.0, probe)
f_b = bondi.force(0.0, probe)
f_sum = augmented.force(0.0, probe)
print(f"  |F_graviton(probe)|         = {float(np.linalg.norm(f_g)):.4e} N")
print(f"  |F_bondi(probe)|            = {float(np.linalg.norm(f_b)):.4e} N (constant body force)")
print(f"  |F_composite(probe)|        = {float(np.linalg.norm(f_sum)):.4e} N")
print()
print(
    "The two thrust scales are independent: graviton scales with R; Bondi is a "
    "constant body force on the rigid composite (no R dependence)."
)
