"""Power → ForceField adapters per model.

Each adapter picks a model's free parameters from a vehicle's power budget,
evaluates the resulting force at a model-specific *reference radius*, and
returns the ``ForceField`` together with an ``assumptions`` block stating the
speculative coupling used. The two Casimir variants are explicitly flagged
``applicable=False`` for free-flight propulsion (no momentum transfer to the
vehicle's centre of mass in steady state).

The whole point of this file is to keep the speculative scalings *visible* —
when in doubt, read the ``assumptions`` dict that accompanies every adapter
return.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.casimir import ParallelPlateCasimir, ScaledCasimir
from usetheforce.missions.vehicles import Vehicle
from usetheforce.protocol import ForceField
from usetheforce.qfield import HeavyElementLattice, ShapedFieldAnsatz, StimulatedEmissionArray
from usetheforce.qgp import QGPGravitonField, QuarkGluonPlasmaSource
from usetheforce.qgp.source import T_C_MEV
from usetheforce.units import ureg

# Reference probe velocity used to convert "power dissipated" → "force": P = F · v.
# Stated assumption; printed in the report.
V_REF: float = 1.0  # m/s


@dataclass(slots=True)
class AdapterResult:
    """What an adapter returns: the field, where to sample it, and the assumptions."""

    field: ForceField
    r_ref_m: float
    applicable: bool
    reason: str
    assumptions: dict[str, Any]


Adapter = Callable[[Vehicle, float], AdapterResult]


# --- Casimir variants (applicable=False) -------------------------------------------------


def parallel_plate_casimir_adapter(vehicle: Vehicle, power: float) -> AdapterResult:
    """Static parallel-plate Casimir at 1 cm² × 100 nm. Anchored physics, but not propulsive."""
    pp = ParallelPlateCasimir(area=1.0 * ureg("cm^2"), separation=100.0 * ureg("nm"))
    return AdapterResult(
        field=pp,
        r_ref_m=100e-9,
        applicable=False,
        reason="Cavity force is internal; no centre-of-mass thrust in steady state",
        assumptions={
            "geometry": "parallel plates, 1 cm², 100 nm separation",
            "power_used": "0 (passive geometry)",
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
            "note": "Real physics; reported for completeness but not propulsive",
        },
    )


def scaled_casimir_adapter(vehicle: Vehicle, power: float) -> AdapterResult:
    """Scaled-geometry Casimir variant, geometry factor g=1. Still not propulsive."""
    sc = ScaledCasimir(
        area=1.0 * ureg("cm^2"),
        separation=100.0 * ureg("nm"),
        geometry_factor=1.0,
    )
    return AdapterResult(
        field=sc,
        r_ref_m=100e-9,
        applicable=False,
        reason="Same internal-force objection as parallel-plate Casimir",
        assumptions={
            "geometry_factor": 1.0,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
            "note": "Speculative scaling; still not net-propulsive",
        },
    )


# --- Speculative free-flight models (applicable=True) -----------------------------------


def shaped_field_ansatz_adapter(
    vehicle: Vehicle,
    power: float,
    sigma_m: float = 1.0,
) -> AdapterResult:
    """Gaussian-well ansatz. Choose A so |F(r=σ)| = power / V_REF."""
    target_force = power / V_REF
    # |F(r=σ)| = A · exp(-1/2) / σ  ⇒  A = |F| · σ · exp(1/2)
    amplitude = target_force * sigma_m * math.exp(0.5)
    field = ShapedFieldAnsatz(amplitude=amplitude, sigma=sigma_m)
    return AdapterResult(
        field=field,
        r_ref_m=sigma_m,
        applicable=True,
        reason="",
        assumptions={
            "scaling": "A chosen so |F(r=σ)| = power / V_REF",
            "sigma_m": sigma_m,
            "amplitude": amplitude,
            "v_ref_mps": V_REF,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
        },
    )


def heavy_element_lattice_adapter(
    vehicle: Vehicle,
    power: float,
    r_ref_m: float = 10.0,
    softening_m: float = 1.0,
) -> AdapterResult:
    """Single-site softened-Coulomb. Choose κ so |F(r=r_ref)| = power / V_REF."""
    target_force = power / V_REF
    # |F| = κ · r / (r² + ε²)^(3/2)  ⇒  κ = |F| · (r² + ε²)^(3/2) / r
    kappa = target_force * (r_ref_m**2 + softening_m**2) ** 1.5 / r_ref_m
    field = HeavyElementLattice(
        sites=np.zeros((1, 3)),
        strengths=np.array([1.0]),
        coupling=kappa,
        softening=softening_m,
    )
    return AdapterResult(
        field=field,
        r_ref_m=r_ref_m,
        applicable=True,
        reason="",
        assumptions={
            "scaling": "κ chosen so |F(r=r_ref)| = power / V_REF",
            "r_ref_m": r_ref_m,
            "softening_m": softening_m,
            "coupling_kappa": kappa,
            "v_ref_mps": V_REF,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
        },
    )


def stimulated_emission_array_adapter(
    vehicle: Vehicle,
    power: float,
    r_ref_m: float = 1.0,
    wavenumber: float = 1.0,
) -> AdapterResult:
    """Single coherent emitter at origin. Choose A so |F(r=r_ref)| = power / V_REF."""
    target_force = power / V_REF
    # Single emitter, α=1: |F| = 2 A² / R³  ⇒  A = sqrt(|F| · R³ / 2)
    amplitude = math.sqrt(target_force * r_ref_m**3 / 2.0)
    field = StimulatedEmissionArray(
        positions=np.zeros((1, 3)),
        amplitudes=[amplitude],
        phases=[0.0],
        wavenumber=wavenumber,
        coupling=1.0,
    )
    return AdapterResult(
        field=field,
        r_ref_m=r_ref_m,
        applicable=True,
        reason="",
        assumptions={
            "scaling": "amplitude chosen so |F(r=r_ref)| = power / V_REF for a single emitter",
            "r_ref_m": r_ref_m,
            "wavenumber": wavenumber,
            "amplitude": amplitude,
            "v_ref_mps": V_REF,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
        },
    )


def antimatter_graviton_adapter(
    vehicle: Vehicle,
    power: float,
    r_ref_m: float = 1.0,
    screening_m: float = 1.0e6,
    coupling_g: float = 1.0,
    efficiency: float = 1.0,
) -> AdapterResult:
    """Yukawa graviton from annihilation. Choose Γ so |F(r=r_ref)| = power / V_REF · η."""
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    target_force = (power * efficiency) / V_REF
    # |F| = m_p · g · Γ · exp(-r/λ) · (r + λ) / (λ r²)
    pre = (
        coupling_g
        * math.exp(-r_ref_m / screening_m)
        * (r_ref_m + screening_m)
        / (screening_m * r_ref_m**2)
    )
    gamma = target_force / (vehicle.mass_kg * pre)
    field = AntimatterGravitonField(
        source=np.zeros(3),
        gamma=gamma,
        coupling=coupling_g,
        screening=screening_m,
        probe_mass=vehicle.mass_kg,
    )
    return AdapterResult(
        field=field,
        r_ref_m=r_ref_m,
        applicable=True,
        reason="",
        assumptions={
            "scaling": "Γ chosen so |F(r=r_ref)| = (power · η) / V_REF",
            "r_ref_m": r_ref_m,
            "screening_lambda_m": screening_m,
            "coupling_g": coupling_g,
            "efficiency_eta": efficiency,
            "annihilation_rate_gamma": gamma,
            "probe_mass_kg": vehicle.mass_kg,
            "v_ref_mps": V_REF,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
        },
    )


def qgp_graviton_adapter(
    vehicle: Vehicle,
    power: float,
    r_ref_m: float = 1.0,
    screening_m: float = 1.0e6,
    coupling_g: float = 1.0,
    temperature_mev: float = 200.0,
    volume_scale_m3_per_kg_third: float = 1.0,
) -> AdapterResult:
    """QGP-sourced graviton field. Anchored ε(T) × speculative graviton coupling.

    Volume scales with vehicle.mass^(1/3) so larger ships have larger QGP cores.
    Temperature is fixed at 200 MeV (well above T_c ≈ 155 MeV) so g_eff sits at
    its plateau.

    The *only* speculative knob this adapter rescales is ``graviton_yield``;
    ``containment_efficiency`` is held at 1.0 by convention so the speculative
    leap is a single number, traceable in the assumptions block.
    """
    if temperature_mev <= T_C_MEV:
        raise ValueError(
            f"temperature_mev={temperature_mev} must exceed T_c={T_C_MEV} MeV (deconfinement)"
        )
    target_force = power / V_REF
    volume_m3 = volume_scale_m3_per_kg_third * vehicle.mass_kg ** (1 / 3)
    temperature_k = temperature_mev * 1e6 * 1.602176634e-19 / 1.380649e-23  # MeV→K
    # Convention: containment_efficiency stays at 1.0; we calibrate ONLY graviton_yield.
    natural_source = QuarkGluonPlasmaSource(
        volume=volume_m3,
        temperature=temperature_k,
        containment_efficiency=1.0,
        graviton_yield=1.0,
    )
    # Yukawa pre-factor at r_ref: |F| = m_p · g · Γ · pre(r_ref).
    pre = (
        coupling_g
        * math.exp(-r_ref_m / screening_m)
        * (r_ref_m + screening_m)
        / (screening_m * r_ref_m**2)
    )
    desired_gamma = target_force / (vehicle.mass_kg * pre)
    natural_gamma = natural_source.graviton_emission_rate()
    if natural_gamma <= 0:
        raise RuntimeError("QGP source produced zero graviton rate at chosen T/V")
    yield_factor = desired_gamma / natural_gamma
    # Rebuild the source with the calibrated yield.
    source = QuarkGluonPlasmaSource(
        volume=volume_m3,
        temperature=temperature_k,
        containment_efficiency=1.0,
        graviton_yield=yield_factor,
    )
    assert source.containment_efficiency == 1.0, (
        "containment_efficiency must remain 1.0 — graviton_yield is the only calibration knob"
    )
    field = QGPGravitonField(
        source=source,
        source_position=np.zeros(3),
        screening_length=screening_m,
        coupling_g=coupling_g,
        probe_mass=vehicle.mass_kg,
    )
    return AdapterResult(
        field=field,
        r_ref_m=r_ref_m,
        applicable=True,
        reason="",
        assumptions={
            "scaling": "graviton_yield chosen so |F(r=r_ref)| = power / V_REF",
            "r_ref_m": r_ref_m,
            "screening_lambda_m": screening_m,
            "coupling_g": coupling_g,
            "temperature_MeV": temperature_mev,
            "volume_m3": volume_m3,
            "qgp_energy_density_J_per_m3": source.energy_density(),
            "qgp_total_energy_J": source.total_energy(),
            "graviton_yield_eta": yield_factor,
            "annihilation_rate_gamma": field.gamma,
            "v_ref_mps": V_REF,
            "vehicle_power_W": power,
            "vehicle_mass_kg": vehicle.mass_kg,
            "anchored_components": "Stefan–Boltzmann ε(T) with lattice-QCD g_eff(T)",
            "speculative_components": (
                "graviton emission coupling, Yukawa potential form, conversion to thrust"
            ),
        },
    )


ALL_ADAPTERS: dict[str, Adapter] = {
    "parallel_plate_casimir": parallel_plate_casimir_adapter,
    "scaled_casimir": scaled_casimir_adapter,
    "shaped_field_ansatz": shaped_field_ansatz_adapter,
    "heavy_element_lattice": heavy_element_lattice_adapter,
    "stimulated_emission_array": stimulated_emission_array_adapter,
    "antimatter_graviton": antimatter_graviton_adapter,
    "qgp_graviton": qgp_graviton_adapter,
}
