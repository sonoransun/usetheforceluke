"""Adapter contracts: each returns a valid ForceField; |F(r_ref)| ≈ power / V_REF."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.missions import ALL_ADAPTERS, VEHICLES
from usetheforce.missions.adapters import V_REF


@pytest.mark.parametrize("model_key", list(ALL_ADAPTERS))
def test_adapter_returns_force_field(model_key: str) -> None:
    adapter = ALL_ADAPTERS[model_key]
    vehicle = VEHICLES["smallsat"]
    result = adapter(vehicle, vehicle.power_w)
    assert isinstance(result.field, ForceField)
    assert isinstance(result.assumptions, dict)
    assert "vehicle_power_W" in result.assumptions


@pytest.mark.parametrize(
    "model_key",
    [
        "shaped_field_ansatz",
        "heavy_element_lattice",
        "stimulated_emission_array",
        "antimatter_graviton",
    ],
)
@pytest.mark.parametrize("vehicle_key", list(VEHICLES.keys()))
def test_force_at_reference_matches_power_target(model_key: str, vehicle_key: str) -> None:
    """For applicable models: |F(r_ref)| should equal power / V_REF."""
    vehicle = VEHICLES[vehicle_key]
    result = ALL_ADAPTERS[model_key](vehicle, vehicle.power_w)
    assert result.applicable
    r_probe = np.array([result.r_ref_m, 0.0, 0.0])
    f_mag = float(np.linalg.norm(result.field.force(0.0, r_probe)))
    expected = vehicle.power_w / V_REF
    assert f_mag == pytest.approx(expected, rel=1e-6), (
        f"{model_key} on {vehicle_key}: got {f_mag}, expected {expected}"
    )


@pytest.mark.parametrize("model_key", ["parallel_plate_casimir", "scaled_casimir"])
def test_casimir_variants_flagged_inapplicable(model_key: str) -> None:
    vehicle = VEHICLES["smallsat"]
    result = ALL_ADAPTERS[model_key](vehicle, vehicle.power_w)
    assert result.applicable is False
    assert result.reason  # non-empty explanation


def test_blackhole_counter_drive_adapter_registered() -> None:
    """Blackhole adapter is in the registry and exposes BH-specific assumptions."""
    assert "blackhole_counter_drive" in ALL_ADAPTERS
    vehicle = VEHICLES["city_ship"]
    result = ALL_ADAPTERS["blackhole_counter_drive"](vehicle, vehicle.power_w)
    assert result.applicable is True
    assert result.r_ref_m > 0
    a = result.assumptions
    for key in (
        "bh_mass_kg",
        "bh_mass_solar",
        "r_s_m",
        "r_ref_m",
        "hover_radius_factor",
        "efficiency_eta",
        "required_hover_force_n",
        "supplied_force_n",
        "hover_shortfall_ratio",
    ):
        assert key in a, f"missing assumption: {key}"


def test_blackhole_counter_drive_force_at_reference_matches_power_target() -> None:
    """For realistic vehicles η ≪ 1; the supplied counter-thrust = power / V_REF."""
    vehicle = VEHICLES["city_ship"]
    result = ALL_ADAPTERS["blackhole_counter_drive"](vehicle, vehicle.power_w)
    r_probe = np.array([result.r_ref_m, 0.0, 0.0])
    f_mag = float(np.linalg.norm(result.field.force(0.0, r_probe)))
    expected = vehicle.power_w  # V_REF = 1 m/s
    assert f_mag == pytest.approx(expected, rel=1e-6)
    # Shortfall ratio is huge — the headline finding.
    assert result.assumptions["hover_shortfall_ratio"] > 1e3
