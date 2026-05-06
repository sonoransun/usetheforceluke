"""Locks the matched-power normalisation against silent drift.

If any adapter's scaling logic changes accidentally, this test fails. The
reference values are computed analytically from `target_force = power / V_REF`
and `accel = target_force / mass`, so they hold by construction *unless* the
adapter math drifts.
"""

from __future__ import annotations

import pytest

from usetheforce.missions import ALL_ADAPTERS, VEHICLES, evaluate_snapshot
from usetheforce.missions.adapters import V_REF


@pytest.mark.parametrize(
    "model_key",
    [
        "shaped_field_ansatz",
        "heavy_element_lattice",
        "stimulated_emission_array",
        "antimatter_graviton",
        "qgp_graviton",
    ],
)
def test_matched_power_acceleration_invariant(model_key: str) -> None:
    """For every applicable model, accel(vehicle) = vehicle.power / (V_REF · vehicle.mass)."""
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    for vk, vehicle in VEHICLES.items():
        cell = matrix[vk][model_key]
        expected_accel = vehicle.power_w / (V_REF * vehicle.mass_kg)
        assert cell.accel_mps2 == pytest.approx(expected_accel, rel=1e-9), (
            f"{model_key} on {vk}: got {cell.accel_mps2}, expected {expected_accel}"
        )


def test_casimir_columns_inapplicable() -> None:
    """The Casimir columns are flagged inapplicable across every vehicle."""
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    for vk in VEHICLES:
        assert matrix[vk]["parallel_plate_casimir"].applicable is False
        assert matrix[vk]["scaled_casimir"].applicable is False


def test_falloff_ratio_is_smaller_for_faster_decaying_models() -> None:
    """At matched-power, shaped Gaussian decays fastest; Yukawa graviton decays slowest.

    Locks the relative ordering against future regressions in the snapshot.
    """
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    row = matrix["smallsat"]
    # At r=1 km, ε(Gaussian) → 0; |F_he| ~ 1e-4 of peak; |F_emit| ~ 1e-9; |F_grav| ~ 1e-6.
    assert row["shaped_field_ansatz"].falloff_ratio < row["stimulated_emission_array"].falloff_ratio
    assert row["stimulated_emission_array"].falloff_ratio < row["antimatter_graviton"].falloff_ratio
    assert row["antimatter_graviton"].falloff_ratio < row["heavy_element_lattice"].falloff_ratio
