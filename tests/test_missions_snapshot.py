"""Snapshot grid contracts."""

from __future__ import annotations

import itertools

import numpy as np

from usetheforce.missions import ALL_ADAPTERS, VEHICLES, evaluate_snapshot


def test_matrix_has_full_grid() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    assert set(matrix.keys()) == set(VEHICLES.keys())
    for row in matrix.values():
        assert set(row.keys()) == set(ALL_ADAPTERS.keys())


def test_casimir_cells_marked_inapplicable() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    for row in matrix.values():
        assert row["parallel_plate_casimir"].applicable is False
        assert row["scaled_casimir"].applicable is False


def test_applicable_cells_have_finite_metrics() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    applicable_models = {
        "shaped_field_ansatz",
        "heavy_element_lattice",
        "stimulated_emission_array",
        "antimatter_graviton",
    }
    for row in matrix.values():
        for mk in applicable_models:
            cell = row[mk]
            assert cell.applicable
            assert np.isfinite(cell.force_n) and cell.force_n > 0
            assert np.isfinite(cell.accel_mps2) and cell.accel_mps2 > 0
            assert np.isfinite(cell.energy_per_dv_jspm) and cell.energy_per_dv_jspm > 0


def test_acceleration_decreases_with_vehicle_mass() -> None:
    """Same |F|=power/V_REF, divided by larger m, gives smaller a."""
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    accels = [matrix[vk]["shaped_field_ansatz"].accel_mps2 for vk in VEHICLES]
    # Power scales sub-linearly with mass, so accel = (power/V_REF) / mass strictly decreases.
    assert all(a_next < a_prev for a_prev, a_next in itertools.pairwise(accels))
