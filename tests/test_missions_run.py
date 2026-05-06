"""Mission runner: a short integrated burn produces the expected Δv from F·t/m."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce.missions import ALL_ADAPTERS, MISSIONS, VEHICLES, run_mission
from usetheforce.missions.missions import _ConstantThrustField


def test_free_burn_smallsat_shaped_field() -> None:
    """In free space, Δv = F·t / m exactly for constant thrust."""
    vehicle = VEHICLES["smallsat"]
    mission = MISSIONS["free_burn_100s"]
    result = run_mission(
        vehicle, "shaped_field_ansatz", ALL_ADAPTERS["shaped_field_ansatz"], mission
    )
    expected_dv = result.thrust_n * result.burn_time_s / vehicle.mass_kg
    assert result.delta_v_mps == pytest.approx(expected_dv, rel=1e-6)
    assert result.peak_g > 0
    assert "vehicle_power_W" in result.assumptions


def test_run_mission_rejects_inapplicable_model() -> None:
    vehicle = VEHICLES["smallsat"]
    mission = MISSIONS["free_burn_100s"]
    with pytest.raises(ValueError, match="not applicable"):
        run_mission(
            vehicle,
            "parallel_plate_casimir",
            ALL_ADAPTERS["parallel_plate_casimir"],
            mission,
        )


def test_burn_under_central_gravity_runs_and_is_finite() -> None:
    """Under central gravity Δv reflects both thrust and the orbital velocity rotation;
    we just verify the integration runs and produces sane values, not a closed form.
    """
    vehicle = VEHICLES["interplanetary"]
    mission = MISSIONS["leo_raise_100s"]
    result = run_mission(
        vehicle, "antimatter_graviton", ALL_ADAPTERS["antimatter_graviton"], mission
    )
    assert result.delta_v_mps > 0
    assert np.all(np.isfinite(result.trajectory.r))
    assert np.all(np.isfinite(result.trajectory.v))
    # Acceleration is non-trivial: thrust isn't lost in numerical noise.
    assert result.peak_accel_mps2 > 0


def test_constant_thrust_field_independent_of_position() -> None:
    """The wrapper applies a constant body-frame thrust regardless of r."""
    vehicle = VEHICLES["smallsat"]

    def zero_bg(r: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    f = _ConstantThrustField(
        thrust_n=10.0,
        axis=(1.0, 0.0, 0.0),
        vehicle=vehicle,
        background=zero_bg,
        underlying_metadata={"avenue": "test", "model": "test", "speculative": True},
    )
    f0 = f.force(0.0, np.zeros(3))
    f1 = f.force(0.0, np.array([1e6, 1e6, 1e6]))
    np.testing.assert_array_equal(f0, f1)
    np.testing.assert_array_equal(f0, np.array([10.0, 0.0, 0.0]))
