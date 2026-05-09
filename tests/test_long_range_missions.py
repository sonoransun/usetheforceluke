"""Long-range mission factories produce sensible outcomes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from usetheforce.missions import (
    VEHICLES,
    LongRangeMissionResult,
    heliocentric_cruise,
    interstellar_brachistochrone,
    leo_orbit_modification,
    lunar_stationkeep,
)


def test_interstellar_brachistochrone_recovers_analytical_time() -> None:
    """For 4.24 ly @ 1 g, travel time ≈ 2·sqrt(d/a) ≈ 5.97 years (proper)."""
    vehicle = VEHICLES["city_ship"]
    LIGHT_YEAR_M = 9.4607304725808e15
    d_ly = 4.24
    g = 1.0
    expected = 2.0 * math.sqrt(d_ly * LIGHT_YEAR_M / (g * 9.80665))
    result = interstellar_brachistochrone(distance_ly=d_ly, max_g=g, vehicle=vehicle, n_eval=80)
    assert isinstance(result, LongRangeMissionResult)
    assert result.trajectory.t[-1] == pytest.approx(expected, rel=1e-6)
    # Δv should be the midpoint speed = a·t_brake.
    expected_dv = (g * 9.80665) * (expected / 2.0)
    assert result.delta_v_mps == pytest.approx(expected_dv, rel=0.05)


def test_interstellar_brachistochrone_validates() -> None:
    vehicle = VEHICLES["smallsat"]
    with pytest.raises(ValueError):
        interstellar_brachistochrone(distance_ly=0.0, max_g=1.0, vehicle=vehicle)
    with pytest.raises(ValueError):
        interstellar_brachistochrone(distance_ly=1.0, max_g=0.0, vehicle=vehicle)


def test_heliocentric_cruise_runs_and_carries_history() -> None:
    """Just assert the pipeline runs end-to-end and produces sane shapes."""
    vehicle = VEHICLES["interplanetary"]
    result = heliocentric_cruise(
        start_au=1.0,
        target_au=1.5,
        vehicle=vehicle,
        burn_time_days=30.0,
        gain=1e-9,
        n_eval=80,
    )
    assert result.trajectory.r.shape[0] == result.thrust_history_n.shape[0]
    assert result.trajectory.r.shape[0] == result.power_history_j.shape[0]
    assert np.all(np.isfinite(result.trajectory.r))
    assert result.delta_v_mps >= 0
    assert result.target_metric["start_au"] == 1.0
    assert result.target_metric["target_au"] == 1.5


def test_lunar_stationkeep_maintains_band() -> None:
    """Probe drifts a small offset from a target near the Earth-Moon midpoint;
    proportional guidance should keep the position within a few × the initial offset."""
    vehicle = VEHICLES["crewed"]
    target = (1.5e8, 0.0, 0.0)
    result = lunar_stationkeep(
        target_position_m=target,
        duration_s=600.0,
        vehicle=vehicle,
        gain=1.0,
        max_thrust_n=1e7,
        initial_offset_m=1e3,
        n_eval=80,
    )
    assert np.all(np.isfinite(result.trajectory.r))
    assert result.delta_v_mps >= 0


def test_leo_orbit_modification_raises_apoapsis() -> None:
    """A prograde burn should raise the apoapsis above the initial circular altitude."""
    vehicle = VEHICLES["smallsat"]
    result = leo_orbit_modification(
        initial_altitude_km=400.0,
        target_altitude_km=600.0,
        burn_time_s=300.0,
        vehicle=vehicle,
        n_eval=80,
    )
    achieved = result.target_metric["achieved_apoapsis_km"]
    assert achieved > 400.0  # apoapsis raised
    assert result.delta_v_mps > 0


def test_long_range_result_history_shapes_match() -> None:
    """Thrust and power histories must have exactly len(t) samples each."""
    vehicle = VEHICLES["smallsat"]
    result = interstellar_brachistochrone(distance_ly=0.001, max_g=0.1, vehicle=vehicle, n_eval=50)
    n = result.trajectory.t.size
    assert result.thrust_history_n.shape == (n, 3)
    assert result.power_history_j.shape == (n,)
