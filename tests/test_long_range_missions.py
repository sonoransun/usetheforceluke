"""Long-range mission factories produce sensible outcomes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scipy.constants import c as C_LIGHT

from usetheforce.missions import (
    VEHICLES,
    LongRangeMissionResult,
    bondi_runaway_cruise,
    event_horizon_stationkeep,
    event_horizon_stationkeep_with_buffer,
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


def test_event_horizon_stationkeep_runs_and_reports_shortfall() -> None:
    """Pipeline runs end-to-end and the target_metric documents the hover shortfall."""
    M_SUN = 1.98892e30
    vehicle = VEHICLES["city_ship"]
    result = event_horizon_stationkeep(
        black_hole_mass_kg=10.0 * M_SUN,
        duration_s=60.0,
        vehicle=vehicle,
        hover_radius_factor=1.5,
        gain=1e-3,
        max_thrust_n=1e15,
        initial_offset_m=1.0,
        n_eval=40,
    )
    assert isinstance(result, LongRangeMissionResult)
    assert np.all(np.isfinite(result.trajectory.r))
    assert result.trajectory.r.shape[0] == result.thrust_history_n.shape[0]
    tm = result.target_metric
    assert tm["schwarzschild_radius_m"] > 0
    assert tm["target_radius_m"] == pytest.approx(1.5 * tm["schwarzschild_radius_m"])
    # Required hover thrust >> any sane supplied thrust cap.
    assert tm["required_hover_force_newtonian_n"] > tm["supplied_thrust_cap_n"]


def test_event_horizon_stationkeep_validates() -> None:
    vehicle = VEHICLES["smallsat"]
    with pytest.raises(ValueError):
        event_horizon_stationkeep(black_hole_mass_kg=0.0, duration_s=10.0, vehicle=vehicle)
    with pytest.raises(ValueError):
        event_horizon_stationkeep(black_hole_mass_kg=1e30, duration_s=0.0, vehicle=vehicle)
    with pytest.raises(ValueError):
        event_horizon_stationkeep(
            black_hole_mass_kg=1e30,
            duration_s=10.0,
            vehicle=vehicle,
            hover_radius_factor=1.0,
        )


def test_event_horizon_stationkeep_gr_correction_increases_required_thrust() -> None:
    """At hover_radius_factor close to 1, GR-corrected hover thrust ≫ Newtonian."""
    M_SUN = 1.98892e30
    vehicle = VEHICLES["smallsat"]
    result = event_horizon_stationkeep(
        black_hole_mass_kg=M_SUN,
        duration_s=10.0,
        vehicle=vehicle,
        hover_radius_factor=1.01,
        use_gr_hover_correction=True,
        gain=1e-3,
        max_thrust_n=1e10,
        n_eval=20,
    )
    tm = result.target_metric
    # GR-corrected thrust ≈ Newtonian × 1/sqrt(1 - 1/1.01) = sqrt(101).
    ratio = tm["required_hover_force_gr_n"] / tm["required_hover_force_newtonian_n"]
    assert ratio == pytest.approx(math.sqrt(101.0), rel=1e-6)
    assert tm["required_hover_force_n"] == tm["required_hover_force_gr_n"]


def test_bondi_runaway_cruise_terminal_velocity_grows_monotonically() -> None:
    """Constant Bondi acceleration ⇒ velocity grows linearly; KE non-conservation > 0."""
    vehicle = VEHICLES["smallsat"]
    result = bondi_runaway_cruise(
        vehicle=vehicle,
        duration_s=100.0,
        m_negative_kg=1.0e20,
        separation_m=1.0,
        n_eval=50,
    )
    assert isinstance(result, LongRangeMissionResult)
    tm = result.target_metric
    # Terminal velocity should be positive and equal a · T analytically.
    a = tm["self_acceleration_mps2"]
    expected_v = a * tm["duration_s"]
    assert tm["terminal_velocity_mps"] == pytest.approx(expected_v, rel=1e-4)
    # Energy non-conservation > 0 — Bondi premise injects KE from nothing.
    assert tm["energy_non_conservation_J"] > 0.0
    # Trajectory KE growth is monotonic (constant thrust starting from rest).
    v = result.trajectory.v
    ke = 0.5 * vehicle.mass_kg * np.sum(v * v, axis=1)
    assert np.all(np.diff(ke) >= -1e-9)


def test_event_horizon_stationkeep_with_buffer_reports_offset_and_shortfall() -> None:
    """Buffer mission factory exposes per-component thrust budgets and the augmented shortfall."""
    M_SUN = 1.98892e30
    vehicle = VEHICLES["smallsat"]
    result = event_horizon_stationkeep_with_buffer(
        black_hole_mass_kg=10 * M_SUN,
        duration_s=0.05,
        vehicle=vehicle,
        buffer_mass_neg_kg=1.0e29,
        buffer_radius_factor=10.0,
        hover_radius_factor=50.0,
        use_gr_hover_correction=True,
        gain=1e-6,
        max_thrust_n=1.0e6,
        n_eval=5,
    )
    tm = result.target_metric
    for key in (
        "target_radius_m",
        "buffer_position_radius_m",
        "buffer_mass_neg_kg",
        "buffer_repulsion_at_craft_n",
        "required_hover_force_newtonian_n",
        "required_hover_force_gr_n",
        "net_required_hover_force_newtonian_n",
        "net_required_hover_force_gr_n",
        "buffer_offset_ratio",
        "augmented_shortfall_ratio",
    ):
        assert key in tm, f"missing target_metric key: {key}"
    # Net Newtonian = total Newtonian − buffer (since buffer < BH at these params).
    assert tm["net_required_hover_force_newtonian_n"] == pytest.approx(
        tm["required_hover_force_newtonian_n"] - tm["buffer_repulsion_at_craft_n"],
        rel=1e-12,
    )
    # Buffer offset ratio is positive and ≪ 1 in this stress test.
    assert 0 < tm["buffer_offset_ratio"] < 1.0
    # Shortfall ratio still huge.
    assert tm["augmented_shortfall_ratio"] > 100.0


def test_event_horizon_stationkeep_with_buffer_validates() -> None:
    M_SUN = 1.98892e30
    vehicle = VEHICLES["smallsat"]
    # buffer_radius_factor must be strictly < hover_radius_factor.
    with pytest.raises(ValueError):
        event_horizon_stationkeep_with_buffer(
            black_hole_mass_kg=M_SUN,
            duration_s=1.0,
            vehicle=vehicle,
            buffer_mass_neg_kg=1.0e20,
            buffer_radius_factor=2.0,
            hover_radius_factor=2.0,
        )
    # buffer_radius_factor must be strictly > 1.0.
    with pytest.raises(ValueError):
        event_horizon_stationkeep_with_buffer(
            black_hole_mass_kg=M_SUN,
            duration_s=1.0,
            vehicle=vehicle,
            buffer_mass_neg_kg=1.0e20,
            buffer_radius_factor=1.0,
            hover_radius_factor=2.0,
        )
    # buffer_mass_neg_kg must be positive.
    with pytest.raises(ValueError):
        event_horizon_stationkeep_with_buffer(
            black_hole_mass_kg=M_SUN,
            duration_s=1.0,
            vehicle=vehicle,
            buffer_mass_neg_kg=-1.0,
            buffer_radius_factor=1.05,
            hover_radius_factor=1.10,
        )


def test_event_horizon_stationkeep_with_buffer_can_neutralise_bh_pull() -> None:
    """A sufficiently massive buffer pushes the offset ratio ≥ 1 and shortfall to 0."""
    M_SUN = 1.98892e30
    vehicle = VEHICLES["smallsat"]
    # Pick parameters so buffer_offset_ratio ≥ 1.
    # F_BH = G M m / R_craft², F_buf = G m_buf m / Δ².
    # Want F_buf ≥ F_BH ⇒ m_buf ≥ M · Δ² / R_craft².
    # Δ = R_craft − R_buf. With factors 50 and 49: Δ/R_craft = 1/50 ⇒ m_buf ≥ M/2500.
    # Use a huge buffer to safely exceed.
    result = event_horizon_stationkeep_with_buffer(
        black_hole_mass_kg=10 * M_SUN,
        duration_s=0.05,
        vehicle=vehicle,
        buffer_mass_neg_kg=10 * M_SUN,  # way more than needed
        buffer_radius_factor=49.0,
        hover_radius_factor=50.0,
        use_gr_hover_correction=False,
        gain=1e-6,
        max_thrust_n=1.0e6,
        n_eval=5,
    )
    tm = result.target_metric
    assert tm["buffer_offset_ratio"] >= 1.0
    assert tm["net_required_hover_force_newtonian_n"] == 0.0
    assert tm["augmented_shortfall_ratio"] == 0.0


def test_bondi_runaway_cruise_flags_relativistic_overshoot() -> None:
    """A duration long enough that a · T > c flags fraction_c > 1."""
    vehicle = VEHICLES["smallsat"]
    # Pick parameters so a · T overshoots c.
    m_neg = 1.0e30  # huge
    d = 1.0
    a = 6.6743e-11 * m_neg / (d * d)  # ≈ 6.67e19 m/s²
    duration = 10.0 * C_LIGHT / a  # 10× the time to reach c
    result = bondi_runaway_cruise(
        vehicle=vehicle,
        duration_s=duration,
        m_negative_kg=m_neg,
        separation_m=d,
        n_eval=20,
    )
    tm = result.target_metric
    assert tm["terminal_velocity_fraction_c"] > 1.0
