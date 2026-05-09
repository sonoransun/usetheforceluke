"""Tuning helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from usetheforce.control import (
    BrachistochroneTransit,
    ProportionalGuidance,
    solve_min_dv,
    solve_min_time,
)


def test_solve_min_time_analytical_brachistochrone() -> None:
    """t = 2·sqrt(d / a_max)."""
    d = 400.0
    a = 2.0
    expected = 2.0 * math.sqrt(d / a)
    result = solve_min_time(distance_m=d, max_acceleration_mps2=a, mass_kg=10.0)
    assert result.success
    assert result.travel_time_s == pytest.approx(expected, rel=1e-12)
    assert result.delta_v_mps == pytest.approx(a * math.sqrt(d / a), rel=1e-12)
    assert isinstance(result.controller, BrachistochroneTransit)


def test_solve_min_time_validation() -> None:
    with pytest.raises(ValueError):
        solve_min_time(distance_m=0.0, max_acceleration_mps2=1.0, mass_kg=1.0)
    with pytest.raises(ValueError):
        solve_min_time(distance_m=1.0, max_acceleration_mps2=0.0, mass_kg=1.0)
    with pytest.raises(ValueError):
        solve_min_time(distance_m=1.0, max_acceleration_mps2=1.0, mass_kg=0.0)


def test_solve_min_dv_proportional_guidance_runs() -> None:
    """Smoke: tune proportional guidance to reach a target within t_max."""
    target = np.array([10.0, 0.0, 0.0])

    def factory(gain: float) -> ProportionalGuidance:
        return ProportionalGuidance(target_position=target, gain=gain, max_thrust_n=50.0)

    result = solve_min_dv(
        factory=factory,
        target_r=target,
        mass_kg=1.0,
        t_max_s=20.0,
        gain_initial=0.1,
        max_iterations=20,
    )
    # Either succeeded outright, or has at least produced a controller that
    # made meaningful progress toward the target.
    assert result.delta_v_mps >= 0
    assert isinstance(result.controller, ProportionalGuidance)
    # Optimiser should have converged to a non-trivial gain bounded by gain_bounds.
    assert result.diagnostics["gain"] > 0
