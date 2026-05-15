"""Dipole graviton radiator: cos θ pattern, WEP-violation premise, zero-α degeneracy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.negmass import DipoleGravitonRadiator


def _make(
    m_negative_kg: float = 1.0,
    separation_m: float = 1.0,
    wep_violation: float = 1.0,
    probe_mass_kg: float = 1.0,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> DipoleGravitonRadiator:
    return DipoleGravitonRadiator(
        m_negative_kg=m_negative_kg,
        separation_m=separation_m,
        wep_violation=wep_violation,
        probe_mass_kg=probe_mass_kg,
        axis=axis,
    )


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "negmass"
    expected_sc = {"m_negative_kg", "negative_mass_premise", "wep_violation_amplitude"}
    assert set(ff.metadata["speculative_components"]) == expected_sc
    assert ff.metadata["applicable_for_trajectory"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        _make(m_negative_kg=0.0)
    with pytest.raises(ValueError):
        _make(separation_m=-1.0)
    with pytest.raises(ValueError):
        _make(wep_violation=-0.1)
    with pytest.raises(ValueError):
        _make(wep_violation=1.5)
    with pytest.raises(ValueError):
        _make(probe_mass_kg=0.0)
    with pytest.raises(ValueError):
        _make(axis=(0.0, 0.0, 0.0))


def test_dipole_pattern_along_axis_is_max_off_axis_is_zero() -> None:
    """Dipole signature: force ∝ cos θ. Max at θ=0, zero at θ=π/2."""
    ff = _make(axis=(1.0, 0.0, 0.0))
    R = 10.0
    f_axis = ff.force(0.0, np.array([R, 0.0, 0.0]))
    f_perp = ff.force(0.0, np.array([0.0, R, 0.0]))
    # cos(0) = 1: full magnitude.
    assert float(np.linalg.norm(f_axis)) > 0
    # cos(π/2) = 0: zero force at perpendicular.
    np.testing.assert_allclose(f_perp, np.zeros(3), atol=1e-25)


def test_dipole_not_quadrupole_pattern() -> None:
    """At θ = π/2 dipole gives 0, quadrupole would give nonzero (sin² θ = 1).

    Sanity check that we aren't accidentally implementing the quadrupole
    pattern.
    """
    ff = _make(axis=(0.0, 0.0, 1.0))
    R = 5.0
    # θ = π/2: probe perpendicular to z-axis.
    f = ff.force(0.0, np.array([R, 0.0, 0.0]))
    np.testing.assert_allclose(f, np.zeros(3), atol=1e-25)


def test_angular_dependence_is_cosine() -> None:
    """Force magnitude at angle θ scales as |cos θ| (sign flip at θ > π/2)."""
    ff = _make()
    R = 5.0
    angles = [0.0, math.pi / 4, math.pi / 3, math.pi / 2]
    f_at_0 = ff.force(0.0, np.array([R, 0.0, 0.0]))
    mag_0 = float(np.linalg.norm(f_at_0))
    for theta in angles:
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        f = ff.force(0.0, np.array([x, y, 0.0]))
        mag = float(np.linalg.norm(f))
        expected = mag_0 * abs(math.cos(theta))
        assert mag == pytest.approx(expected, rel=1e-10, abs=1e-25)


def test_wep_violation_zero_recovers_zero_force() -> None:
    """α = 0 is the no-WEP-violation degenerate case; no dipole radiation."""
    ff = _make(wep_violation=0.0)
    for r in (np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])):
        np.testing.assert_allclose(ff.force(0.0, r), np.zeros(3))


def test_potential_is_none() -> None:
    ff = _make()
    assert ff.potential(np.array([1.0, 0.0, 0.0])) is None
