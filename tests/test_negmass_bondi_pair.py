"""Bondi runaway pair: constant body force, energy non-conservation is the headline.

Read the assertion in ``test_kinetic_energy_grows_under_integration`` carefully:
the Bondi premise *requires* energy non-conservation. The test is written to
FAIL LOUDLY if anyone silently zeroes the runaway in a later refactor. Do not
"fix" this test.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce.negmass import BondiRunawayPair
from usetheforce.symbolic.negative_mass import bondi_acceleration_lambdified
from usetheforce.trajectories import integrate

G = sc.G


def _make(
    m_negative_kg: float = 1.0,
    separation_m: float = 1.0,
    craft_mass_kg: float = 1.0,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> BondiRunawayPair:
    return BondiRunawayPair(
        m_negative_kg=m_negative_kg,
        separation_m=separation_m,
        craft_mass_kg=craft_mass_kg,
        axis=axis,
    )


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "negmass"
    assert "m_negative_kg" in ff.metadata["speculative_components"]
    assert "negative_mass_premise" in ff.metadata["speculative_components"]
    assert ff.metadata["applicable_for_trajectory"] is False


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        _make(m_negative_kg=0.0)
    with pytest.raises(ValueError):
        _make(separation_m=-1.0)
    with pytest.raises(ValueError):
        _make(craft_mass_kg=-2.0)
    with pytest.raises(ValueError):
        _make(axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        BondiRunawayPair(
            m_negative_kg=1.0,
            separation_m=1.0,
            craft_mass_kg=1.0,
            axis=(1.0, 0.0),  # type: ignore[arg-type]
        )


def test_force_is_constant_in_space_and_time() -> None:
    ff = _make()
    f0 = ff.force(0.0, np.zeros(3))
    f1 = ff.force(0.0, np.array([1e6, -3e2, 4.0]))
    f2 = ff.force(1e9, np.array([-1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(f0, f1)
    np.testing.assert_array_equal(f0, f2)


def test_acceleration_matches_symbolic() -> None:
    a_lambd = bondi_acceleration_lambdified()
    for m_neg in (1.0, 1e6, 1e12):
        for d in (0.1, 1.0, 100.0):
            ff = _make(m_negative_kg=m_neg, separation_m=d, craft_mass_kg=42.0)
            assert ff.self_acceleration_mps2 == pytest.approx(
                a_lambd(m_neg, d, G), rel=1e-12
            )


def test_force_direction_along_axis() -> None:
    ff = _make(axis=(0.0, 1.0, 0.0))
    f = ff.force(0.0, np.zeros(3))
    assert f[1] > 0 and f[0] == 0.0 and f[2] == 0.0


def test_potential_is_none() -> None:
    ff = _make()
    assert ff.potential(np.zeros(3)) is None


def test_kinetic_energy_grows_under_integration() -> None:
    """Bondi pair injects KE from nothing — this is the load-bearing pathology.

    If this test passes with a *shrinking* ΔKE, someone has broken the Bondi
    premise. Restore the runaway; do not edit this assertion.
    """
    # Use a large m_neg so the acceleration is large enough to see in a 1 s
    # integration with default DOP853 tolerances (rtol=1e-10).
    m_neg = 1.0e20  # kg
    d = 1.0  # m
    craft_mass = 1.0  # kg
    ff = _make(m_negative_kg=m_neg, separation_m=d, craft_mass_kg=craft_mass)
    duration = 10.0
    traj = integrate(
        ff,
        mass=craft_mass,
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        t_span=(0.0, duration),
        n_eval=100,
    )
    ke = 0.5 * craft_mass * np.sum(traj.v * traj.v, axis=1)
    # Monotonic growth from zero (constant +x thrust starting at rest).
    assert ke[0] == pytest.approx(0.0, abs=1e-30)
    assert ke[-1] > 0.0
    # Strict monotonic growth (no dips from integrator wobble; constant body force).
    diffs = np.diff(ke)
    assert np.all(diffs >= -1e-10), "Bondi runaway KE must grow monotonically"
    # Analytical check: v(T) = a · T, KE(T) = 0.5 · m · a² · T².
    a = G * m_neg / (d * d)
    expected_ke = 0.5 * craft_mass * (a * duration) ** 2
    assert ke[-1] == pytest.approx(expected_ke, rel=1e-6)
