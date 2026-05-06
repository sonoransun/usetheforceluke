"""Vehicle catalogue invariants."""

from __future__ import annotations

import itertools

import pytest

from usetheforce.missions import VEHICLES, Vehicle, power_budget


def test_six_named_vehicles() -> None:
    expected = {
        "cubesat_6u",
        "smallsat",
        "crewed",
        "interplanetary",
        "generation_ship",
        "city_ship",
    }
    assert set(VEHICLES.keys()) == expected


def test_mass_strictly_increasing() -> None:
    masses = [v.mass_kg for v in VEHICLES.values()]
    assert masses == sorted(masses)
    assert all(m_next > m_prev for m_prev, m_next in itertools.pairwise(masses))


def test_power_strictly_increasing() -> None:
    powers = [v.power_w for v in VEHICLES.values()]
    assert powers == sorted(powers)
    assert all(p_next > p_prev for p_prev, p_next in itertools.pairwise(powers))


def test_power_budget_sublinear_in_mass() -> None:
    """A 1000× heavier ship should have <1000× the power (sub-linear scaling)."""
    p_small = power_budget(12.0)
    p_big = power_budget(12.0 * 1000.0)
    assert p_big / p_small < 1000.0
    assert p_big / p_small > 1.0  # but it should still grow


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        Vehicle(key="x", description="x", mass_kg=0.0, power_w=1.0)
    with pytest.raises(ValueError):
        Vehicle(key="x", description="x", mass_kg=1.0, power_w=0.0)
    with pytest.raises(ValueError):
        power_budget(-1.0)
