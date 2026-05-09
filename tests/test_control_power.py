"""VehiclePowerState bookkeeping."""

from __future__ import annotations

import pytest

from usetheforce.control import VehiclePowerState


def test_initial_state() -> None:
    p = VehiclePowerState(initial_energy_j=100.0, instantaneous_power_w=10.0)
    assert p.remaining_energy_j == 100.0
    assert p.is_depleted() is False
    assert p.available_power() == 10.0


def test_consume_depletes_reserve() -> None:
    p = VehiclePowerState(initial_energy_j=100.0, instantaneous_power_w=10.0)
    p.consume(dt=5.0, power_w=10.0)  # 50 J
    assert p.elapsed_energy_j == 50.0
    assert p.remaining_energy_j == 50.0
    assert p.is_depleted() is False
    p.consume(dt=10.0, power_w=10.0)  # +100 J → fully depleted
    assert p.is_depleted() is True
    assert p.available_power() == 0.0


def test_reset_restores_reserve() -> None:
    p = VehiclePowerState(initial_energy_j=100.0, instantaneous_power_w=10.0)
    p.consume(dt=5.0, power_w=10.0)
    p.reset()
    assert p.elapsed_energy_j == 0.0
    assert p.remaining_energy_j == 100.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        VehiclePowerState(initial_energy_j=0.0, instantaneous_power_w=1.0)
    with pytest.raises(ValueError):
        VehiclePowerState(initial_energy_j=1.0, instantaneous_power_w=-1.0)
    with pytest.raises(ValueError):
        VehiclePowerState(initial_energy_j=1.0, instantaneous_power_w=1.0, elapsed_energy_j=-1.0)
    p = VehiclePowerState(initial_energy_j=1.0, instantaneous_power_w=1.0)
    with pytest.raises(ValueError):
        p.consume(dt=-1.0, power_w=1.0)
    with pytest.raises(ValueError):
        p.consume(dt=1.0, power_w=-1.0)
