"""ControlledThrustField composition behavior."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.control import (
    ConstantThrust,
    ControlledThrustField,
    VehiclePowerState,
)


def _free(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.zeros(3)


def _const_g(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.array([0.0, 0.0, -9.80665])


def test_field_satisfies_protocol() -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    field = ControlledThrustField(controller=c, mass_kg=1.0, background=_free)
    assert isinstance(field, ForceField)
    f = field.force(0.0, np.zeros(3))
    assert f.shape == (3,)


def test_field_thrust_plus_gravity() -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    field = ControlledThrustField(controller=c, mass_kg=2.0, background=_const_g)
    f = field.force(0.0, np.zeros(3))
    # Thrust = 10 along x, gravity = m·g = 2·-9.80665 along z.
    np.testing.assert_array_almost_equal(f, np.array([10.0, 0.0, -2.0 * 9.80665]))


def test_potential_returns_none() -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    field = ControlledThrustField(controller=c, mass_kg=1.0, background=_free)
    assert field.potential(np.zeros(3)) is None


def test_validates_input() -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ControlledThrustField(controller=c, mass_kg=0.0, background=_free)


def test_reset_clears_state() -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    power = VehiclePowerState(initial_energy_j=1e6, instantaneous_power_w=1e3)
    field = ControlledThrustField(controller=c, mass_kg=1.0, background=_free, power=power)
    # Drive a few force calls so internal state populates.
    field.force(0.0, np.zeros(3))
    field.force(1.0, np.array([1.0, 0.0, 0.0]))
    assert len(field.thrust_log) == 2
    field.reset()
    assert len(field.thrust_log) == 0
    assert field.power.elapsed_energy_j == 0.0


def test_thrust_log_records_calls() -> None:
    c = ConstantThrust(magnitude_n=5.0, direction=(0.0, 1.0, 0.0))
    field = ControlledThrustField(controller=c, mass_kg=1.0, background=_free)
    field.force(0.0, np.zeros(3))
    field.force(1.0, np.zeros(3))
    field.force(2.0, np.zeros(3))
    times = [entry[0] for entry in field.thrust_log]
    assert times == [0.0, 1.0, 2.0]
    # Each thrust vector should equal the configured constant.
    for _, F in field.thrust_log:
        np.testing.assert_array_almost_equal(F, np.array([0.0, 5.0, 0.0]))
