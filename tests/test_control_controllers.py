"""Per-controller behavior."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce.control import (
    BangBangAltitude,
    BrachistochroneTransit,
    ConstantAcceleration,
    ConstantThrust,
    ProportionalGuidance,
    ScheduledThrust,
    ThrustController,
    VehiclePowerState,
)


@pytest.fixture
def power() -> VehiclePowerState:
    return VehiclePowerState(initial_energy_j=1e9, instantaneous_power_w=1e6)


def test_constant_thrust_protocol(power: VehiclePowerState) -> None:
    c = ConstantThrust(magnitude_n=10.0, direction=(1.0, 0.0, 0.0))
    assert isinstance(c, ThrustController)
    f = c.thrust(0.0, np.zeros(3), np.zeros(3), power)
    np.testing.assert_array_equal(f, np.array([10.0, 0.0, 0.0]))
    assert c.metadata["closed_loop"] is False


def test_constant_thrust_validates() -> None:
    with pytest.raises(ValueError):
        ConstantThrust(magnitude_n=-1.0, direction=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ConstantThrust(magnitude_n=1.0, direction=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ConstantThrust(magnitude_n=1.0, direction=(1.0, 0.0))  # bad shape


def test_scheduled_thrust(power: VehiclePowerState) -> None:
    def profile(t: float) -> tuple[float, np.ndarray]:
        # Ramp from 0 to 100 N along x over t in [0, 10].
        return min(100.0, 10.0 * t), np.array([1.0, 0.0, 0.0])

    c = ScheduledThrust(profile=profile)
    np.testing.assert_array_almost_equal(
        c.thrust(0.0, np.zeros(3), np.zeros(3), power), np.zeros(3)
    )
    np.testing.assert_array_almost_equal(
        c.thrust(5.0, np.zeros(3), np.zeros(3), power), np.array([50.0, 0.0, 0.0])
    )
    np.testing.assert_array_almost_equal(
        c.thrust(20.0, np.zeros(3), np.zeros(3), power), np.array([100.0, 0.0, 0.0])
    )


def test_constant_acceleration_holds_force(power: VehiclePowerState) -> None:
    c = ConstantAcceleration(a_target_mps2=2.0, mass_kg=5.0, direction_policy="velocity")
    v = np.array([0.0, 1.0, 0.0])
    f = c.thrust(0.0, np.zeros(3), v, power)
    # |F| should be m·a = 10 N along velocity.
    assert float(np.linalg.norm(f)) == pytest.approx(10.0, rel=1e-12)
    np.testing.assert_array_almost_equal(f / 10.0, np.array([0.0, 1.0, 0.0]))


def test_constant_acceleration_validates() -> None:
    with pytest.raises(ValueError):
        ConstantAcceleration(a_target_mps2=0.0, mass_kg=1.0)
    with pytest.raises(ValueError):
        ConstantAcceleration(a_target_mps2=1.0, mass_kg=0.0)
    with pytest.raises(ValueError):
        ConstantAcceleration(a_target_mps2=1.0, mass_kg=1.0, direction_policy="bogus")
    with pytest.raises(ValueError):
        ConstantAcceleration(a_target_mps2=1.0, mass_kg=1.0, direction_policy="toward_target")


def test_brachistochrone_analytical_time() -> None:
    """For d=400 m and a_max=2 m/s², travel time = 2·√(400/2) = 2·√200 ≈ 28.28 s."""
    c = BrachistochroneTransit(
        r_start=(0.0, 0.0, 0.0),
        r_target=(400.0, 0.0, 0.0),
        max_acceleration_mps2=2.0,
        mass_kg=1.0,
    )
    expected = 2.0 * np.sqrt(400.0 / 2.0)
    assert c.analytical_travel_time_s == pytest.approx(expected, rel=1e-12)
    assert c.t_brake_s == pytest.approx(expected / 2.0, rel=1e-12)


def test_brachistochrone_thrust_direction(power: VehiclePowerState) -> None:
    c = BrachistochroneTransit(
        r_start=(0.0, 0.0, 0.0),
        r_target=(100.0, 0.0, 0.0),
        max_acceleration_mps2=2.0,
        mass_kg=3.0,
    )
    # Before brake: thrust points along start→target axis.
    f1 = c.thrust(c.t_brake_s * 0.5, np.zeros(3), np.array([1.0, 0.0, 0.0]), power)
    np.testing.assert_array_almost_equal(f1 / np.linalg.norm(f1), np.array([1.0, 0.0, 0.0]))
    # After brake: anti-parallel to current velocity.
    f2 = c.thrust(c.t_brake_s * 1.5, np.zeros(3), np.array([1.0, 0.0, 0.0]), power)
    np.testing.assert_array_almost_equal(f2 / np.linalg.norm(f2), np.array([-1.0, 0.0, 0.0]))


def test_proportional_guidance_zero_distance(power: VehiclePowerState) -> None:
    target = np.array([10.0, 0.0, 0.0])
    c = ProportionalGuidance(target_position=target, gain=1.0, max_thrust_n=100.0)
    np.testing.assert_array_equal(c.thrust(0.0, target, np.zeros(3), power), np.zeros(3))


def test_proportional_guidance_capped(power: VehiclePowerState) -> None:
    c = ProportionalGuidance(target_position=(1000.0, 0.0, 0.0), gain=1.0, max_thrust_n=100.0)
    # Distance is 1000 m, gain=1 → uncapped magnitude would be 1000 N.
    f = c.thrust(0.0, np.zeros(3), np.zeros(3), power)
    assert float(np.linalg.norm(f)) == pytest.approx(100.0, rel=1e-12)


def test_bang_bang_altitude_hysteresis(power: VehiclePowerState) -> None:
    c = BangBangAltitude(target_altitude_m=100.0, threshold_m=5.0, magnitude_n=50.0)
    # Below band → on.
    f1 = c.thrust(0.0, np.array([0.0, 0.0, 90.0]), np.zeros(3), power)
    assert float(np.linalg.norm(f1)) == pytest.approx(50.0)
    # In band → holds (still on).
    f2 = c.thrust(0.0, np.array([0.0, 0.0, 100.0]), np.zeros(3), power)
    assert float(np.linalg.norm(f2)) == pytest.approx(50.0)
    # Above band → off.
    f3 = c.thrust(0.0, np.array([0.0, 0.0, 110.0]), np.zeros(3), power)
    np.testing.assert_array_equal(f3, np.zeros(3))
    # In band again → holds (off).
    f4 = c.thrust(0.0, np.array([0.0, 0.0, 100.0]), np.zeros(3), power)
    np.testing.assert_array_equal(f4, np.zeros(3))


def test_bang_bang_validates() -> None:
    with pytest.raises(ValueError):
        BangBangAltitude(target_altitude_m=100.0, threshold_m=0.0, magnitude_n=1.0)
    with pytest.raises(ValueError):
        BangBangAltitude(target_altitude_m=100.0, threshold_m=1.0, magnitude_n=0.0)
