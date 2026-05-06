"""Trajectory planning recovers an analytical projectile launch velocity."""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce.trajectories import delta_v_for_target


class ConstantGravity:
    metadata: dict[str, Any] = {"avenue": "test", "model": "constant g", "speculative": False}

    def __init__(self, g: float, mass: float) -> None:
        self._g = g
        self._m = mass

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, -self._m * self._g])

    def potential(self, r: np.ndarray) -> float:
        return self._m * self._g * float(r[2])


def test_projectile_target_velocity() -> None:
    """For uniform g, hitting (X, 0, 0) at time T from the origin needs v = (X/T, 0, gT/2)."""
    g = 9.81
    mass = 1.0
    t_flight = 2.0
    target = np.array([10.0, 0.0, 0.0])
    expected_v = np.array([target[0] / t_flight, 0.0, g * t_flight / 2])
    ff = ConstantGravity(g=g, mass=mass)
    delta_v = delta_v_for_target(
        ff,
        mass=mass,
        r0=[0.0, 0.0, 0.0],
        v0_guess=[0.0, 0.0, 0.0],
        r_target=target,
        t_flight=t_flight,
    )
    np.testing.assert_allclose(delta_v, expected_v, atol=1e-6)
