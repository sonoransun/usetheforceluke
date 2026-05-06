"""Trajectory integrator recovers textbook Newtonian and Kepler limits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from usetheforce.trajectories import integrate


class ConstantGravity:
    metadata: dict[str, Any] = {"avenue": "test", "model": "constant g", "speculative": False}

    def __init__(self, g: float = 9.81, mass: float = 1.0) -> None:
        self._g = g
        self._m = mass

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, -self._m * self._g])

    def potential(self, r: np.ndarray) -> float:
        return self._m * self._g * float(r[2])


class InverseSquare:
    metadata: dict[str, Any] = {"avenue": "test", "model": "Kepler", "speculative": False}

    def __init__(self, gm: float, mass: float) -> None:
        self._gm = gm
        self._m = mass

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        rn = float(np.linalg.norm(r))
        return -self._gm * self._m * np.asarray(r, dtype=float) / rn**3

    def potential(self, r: np.ndarray) -> float:
        rn = float(np.linalg.norm(r))
        return -self._gm * self._m / rn


def test_drop_under_constant_gravity() -> None:
    mass = 2.0
    g = 9.81
    ff = ConstantGravity(g=g, mass=mass)
    t_end = 3.0
    traj = integrate(ff, mass, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, t_end), n_eval=50)
    z_expected = -0.5 * g * traj.t**2
    np.testing.assert_allclose(traj.r[:, 2], z_expected, atol=1e-7)
    np.testing.assert_allclose(traj.r[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj.r[:, 1], 0.0, atol=1e-12)


def test_circular_kepler_orbit_closes() -> None:
    mass = 1.0
    gm = 1.0
    radius = 1.0
    v_circ = float(np.sqrt(gm / radius))
    period = 2.0 * np.pi * np.sqrt(radius**3 / gm)
    ff = InverseSquare(gm=gm, mass=mass)
    traj = integrate(
        ff,
        mass,
        [radius, 0.0, 0.0],
        [0.0, v_circ, 0.0],
        (0.0, period),
        n_eval=400,
    )
    # End position should match start to high precision.
    err = np.linalg.norm(traj.r[-1] - traj.r[0])
    assert err < 1e-6, f"orbit closure error too large: {err}"
    # Energy should be conserved.
    energy = traj.total_energy(ff)
    drift = float(np.std(energy) / abs(np.mean(energy)))
    assert drift < 1e-8, f"energy drift {drift} too large"


class _NaNPotentialField:
    """A ForceField whose potential returns NaN — total_energy must reject it."""

    metadata: dict[str, Any] = {
        "avenue": "test",
        "model": "nan-potential",
        "speculative": False,
    }

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    def potential(self, r: np.ndarray) -> float:
        return float("nan")


def test_total_energy_rejects_nan_potential() -> None:
    ff = _NaNPotentialField()
    traj = integrate(ff, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 1.0), n_eval=10)
    with pytest.raises(ValueError, match="non-finite"):
        traj.total_energy(ff)


def test_total_energy_rejects_none_potential() -> None:
    """A field with no potential (returns None) must surface a clear error, not crash."""

    class _NoPotentialField:
        metadata: dict[str, Any] = {"avenue": "test", "model": "none-pot", "speculative": False}

        def force(self, t: float, r: np.ndarray) -> np.ndarray:
            return np.zeros(3)

        def potential(self, r: np.ndarray) -> None:
            return None

    ff = _NoPotentialField()
    traj = integrate(ff, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 1.0), n_eval=10)
    with pytest.raises(ValueError, match="does not provide a potential"):
        traj.total_energy(ff)


def test_integrate_rejects_bad_input() -> None:
    ff = ConstantGravity()
    with pytest.raises(ValueError):
        integrate(ff, 1.0, [0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 1.0))  # bad r0
    with pytest.raises(ValueError):
        integrate(ff, -1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 1.0))  # bad mass
