"""Antimatter graviton field: protocol, λ→∞ inverse-square limit, energy conservation."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.symbolic.graviton import (
    graviton_force_radial_lambdified,
    graviton_potential_lambdified,
)
from usetheforce.trajectories import integrate


def _make(lam: float = 10.0, **kwargs) -> AntimatterGravitonField:
    defaults: dict = {
        "source": (0.0, 0.0, 0.0),
        "gamma": 1.0,
        "coupling": 1.0,
        "screening": lam,
        "probe_mass": 1.0,
    }
    defaults.update(kwargs)
    return AntimatterGravitonField(**defaults)


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array([1.0, 0.0, 0.0]))
    assert f.shape == (3,) and np.all(np.isfinite(f))
    assert ff.metadata["speculative"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        AntimatterGravitonField(
            source=(0.0, 0.0), gamma=1.0, coupling=1.0, screening=1.0, probe_mass=1.0
        )
    with pytest.raises(ValueError):
        _make(gamma=-1.0)
    with pytest.raises(ValueError):
        _make(coupling=0.0)
    with pytest.raises(ValueError):
        _make(screening=-2.0)
    with pytest.raises(ValueError):
        _make(probe_mass=0.0)


def test_force_attractive_toward_source() -> None:
    ff = _make()
    f = ff.force(0.0, np.array([2.0, 0.0, 0.0]))
    # Force should point toward the source at the origin (i.e. -x).
    assert f[0] < 0
    assert f[1] == 0.0 and f[2] == 0.0


@pytest.mark.parametrize("dist", [0.5, 1.0, 3.0])
def test_potential_matches_symbolic(dist: float) -> None:
    g_val, gamma_val, lam_val, m_val = 0.7, 1.3, 5.0, 2.5
    ff = _make(coupling=g_val, gamma=gamma_val, screening=lam_val, probe_mass=m_val)
    phi = graviton_potential_lambdified()
    # potential() returns m_probe · φ.
    assert ff.potential(np.array([dist, 0.0, 0.0])) == pytest.approx(
        m_val * phi(dist, g_val, gamma_val, lam_val), rel=1e-12
    )


@pytest.mark.parametrize("dist", [0.5, 1.0, 3.0])
def test_radial_force_matches_symbolic(dist: float) -> None:
    g_val, gamma_val, lam_val, m_val = 0.7, 1.3, 5.0, 2.5
    ff = _make(coupling=g_val, gamma=gamma_val, screening=lam_val, probe_mass=m_val)
    f_sym = graviton_force_radial_lambdified()
    f_x = ff.force(0.0, np.array([dist, 0.0, 0.0]))[0]
    assert f_x == pytest.approx(f_sym(dist, g_val, gamma_val, lam_val, m_val), rel=1e-12)


def test_large_lambda_recovers_inverse_square() -> None:
    """λ ≫ R ⇒ F ≈ -m g Γ / R² along -r̂."""
    g_val, gamma_val, m_val = 1.0, 1.0, 1.0
    ff = _make(coupling=g_val, gamma=gamma_val, screening=1e9, probe_mass=m_val)
    for d in [0.5, 1.0, 3.0]:
        f = ff.force(0.0, np.array([d, 0.0, 0.0]))
        expected = np.array([-m_val * g_val * gamma_val / d**2, 0.0, 0.0])
        np.testing.assert_allclose(f, expected, rtol=1e-6)


def test_force_is_negative_gradient_of_potential() -> None:
    ff = _make(coupling=0.7, gamma=1.3, screening=5.0, probe_mass=2.5)
    r0 = np.array([1.5, -0.7, 0.4])
    h = 1e-6
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, atol=1e-7)


def test_energy_conservation_circular_limit() -> None:
    """Integrate a near-circular orbit in the λ→∞ limit and check ΔE/E."""
    g_val, gamma_val, m_val = 1.0, 1.0, 1.0
    ff = _make(coupling=g_val, gamma=gamma_val, screening=1e9, probe_mass=m_val)
    # In this limit |F| = m g Γ / r² ⇒ for circular orbit at r=1: v² = g Γ / r.
    v_circ = float(np.sqrt(g_val * gamma_val / 1.0))
    period = 2.0 * np.pi  # since gΓ=r=1
    traj = integrate(
        ff,
        mass=m_val,
        r0=[1.0, 0.0, 0.0],
        v0=[0.0, v_circ, 0.0],
        t_span=(0.0, period),
        n_eval=300,
    )
    energy = traj.total_energy(ff)
    drift = float(np.std(energy) / abs(np.mean(energy)))
    assert drift < 1e-6, f"energy drift {drift} too large"
