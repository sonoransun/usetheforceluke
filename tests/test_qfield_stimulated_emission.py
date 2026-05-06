"""Stimulated-emission array: protocol, single-emitter limit, destructive null, gradient consistency."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.qfield import StimulatedEmissionArray
from usetheforce.symbolic.stimulated_emission import (
    single_emitter_force_radial_lambdified,
    single_emitter_intensity_lambdified,
)


def test_protocol() -> None:
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0]],
        amplitudes=[1.0],
        phases=[0.0],
        wavenumber=2.0,
    )
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array([1.0, 0.0, 0.0]))
    assert f.shape == (3,) and np.all(np.isfinite(f))
    assert ff.metadata["speculative"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        StimulatedEmissionArray(
            positions=[[0.0, 0.0]], amplitudes=[1.0], phases=[0.0], wavenumber=1.0
        )
    with pytest.raises(ValueError):
        StimulatedEmissionArray(
            positions=[[0.0, 0.0, 0.0]],
            amplitudes=[1.0, 2.0],
            phases=[0.0],
            wavenumber=1.0,
        )
    with pytest.raises(ValueError):
        StimulatedEmissionArray(
            positions=[[0.0, 0.0, 0.0]], amplitudes=[1.0], phases=[0.0], wavenumber=0.0
        )


@pytest.mark.parametrize("dist", [0.5, 1.0, 3.0, 10.0])
def test_single_emitter_intensity_matches_symbolic(dist: float) -> None:
    A_val = 1.7
    alpha_val = 2.3
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0]],
        amplitudes=[A_val],
        phases=[0.0],
        wavenumber=4.0,
        coupling=alpha_val,
    )
    r_vec = np.array([dist, 0.0, 0.0])
    I_sym = single_emitter_intensity_lambdified()
    # U = -α I  ⇒  -U/α = I
    assert -ff.potential(r_vec) / alpha_val == pytest.approx(I_sym(dist, A_val), rel=1e-12)


@pytest.mark.parametrize("dist", [0.7, 1.5, 4.0])
def test_single_emitter_force_matches_symbolic(dist: float) -> None:
    A_val = 1.2
    alpha_val = 0.8
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0]],
        amplitudes=[A_val],
        phases=[0.0],
        wavenumber=4.0,
        coupling=alpha_val,
    )
    r_vec = np.array([dist, 0.0, 0.0])
    f_sym = single_emitter_force_radial_lambdified()
    f_x = ff.force(0.0, r_vec)[0]
    assert f_x == pytest.approx(f_sym(dist, A_val, alpha_val), rel=1e-12)


def test_destructive_interference_null() -> None:
    """Two equal-amplitude, π-out-of-phase emitters → zero intensity on the bisector."""
    sep = 1.0
    ff = StimulatedEmissionArray(
        positions=[[-sep / 2, 0.0, 0.0], [sep / 2, 0.0, 0.0]],
        amplitudes=[1.0, 1.0],
        phases=[0.0, np.pi],
        wavenumber=2.0,
    )
    # On the y axis (perpendicular bisector) the two amplitudes cancel exactly.
    for y in [0.5, 1.0, 2.0]:
        r_vec = np.array([0.0, y, 0.0])
        assert abs(ff.potential(r_vec)) < 1e-15
        # Force on the bisector is also zero by symmetry.
        np.testing.assert_allclose(ff.force(0.0, r_vec), 0.0, atol=1e-14)


def test_min_distance_guard() -> None:
    """A non-zero ``min_distance_m`` rejects probes too close to any emitter."""
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0]],
        amplitudes=[1.0],
        phases=[0.0],
        wavenumber=2.0,
        min_distance_m=0.1,
    )
    # Outside the floor — fine.
    assert np.all(np.isfinite(ff.force(0.0, np.array([0.2, 0.0, 0.0]))))
    # Inside the floor — raises.
    with pytest.raises(ValueError, match="min_distance"):
        ff.force(0.0, np.array([0.05, 0.0, 0.0]))


def test_force_is_negative_gradient_of_potential() -> None:
    """Numerical -∇U via central differences matches force()."""
    rng = np.random.default_rng(11)
    positions = rng.normal(size=(3, 3))
    amplitudes = rng.uniform(0.5, 2.0, size=3)
    phases = rng.uniform(0.0, 2 * np.pi, size=3)
    ff = StimulatedEmissionArray(
        positions=positions,
        amplitudes=amplitudes,
        phases=phases,
        wavenumber=1.5,
        coupling=0.7,
    )
    r0 = np.array([2.0, 1.5, -0.5])
    h = 1e-6
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, atol=1e-6)
