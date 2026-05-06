"""Heavy-element lattice: protocol, single-site limit, potential/force consistency."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.qfield import HeavyElementLattice
from usetheforce.symbolic.heavy_elements import (
    heavy_element_force_radial_lambdified,
    heavy_element_potential_lambdified,
)


def test_protocol() -> None:
    ff = HeavyElementLattice(sites=[[0.0, 0.0, 0.0]], strengths=[1.0], coupling=1.0, softening=0.1)
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array([1.0, 0.0, 0.0]))
    assert f.shape == (3,) and np.all(np.isfinite(f))
    assert ff.metadata["speculative"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        HeavyElementLattice(sites=[[0.0, 0.0]], strengths=[1.0])  # bad shape
    with pytest.raises(ValueError):
        HeavyElementLattice(sites=[[0.0, 0.0, 0.0]], strengths=[1.0, 2.0])  # mismatch
    with pytest.raises(ValueError):
        HeavyElementLattice(
            sites=[[0.0, 0.0, 0.0]], strengths=[1.0], softening=-0.1
        )  # bad softening


@pytest.mark.parametrize("dist", [0.5, 1.0, 2.0, 5.0])
def test_single_site_unsoftened_recovers_inverse_square(dist: float) -> None:
    """ε=0, μ=1, κ=1 ⇒ |F| = 1/r² along -r̂."""
    ff = HeavyElementLattice(sites=[[0.0, 0.0, 0.0]], strengths=[1.0], coupling=1.0, softening=0.0)
    r = np.array([dist, 0.0, 0.0])
    f = ff.force(0.0, r)
    expected = -np.array([1.0 / dist**2, 0.0, 0.0])
    np.testing.assert_allclose(f, expected, rtol=1e-12)


def test_potential_matches_symbolic() -> None:
    """Single-site numerical potential matches the SymPy expression."""
    kappa, mu, eps = 2.5, 1.5, 0.3
    ff = HeavyElementLattice(sites=[[0.0, 0.0, 0.0]], strengths=[mu], coupling=kappa, softening=eps)
    u_sym = heavy_element_potential_lambdified()
    for d in [0.4, 1.0, 2.5, 7.0]:
        r_vec = np.array([d, 0.0, 0.0])
        assert ff.potential(r_vec) == pytest.approx(u_sym(d, kappa, mu, eps), rel=1e-12)


def test_radial_force_matches_symbolic() -> None:
    kappa, mu, eps = 2.5, 1.5, 0.3
    ff = HeavyElementLattice(sites=[[0.0, 0.0, 0.0]], strengths=[mu], coupling=kappa, softening=eps)
    f_sym = heavy_element_force_radial_lambdified()
    for d in [0.4, 1.0, 2.5, 7.0]:
        r_vec = np.array([d, 0.0, 0.0])
        f_x = ff.force(0.0, r_vec)[0]
        # Symbolic gives the radial component along +r̂; here r̂=+x, so they agree in sign.
        assert f_x == pytest.approx(f_sym(d, kappa, mu, eps), rel=1e-12)


def test_unsoftened_lattice_at_site_raises() -> None:
    """With softening=0, calling force/potential exactly at a lattice site raises."""
    ff = HeavyElementLattice(sites=[[1.0, 0.0, 0.0]], strengths=[1.0], coupling=1.0, softening=0.0)
    with pytest.raises(ValueError, match="softening"):
        ff.force(0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="softening"):
        ff.potential(np.array([1.0, 0.0, 0.0]))


def test_softened_lattice_at_site_finite() -> None:
    """With softening > 0 the potential at a site is finite and well-defined."""
    ff = HeavyElementLattice(sites=[[1.0, 0.0, 0.0]], strengths=[1.0], coupling=1.0, softening=0.5)
    val = ff.potential(np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(val)


def test_force_is_negative_gradient_of_potential() -> None:
    """Numerical -∇U via central differences matches force()."""
    rng = np.random.default_rng(7)
    sites = rng.normal(size=(4, 3))
    strengths = rng.uniform(0.5, 2.0, size=4)
    ff = HeavyElementLattice(sites=sites, strengths=strengths, coupling=1.3, softening=0.2)
    r0 = np.array([0.5, -0.3, 0.7])
    h = 1e-6
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, atol=1e-7)
