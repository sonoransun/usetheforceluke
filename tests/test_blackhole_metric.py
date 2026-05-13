"""Schwarzschild gravity: protocol, Newtonian limit, horizon guard, GR correction."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce._schwarzschild import schwarzschild_radius
from usetheforce.blackhole import SchwarzschildGravity
from usetheforce.symbolic.schwarzschild import (
    newtonian_force_radial_lambdified,
    newtonian_potential_lambdified,
    schwarzschild_radius_lambdified,
)

M_SUN = 1.98892e30
G = sc.G
C = sc.c


def _make(
    mass_kg: float = M_SUN,
    probe_mass_kg: float = 1.0,
    use_gr_hover_correction: bool = False,
    horizon_softening_m: float = 0.0,
) -> SchwarzschildGravity:
    return SchwarzschildGravity(
        mass_kg=mass_kg,
        probe_mass_kg=probe_mass_kg,
        use_gr_hover_correction=use_gr_hover_correction,
        horizon_softening_m=horizon_softening_m,
    )


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    r_s = ff.schwarzschild_radius_m
    f = ff.force(0.0, np.array([10 * r_s, 0.0, 0.0]))
    assert f.shape == (3,) and np.all(np.isfinite(f))
    assert ff.metadata["speculative"] is False
    assert ff.metadata["avenue"] == "blackhole"
    assert ff.metadata["speculative_components"] == []
    assert ff.metadata["applicable_for_trajectory"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        SchwarzschildGravity(mass_kg=0.0, probe_mass_kg=1.0)
    with pytest.raises(ValueError):
        SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=-1.0)
    with pytest.raises(ValueError):
        SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=1.0, horizon_softening_m=-1.0)
    with pytest.raises(ValueError):
        SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=1.0, center=(0.0, 0.0))


def test_schwarzschild_radius_matches_symbolic() -> None:
    """r_s = 2 G M / c² (anchored). Cross-check against the SymPy form."""
    rs_lambd = schwarzschild_radius_lambdified()
    for m_kg in [M_SUN, 10 * M_SUN, 1e6 * M_SUN]:
        ff = _make(mass_kg=m_kg)
        assert ff.schwarzschild_radius_m == pytest.approx(rs_lambd(m_kg, G, C), rel=1e-12)
        # And the helper agrees too.
        assert schwarzschild_radius(m_kg) == pytest.approx(2.0 * G * m_kg / C / C, rel=1e-15)


def test_force_attractive_toward_centre() -> None:
    ff = _make()
    r_s = ff.schwarzschild_radius_m
    f = ff.force(0.0, np.array([10 * r_s, 0.0, 0.0]))
    # Attractive: -x direction at +x position.
    assert f[0] < 0
    assert f[1] == 0.0 and f[2] == 0.0


def test_newtonian_limit_matches_minus_GMm_over_r2() -> None:
    """Far from the horizon (R = 1e6 r_s) the force must recover the Newtonian limit."""
    m_probe = 2.5
    ff = _make(mass_kg=M_SUN, probe_mass_kg=m_probe)
    r_s = ff.schwarzschild_radius_m
    R = 1e6 * r_s
    f = ff.force(0.0, np.array([R, 0.0, 0.0]))
    expected_mag = G * M_SUN * m_probe / (R * R)
    assert f[0] == pytest.approx(-expected_mag, rel=1e-10)


def test_force_matches_symbolic_radial() -> None:
    m_probe = 2.5
    ff = _make(mass_kg=M_SUN, probe_mass_kg=m_probe)
    r_s = ff.schwarzschild_radius_m
    f_lambd = newtonian_force_radial_lambdified()
    for factor in (5.0, 100.0, 1e6):
        R = factor * r_s
        f_x = ff.force(0.0, np.array([R, 0.0, 0.0]))[0]
        # Symbolic F_r = -dU/dr = -G M m / r² (attractive); on +x axis the
        # vector's x-component matches that sign.
        assert f_x == pytest.approx(f_lambd(R, M_SUN, m_probe, G), rel=1e-12)


def test_gr_hover_correction_at_2rs() -> None:
    """At R = 2 r_s the GR factor is 1/sqrt(1/2) = sqrt(2)."""
    ff_n = _make(use_gr_hover_correction=False)
    ff_gr = _make(use_gr_hover_correction=True)
    r_s = ff_n.schwarzschild_radius_m
    R = 2.0 * r_s
    f_n = float(np.linalg.norm(ff_n.force(0.0, np.array([R, 0.0, 0.0]))))
    f_gr = float(np.linalg.norm(ff_gr.force(0.0, np.array([R, 0.0, 0.0]))))
    assert f_gr / f_n == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_gr_hover_correction_diverges_near_horizon() -> None:
    """As R → r_s⁺, the GR-corrected force grows much faster than Newtonian."""
    ff_n = _make(use_gr_hover_correction=False)
    ff_gr = _make(use_gr_hover_correction=True)
    r_s = ff_n.schwarzschild_radius_m
    R = 1.0001 * r_s
    f_n = float(np.linalg.norm(ff_n.force(0.0, np.array([R, 0.0, 0.0]))))
    f_gr = float(np.linalg.norm(ff_gr.force(0.0, np.array([R, 0.0, 0.0]))))
    # 1/sqrt(1 - 1/1.0001) = 1/sqrt(1e-4) = 100.
    assert f_gr / f_n == pytest.approx(100.0, rel=1e-3)


def test_horizon_guard_raises_inside() -> None:
    ff = _make(horizon_softening_m=1.0)
    r_s = ff.schwarzschild_radius_m
    with pytest.raises(ValueError):
        ff.force(0.0, np.array([r_s + 0.5, 0.0, 0.0]))  # inside softening
    with pytest.raises(ValueError):
        ff.force(0.0, np.array([0.5 * r_s, 0.0, 0.0]))  # inside horizon
    with pytest.raises(ValueError):
        ff.potential(np.array([r_s, 0.0, 0.0]))


def test_potential_matches_symbolic_newtonian() -> None:
    m_probe = 2.5
    ff = _make(mass_kg=M_SUN, probe_mass_kg=m_probe)
    r_s = ff.schwarzschild_radius_m
    u_lambd = newtonian_potential_lambdified()
    for factor in (5.0, 100.0, 1e6):
        R = factor * r_s
        assert ff.potential(np.array([R, 0.0, 0.0])) == pytest.approx(
            u_lambd(R, M_SUN, m_probe, G), rel=1e-12
        )


def test_force_is_negative_gradient_of_potential_newtonian_mode() -> None:
    """F = -∇U should hold in Newtonian mode (potential is conservative)."""
    ff = _make()
    r_s = ff.schwarzschild_radius_m
    r0 = np.array([5.0 * r_s, 0.3 * r_s, -0.2 * r_s])
    h = r_s * 1e-4
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, rtol=1e-6)


def test_off_axis_force_points_toward_centre() -> None:
    ff = _make()
    r_s = ff.schwarzschild_radius_m
    r = np.array([3.0 * r_s, 4.0 * r_s, 0.0])  # |r| = 5 r_s
    f = ff.force(0.0, r)
    # Force vector should be anti-parallel to r.
    f_hat = f / np.linalg.norm(f)
    r_hat = r / np.linalg.norm(r)
    np.testing.assert_allclose(f_hat, -r_hat, rtol=1e-12)
