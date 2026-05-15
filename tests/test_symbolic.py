"""SymPy expressions are valid and lambdify roundtrips."""

from __future__ import annotations

import math

import pytest
import sympy as sp
from scipy.constants import G as G_CONST
from scipy.constants import c, hbar

from usetheforce.symbolic import (
    anti_chirp_dfdt_lambdified,
    bondi_acceleration_expr,
    bondi_acceleration_lambdified,
    casimir_pressure_expr,
    casimir_pressure_lambdified,
    gr_hover_factor_lambdified,
    gw_dipole_power_lambdified,
    gw_quadrupole_power_lambdified,
    kinetic_energy_expr,
    newtonian_force_radial_lambdified,
    sb_energy_density_expr,
    schwarzschild_radius_expr,
    schwarzschild_radius_lambdified,
    total_energy_expr,
)


def test_casimir_expr_form() -> None:
    """Symbolic pressure should be -π² ℏ c / (240 a⁴)."""
    a, hbar_s, c_s = sp.symbols("a hbar c", positive=True)
    expected = -(sp.pi**2) * hbar_s * c_s / (240 * a**4)
    assert sp.simplify(casimir_pressure_expr - expected) == 0


def test_casimir_lambdified_evaluates() -> None:
    f = casimir_pressure_lambdified()
    val = f(1e-7, hbar, c)
    assert val < 0
    assert abs(val) > 0


def test_kinetic_energy_expr() -> None:
    m, vx, vy, vz = sp.symbols("m v_x v_y v_z", real=True)
    expected = sp.Rational(1, 2) * m * (vx**2 + vy**2 + vz**2)
    assert sp.simplify(kinetic_energy_expr - expected) == 0


def test_total_energy_includes_potential() -> None:
    U = sp.symbols("U", real=True)
    diff = sp.simplify(total_energy_expr - kinetic_energy_expr - U)
    assert diff == 0


def test_qgp_stefan_boltzmann_form() -> None:
    """Symbolic ε is (π²/30) g_eff (k_B T)⁴ / (ℏc)³."""
    T_, g_, kB_, hb_, c_ = sp.symbols("T g_eff k_B hbar c", positive=True)
    expected = sp.Rational(1, 30) * sp.pi**2 * g_ * (kB_ * T_) ** 4 / (hb_ * c_) ** 3
    assert sp.simplify(sb_energy_density_expr - expected) == 0


def test_schwarzschild_radius_expr_form() -> None:
    """Symbolic r_s should be 2 G M / c²."""
    M_, G_, c_ = sp.symbols("M G c", positive=True)
    expected = 2 * G_ * M_ / c_**2
    assert sp.simplify(schwarzschild_radius_expr - expected) == 0


def test_schwarzschild_lambdified_evaluates() -> None:
    rs = schwarzschild_radius_lambdified()
    M_sun = 1.98892e30
    val = rs(M_sun, G_CONST, c)
    # Solar Schwarzschild radius ≈ 2.95 km
    assert val == pytest.approx(2.95e3, rel=5e-3)


def test_newtonian_force_lambdified_attractive() -> None:
    """F_r = -G M m / r²; lambdified callable returns a negative number on +r."""
    f = newtonian_force_radial_lambdified()
    M_sun = 1.98892e30
    val = f(1e10, M_sun, 1.0, G_CONST)
    expected = -G_CONST * M_sun / (1e10**2)
    assert val == pytest.approx(expected, rel=1e-10)


def test_gr_hover_factor_diverges_near_horizon() -> None:
    """1/sqrt(1 - r_s/R) ≈ 1 at R ≫ r_s and grows as R → r_s."""
    f = gr_hover_factor_lambdified()
    M_sun = 1.98892e30
    rs = 2 * G_CONST * M_sun / c**2
    # Far: factor ≈ 1
    assert f(1e6 * rs, M_sun, G_CONST, c) == pytest.approx(1.0, rel=1e-6)
    # Close: factor ≈ 1/sqrt(1 - 1/2) = sqrt(2)
    assert f(2 * rs, M_sun, G_CONST, c) == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_bondi_acceleration_expr_form() -> None:
    """a = G · m_neg / d²."""
    d_s, m_s, G_s = sp.symbols("d m_neg G", positive=True)
    expected = G_s * m_s / d_s**2
    assert sp.simplify(bondi_acceleration_expr - expected) == 0


def test_bondi_acceleration_lambdified_matches_numeric() -> None:
    f = bondi_acceleration_lambdified()
    for m_neg in (1.0, 1e6, 1e12):
        for d in (0.1, 1.0, 100.0):
            assert f(m_neg, d, G_CONST) == pytest.approx(
                G_CONST * m_neg / d**2, rel=1e-12
            )


def test_anti_chirp_dfdt_is_negative_for_negative_chirp_mass() -> None:
    f = anti_chirp_dfdt_lambdified()
    # Negative M_c (anti-chirp) gives df/dt < 0.
    val_neg = f(-1.0, 100.0, G_CONST, c)
    assert val_neg < 0
    # Positive M_c (normal chirp) gives df/dt > 0 — sanity check on the sign factor.
    val_pos = f(1.0, 100.0, G_CONST, c)
    assert val_pos > 0
    # Magnitudes should match under sign flip.
    assert val_neg == pytest.approx(-val_pos, rel=1e-12)


def test_gw_quadrupole_power_lambdified_matches_formula() -> None:
    f = gw_quadrupole_power_lambdified()
    m1, m2, d = 1.0e3, 1.0e3, 1.0e6
    expected = (32 / 5) * G_CONST**4 / c**5 * (m1 * m2) ** 2 * (m1 + m2) / d**5
    assert f(m1, m2, d, G_CONST, c) == pytest.approx(expected, rel=1e-12)


def test_gw_dipole_power_lambdified_matches_formula() -> None:
    f = gw_dipole_power_lambdified()
    m_neg, d, omega = 1.0e6, 1.0e3, 10.0
    expected = (2 / 3) * G_CONST / c**3 * m_neg**2 * d**2 * omega**4
    assert f(m_neg, d, omega, G_CONST, c) == pytest.approx(expected, rel=1e-12)
