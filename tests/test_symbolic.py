"""SymPy expressions are valid and lambdify roundtrips."""

from __future__ import annotations

import sympy as sp
from scipy.constants import c, hbar

from usetheforce.symbolic import (
    casimir_pressure_expr,
    casimir_pressure_lambdified,
    kinetic_energy_expr,
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
