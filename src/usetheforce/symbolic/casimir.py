"""Symbolic Casimir derivation. Source of truth for ``casimir.parallel_plate``.

The standard regularized result for the energy per unit area between two
perfectly-conducting parallel plates separated by ``a`` is

    u(a) = -π² ℏ c / (720 a³)

and the pressure is the negative derivative with respect to ``a``:

    P(a) = -du/da = -π² ℏ c / (240 a⁴)

We expose both the symbolic expression and a lambdified callable. Tests in
``tests/test_casimir_parallel_plate.py`` cross-check the numerical
implementation against this expression.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

a, hbar_s, c_s = sp.symbols("a hbar c", positive=True)

# Energy per unit area between the plates.
casimir_energy_per_area_expr: sp.Expr = -(sp.pi**2) * hbar_s * c_s / (720 * a**3)

# Pressure (negative = attractive).
casimir_pressure_expr: sp.Expr = -sp.diff(casimir_energy_per_area_expr, a)


def casimir_pressure_lambdified() -> Callable[[float, float, float], float]:
    """Return a numerical ``f(a, hbar, c) -> pressure`` callable."""
    return sp.lambdify((a, hbar_s, c_s), casimir_pressure_expr, modules="math")
