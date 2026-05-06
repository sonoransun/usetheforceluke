"""Symbolic QGP energy density and effective-degrees-of-freedom crossover.

Anchored expressions (textbook QGP thermodynamics):

    g_eff(T) = (g_HRG + g_QGP)/2 + (g_QGP − g_HRG)/2 · tanh((T − T_c) / ΔT)
    ε_SB(T)  = (π² / 30) · g_eff · (k_B T)⁴ / (ℏ c)³

Tests cross-check the numerical kernels in ``usetheforce.qgp.source`` against
the lambdified versions of these expressions.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

T, k_B_s, hbar_s, c_s, g_eff_sym = sp.symbols("T k_B hbar c g_eff", positive=True)
T_c, dT, g_HRG_s, g_QGP_s = sp.symbols("T_c Delta_T g_HRG g_QGP", positive=True)

g_effective_expr: sp.Expr = (g_HRG_s + g_QGP_s) / 2 + (g_QGP_s - g_HRG_s) / 2 * sp.tanh(
    (T - T_c) / dT
)

sb_energy_density_expr: sp.Expr = (
    sp.Rational(1, 30) * sp.pi**2 * g_eff_sym * (k_B_s * T) ** 4 / (hbar_s * c_s) ** 3
)


def sb_energy_density_lambdified() -> Callable[[float, float, float, float, float], float]:
    """Return a numerical ``f(T, g_eff, k_B, hbar, c) -> ε`` callable."""
    return sp.lambdify(
        (T, g_eff_sym, k_B_s, hbar_s, c_s),
        sb_energy_density_expr,
        modules="math",
    )


def g_effective_lambdified() -> Callable[[float, float, float, float, float], float]:
    """Return a numerical ``f(T, T_c, ΔT, g_HRG, g_QGP) -> g_eff`` callable."""
    return sp.lambdify(
        (T, T_c, dT, g_HRG_s, g_QGP_s),
        g_effective_expr,
        modules="math",
    )
