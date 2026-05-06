"""Symbolic single-site softened-Coulomb expression. SPECULATIVE model.

Per-site potential and radial force used by ``qfield.HeavyElementLattice``:

    U(r) = -κ μ / sqrt(r² + ε²)
    F_r(r) = -dU/dr = -κ μ r / (r² + ε²)^(3/2)
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

r, kappa, mu, epsilon = sp.symbols("r kappa mu epsilon", positive=True)

heavy_element_potential_expr: sp.Expr = -kappa * mu / sp.sqrt(r**2 + epsilon**2)
heavy_element_force_radial_expr: sp.Expr = -sp.diff(heavy_element_potential_expr, r)


def heavy_element_potential_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, kappa, mu, epsilon), heavy_element_potential_expr, modules="math")


def heavy_element_force_radial_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, kappa, mu, epsilon), heavy_element_force_radial_expr, modules="math")
