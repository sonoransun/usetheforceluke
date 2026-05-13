"""Symbolic Schwarzschild geometry: radius, Newtonian limit, GR hover factor.

Schwarzschild radius (anchored physics):

    r_s = 2 G M / c²

Newtonian-limit gravitational potential energy of a probe of mass ``m`` at
radius ``r`` from a body of mass ``M``:

    U(r) = -G M m / r,    F_r(r) = -dU/dr = -G M m / r²   (attractive)

GR proper-acceleration hover correction for a stationary observer at radius
``r > r_s``:

    a_hover / a_Newton = 1 / sqrt(1 - r_s / r) = 1 / sqrt(1 - 2 G M / (c² r))

In the limit ``r ≫ r_s`` this reduces to 1; as ``r → r_s⁺`` it diverges,
which is the central finding of the "blackhole explorer" mode.

References
----------
- Schwarzschild (1916); Misner, Thorne, Wheeler *Gravitation* §31.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

r, M, m, G_s, c_s = sp.symbols("r M m G c", positive=True)

schwarzschild_radius_expr: sp.Expr = 2 * G_s * M / c_s**2

newtonian_potential_expr: sp.Expr = -G_s * M * m / r

newtonian_force_radial_expr: sp.Expr = -sp.diff(newtonian_potential_expr, r)

gr_hover_factor_expr: sp.Expr = 1 / sp.sqrt(1 - 2 * G_s * M / (c_s**2 * r))


def schwarzschild_radius_lambdified() -> Callable[[float, float, float], float]:
    return sp.lambdify((M, G_s, c_s), schwarzschild_radius_expr, modules="math")


def newtonian_potential_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, M, m, G_s), newtonian_potential_expr, modules="math")


def newtonian_force_radial_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, M, m, G_s), newtonian_force_radial_expr, modules="math")


def gr_hover_factor_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, M, G_s, c_s), gr_hover_factor_expr, modules="math")
