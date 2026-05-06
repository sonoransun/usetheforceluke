"""Symbolic Yukawa graviton potential and radial force. SPECULATIVE model.

Yukawa potential:

    φ(r) = -g Γ exp(-r/λ) / r

Radial component of the force on a probe of mass ``m``:

    F_r(r) = -m dφ/dr = -m · g Γ exp(-r/λ) (r + λ) / (λ r²)

(negative ⇒ attractive). In the limit ``λ → ∞`` this reduces to ``-m g Γ / r²``.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

r, g, gamma, lam, m = sp.symbols("r g gamma lambda m", positive=True)

graviton_potential_expr: sp.Expr = -g * gamma * sp.exp(-r / lam) / r
graviton_force_radial_expr: sp.Expr = -m * sp.diff(graviton_potential_expr, r)


def graviton_potential_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((r, g, gamma, lam), graviton_potential_expr, modules="math")


def graviton_force_radial_lambdified() -> Callable[[float, float, float, float, float], float]:
    return sp.lambdify((r, g, gamma, lam, m), graviton_force_radial_expr, modules="math")
