"""Symbolic single-emitter intensity. SPECULATIVE model.

For a single emitter of amplitude ``A`` at the origin radiating outward with
wavenumber ``k``, the time-averaged intensity at distance ``r`` is::

    I_single(r) = A² / r²

(independent of ``k``, since |exp(ikr)|² = 1). The radiation potential is
``U = -α I`` and the radial force is ``F_r = -dU/dr``.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

r, A, alpha = sp.symbols("r A alpha", positive=True)

single_emitter_intensity_expr: sp.Expr = A**2 / r**2
single_emitter_potential_expr: sp.Expr = -alpha * single_emitter_intensity_expr
single_emitter_force_radial_expr: sp.Expr = -sp.diff(single_emitter_potential_expr, r)


def single_emitter_intensity_lambdified() -> Callable[[float, float], float]:
    return sp.lambdify((r, A), single_emitter_intensity_expr, modules="math")


def single_emitter_force_radial_lambdified() -> Callable[[float, float, float], float]:
    return sp.lambdify((r, A, alpha), single_emitter_force_radial_expr, modules="math")
