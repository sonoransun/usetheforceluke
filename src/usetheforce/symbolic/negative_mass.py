"""Symbolic forms for negative-mass binaries — Bondi self-acceleration, anti-chirp,
quadrupole and dipole GW radiation.

Bondi 1957 zero-net-mass runaway pair: with ``|m_neg|`` the magnitude of the
negative-mass element and ``d`` the rigid separation,

    a = G · m_neg / d²        (constant body acceleration of the composite)

Peters–Mathews 1963 anti-chirp (using the signed chirp mass ``M_c`` so that a
negative ``M_c`` — Loeb's anti-chirp regime — produces a negative ``df/dt``,
i.e. orbital expansion):

    df/dt = (96/5) · π^(8/3) · (G · M_c / c³)^(5/3) · f^(11/3)

Quadrupole GW power for a positive-mass circular binary:

    P_quad = (32/5) · G⁴ / c⁵ · (m1·m2)² · (m1 + m2) / d⁵

Dipole GW power if the equivalence principle fails for the negative-mass
component (Loeb 2024):

    P_dip  = (2/3) · G / c³ · m_neg² · d² · ω⁴

References
----------
- Bondi, H. (1957). Rev. Mod. Phys. 29:423.
- Peters & Mathews (1963). Phys. Rev. 131:435.
- Loeb, A. (2024). Medium.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

# Positive-only magnitudes (we evaluate everything on absolute values; sign
# information is carried by the signed-chirp-mass symbol below).
d, m_neg, m1, m2, omega, G_s, c_s, f = sp.symbols(
    "d m_neg m1 m2 omega G c f", positive=True
)
# Signed chirp mass — must allow negative values for anti-chirp.
M_c = sp.symbols("M_c", real=True)

bondi_acceleration_expr: sp.Expr = G_s * m_neg / d**2

gw_quadrupole_power_expr: sp.Expr = (
    sp.Rational(32, 5) * G_s**4 / c_s**5 * (m1 * m2) ** 2 * (m1 + m2) / d**5
)

gw_dipole_power_expr: sp.Expr = (
    sp.Rational(2, 3) * G_s / c_s**3 * m_neg**2 * d**2 * omega**4
)

# Peters–Mathews df/dt with signed chirp mass. SymPy's ``Pow`` on a real
# (possibly negative) base + non-integer exponent stays symbolic; lambdified
# evaluation uses ``math.copysign`` on the cube/fifth-root identity:
#     M_c^(5/3) = sign(M_c) · |M_c|^(5/3)   (real-valued odd root)
# To keep the symbolic form clean we write it as ``sign(M_c) · Abs(M_c)^(5/3)``.
anti_chirp_dfdt_expr: sp.Expr = (
    sp.Rational(96, 5)
    * sp.pi ** sp.Rational(8, 3)
    * (G_s / c_s**3) ** sp.Rational(5, 3)
    * sp.sign(M_c)
    * sp.Abs(M_c) ** sp.Rational(5, 3)
    * f ** sp.Rational(11, 3)
)


def bondi_acceleration_lambdified() -> Callable[[float, float, float], float]:
    return sp.lambdify((m_neg, d, G_s), bondi_acceleration_expr, modules="math")


def gw_quadrupole_power_lambdified() -> Callable[[float, float, float, float, float], float]:
    return sp.lambdify((m1, m2, d, G_s, c_s), gw_quadrupole_power_expr, modules="math")


def gw_dipole_power_lambdified() -> Callable[[float, float, float, float, float], float]:
    return sp.lambdify((m_neg, d, omega, G_s, c_s), gw_dipole_power_expr, modules="math")


def anti_chirp_dfdt_lambdified() -> Callable[[float, float, float, float], float]:
    return sp.lambdify((M_c, f, G_s, c_s), anti_chirp_dfdt_expr, modules="math")
