"""Symbolic energy expressions used by trajectory conservation tests."""

from __future__ import annotations

import sympy as sp

m_s, vx, vy, vz = sp.symbols("m v_x v_y v_z", real=True)
U = sp.symbols("U", real=True)

kinetic_energy_expr: sp.Expr = sp.Rational(1, 2) * m_s * (vx**2 + vy**2 + vz**2)
total_energy_expr: sp.Expr = kinetic_energy_expr + U
