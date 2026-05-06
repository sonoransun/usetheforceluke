"""Parallel-plate Casimir matches both the textbook formula and the SymPy derivation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c, hbar

from usetheforce import ureg
from usetheforce.casimir import ParallelPlateCasimir
from usetheforce.symbolic.casimir import casimir_pressure_lambdified


@pytest.mark.parametrize("a_nm", [10.0, 100.0, 1000.0])
def test_pressure_matches_textbook(a_nm: float) -> None:
    a_m = a_nm * 1e-9
    expected = -(np.pi**2) * hbar * c / (240.0 * a_m**4)
    pp = ParallelPlateCasimir(area=1.0 * ureg.cm**2, separation=a_nm * ureg.nm)
    assert pp.pressure == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("a_nm", [10.0, 100.0, 1000.0])
def test_pressure_matches_symbolic(a_nm: float) -> None:
    a_m = a_nm * 1e-9
    pp = ParallelPlateCasimir(area=1.0 * ureg.cm**2, separation=a_nm * ureg.nm)
    f = casimir_pressure_lambdified()
    assert pp.pressure == pytest.approx(f(a_m, hbar, c), rel=1e-12)


def test_force_direction_along_axis() -> None:
    pp = ParallelPlateCasimir(
        area=1.0 * ureg.cm**2,
        separation=100.0 * ureg.nm,
        axis=(0.0, 0.0, 1.0),
    )
    f = pp.force(0.0, np.zeros(3))
    assert f[0] == 0.0 and f[1] == 0.0
    assert f[2] < 0.0  # attractive on the +z plate


def test_potential_independent_of_position() -> None:
    pp = ParallelPlateCasimir(area=1.0 * ureg.cm**2, separation=100.0 * ureg.nm)
    u1 = pp.potential(np.zeros(3))
    u2 = pp.potential(np.array([1.0, 2.0, 3.0]))
    assert u1 == u2 and u1 is not None and u1 < 0
