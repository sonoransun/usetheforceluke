"""QGP source: anchored thermodynamics + lattice-flavoured g_eff(T) crossover."""

from __future__ import annotations

import itertools
import math

import pytest
import scipy.constants as sc

from usetheforce.qgp.source import (
    DELTA_T_MEV,
    G_HRG,
    G_QGP,
    T_C_MEV,
    QuarkGluonPlasmaSource,
    g_effective,
)
from usetheforce.symbolic.qgp import (
    g_effective_lambdified,
    sb_energy_density_lambdified,
)

_MEV_TO_K = 1e6 * sc.e / sc.k


def test_g_effective_low_T_is_HRG() -> None:
    T_low = 1e6  # K, far below T_c (~1.8e12 K) → tanh saturates to -1
    assert g_effective(T_low) == pytest.approx(G_HRG, abs=1e-4)


def test_g_effective_high_T_is_QGP() -> None:
    T_high = 1e15  # K, far above T_c → tanh saturates to +1
    assert g_effective(T_high) == pytest.approx(G_QGP, abs=1e-4)


def test_g_effective_at_T_c_is_average() -> None:
    T_c_K = T_C_MEV * _MEV_TO_K
    assert g_effective(T_c_K) == pytest.approx((G_HRG + G_QGP) / 2.0, rel=1e-12)


def test_g_effective_monotone_through_crossover() -> None:
    T_c_K = T_C_MEV * _MEV_TO_K
    dT_K = DELTA_T_MEV * _MEV_TO_K
    samples = [g_effective(T_c_K + k * dT_K) for k in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert all(s_next > s_prev for s_prev, s_next in itertools.pairwise(samples))


def test_g_effective_rejects_nonpositive_T() -> None:
    with pytest.raises(ValueError):
        g_effective(0.0)
    with pytest.raises(ValueError):
        g_effective(-1.0)


def test_g_effective_matches_symbolic() -> None:
    """Numerical g_effective matches the lambdified SymPy expression."""
    f = g_effective_lambdified()
    T_c_K = T_C_MEV * _MEV_TO_K
    dT_K = DELTA_T_MEV * _MEV_TO_K
    for T_K in [1e6, T_c_K, 2 * T_c_K, 10 * T_c_K]:
        assert g_effective(T_K) == pytest.approx(f(T_K, T_c_K, dT_K, G_HRG, G_QGP), rel=1e-12)


def test_energy_density_at_T_c_is_GeV_per_fm3_scale() -> None:
    """Bjorken benchmark: ε(T ≈ T_c) is of order 1 GeV/fm³ ≈ 1.6 × 10³⁵ J/m³.

    At our crossover midpoint g_eff ≈ 25, so we land near 0.5 GeV/fm³ — within
    a factor of 3 of the textbook estimate. Above T_c the density grows as T⁴.
    """
    T_K = T_C_MEV * _MEV_TO_K
    src = QuarkGluonPlasmaSource(volume=1.0, temperature=T_K)
    eps = src.energy_density()
    expected = 1.6e35
    assert abs(math.log10(eps) - math.log10(expected)) < 0.6, f"ε={eps:.2e} far from {expected:.2e}"


def test_energy_density_T_dependence_is_T4() -> None:
    """In the plateau region (T ≫ T_c), ε ∝ T⁴ to high accuracy."""
    T1 = 200.0 * _MEV_TO_K
    T2 = 400.0 * _MEV_TO_K
    s1 = QuarkGluonPlasmaSource(volume=1.0, temperature=T1)
    s2 = QuarkGluonPlasmaSource(volume=1.0, temperature=T2)
    # g_eff is essentially constant in the plateau ⇒ ratio is ~16 (=2⁴).
    assert s2.energy_density() / s1.energy_density() == pytest.approx(16.0, rel=0.05)


def test_energy_density_matches_symbolic() -> None:
    """Stefan–Boltzmann numerical kernel matches lambdified expression."""
    f = sb_energy_density_lambdified()
    for T_MeV in [100.0, 200.0, 500.0]:
        T_K = T_MeV * _MEV_TO_K
        src = QuarkGluonPlasmaSource(volume=1.0, temperature=T_K)
        g = g_effective(T_K)
        expected = f(T_K, g, sc.k, sc.hbar, sc.c)
        assert src.energy_density() == pytest.approx(expected, rel=1e-12)


def test_total_energy_scales_linearly_with_volume() -> None:
    T_K = 200.0 * _MEV_TO_K
    s1 = QuarkGluonPlasmaSource(volume=1.0, temperature=T_K)
    s2 = QuarkGluonPlasmaSource(volume=10.0, temperature=T_K)
    assert s2.total_energy() == pytest.approx(10.0 * s1.total_energy(), rel=1e-12)


def test_validation() -> None:
    T_K = 200.0 * _MEV_TO_K
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=0.0, temperature=T_K)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=0.0)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=T_K, containment_efficiency=0.0)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=T_K, containment_efficiency=1.5)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=T_K, graviton_yield=-0.1)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=T_K, graviton_energy_quantum_j=0.0)
    with pytest.raises(ValueError):
        QuarkGluonPlasmaSource(volume=1.0, temperature=T_K, time_constant_s=-1.0)


def test_source_metadata_lists_speculative_components() -> None:
    T_K = 200.0 * _MEV_TO_K
    src = QuarkGluonPlasmaSource(volume=1.0, temperature=T_K)
    spec = src.metadata.get("speculative_components", [])
    assert "containment_efficiency" in spec
    assert "graviton_yield" in spec
    assert "graviton_energy_quantum_J" in spec
