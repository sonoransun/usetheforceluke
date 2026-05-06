"""QGPGravitonField: protocol contract, λ→∞ limit, Γ scaling with QGP volume."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce.qgp import QGPGravitonField, QuarkGluonPlasmaSource

_MEV_TO_K = 1e6 * sc.e / sc.k


def _make_source(volume_m3: float = 1.0, temp_mev: float = 200.0) -> QuarkGluonPlasmaSource:
    return QuarkGluonPlasmaSource(volume=volume_m3, temperature=temp_mev * _MEV_TO_K)


def _make_field(**kwargs) -> QGPGravitonField:
    defaults: dict = {
        "source": _make_source(),
        "source_position": (0.0, 0.0, 0.0),
        "screening_length": 1e6,
        "coupling_g": 1.0,
        "probe_mass": 1.0,
    }
    defaults.update(kwargs)
    return QGPGravitonField(**defaults)


def test_protocol_contract() -> None:
    ff = _make_field()
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array([1.0, 0.0, 0.0]))
    assert f.shape == (3,)
    assert np.all(np.isfinite(f))
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "qgp"
    assert "source_metadata" in ff.metadata


def test_validates_input() -> None:
    src = _make_source()
    with pytest.raises(ValueError):
        QGPGravitonField(source=src, source_position=(0.0, 0.0))
    with pytest.raises(ValueError):
        QGPGravitonField(source=src, screening_length=0.0)
    with pytest.raises(ValueError):
        QGPGravitonField(source=src, coupling_g=-1.0)
    with pytest.raises(ValueError):
        QGPGravitonField(source=src, probe_mass=0.0)


def test_force_attractive_toward_source() -> None:
    ff = _make_field()
    f = ff.force(0.0, np.array([2.0, 0.0, 0.0]))
    assert f[0] < 0  # attractive (toward origin)
    assert f[1] == 0.0 and f[2] == 0.0


def test_large_lambda_recovers_inverse_square() -> None:
    """λ ≫ R ⇒ |F| ≈ -m·g·Γ/R² along -r̂ (same limit as antimatter graviton)."""
    src = _make_source()
    ff = _make_field(source=src, screening_length=1.0e15)
    gamma = src.graviton_emission_rate()
    for d in [0.5, 1.0, 3.0]:
        f = ff.force(0.0, np.array([d, 0.0, 0.0]))
        expected = -gamma / d**2  # m=1, g=1
        np.testing.assert_allclose(f, [expected, 0.0, 0.0], rtol=1e-6)


def test_force_scales_linearly_with_qgp_volume() -> None:
    """Γ ∝ ε·V ∝ V at fixed T, so |F| should scale linearly with QGP volume."""
    s1 = _make_source(volume_m3=1.0)
    s2 = _make_source(volume_m3=10.0)
    f1 = _make_field(source=s1).force(0.0, np.array([1.0, 0.0, 0.0]))
    f2 = _make_field(source=s2).force(0.0, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(f2, 10.0 * f1, rtol=1e-12)


def test_force_is_negative_gradient_of_potential() -> None:
    ff = _make_field()
    r0 = np.array([1.5, -0.7, 0.4])
    h = 1e-6
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, atol=1e-7)


def test_gamma_cached() -> None:
    """The cached Γ matches the source's graviton_emission_rate at construction."""
    src = _make_source()
    ff = _make_field(source=src)
    assert ff.gamma == pytest.approx(src.graviton_emission_rate(), rel=1e-12)
