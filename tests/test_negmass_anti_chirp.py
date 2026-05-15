"""Anti-chirp binary: protocol, Yukawa falloff, df/dt < 0 sign."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce._negmass import chirp_mass
from usetheforce.negmass import AntiChirpBinary

G = sc.G


def _make(
    m_positive_kg: float = 10.0,
    m_negative_kg: float = 1.0,
    separation_m: float = 1.0e6,
    probe_mass_kg: float = 100.0,
    screening_m: float = 1.0e12,
) -> AntiChirpBinary:
    return AntiChirpBinary(
        m_positive_kg=m_positive_kg,
        m_negative_kg=m_negative_kg,
        separation_m=separation_m,
        probe_mass_kg=probe_mass_kg,
        screening_m=screening_m,
    )


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "negmass"
    assert ff.metadata["applicable_for_trajectory"] is True


def test_validates_input() -> None:
    # Positive total mass requires m_pos > m_neg.
    with pytest.raises(ValueError):
        _make(m_positive_kg=1.0, m_negative_kg=2.0)
    with pytest.raises(ValueError):
        AntiChirpBinary(
            m_positive_kg=0.0, m_negative_kg=1.0, separation_m=1.0, probe_mass_kg=1.0
        )
    with pytest.raises(ValueError):
        _make(m_negative_kg=-1.0)
    with pytest.raises(ValueError):
        _make(separation_m=-1.0)
    with pytest.raises(ValueError):
        _make(probe_mass_kg=0.0)
    with pytest.raises(ValueError):
        _make(screening_m=-1.0)


def test_signed_chirp_mass_is_negative() -> None:
    """For anti-chirp (one negative component), the signed chirp mass is negative."""
    ff = _make(m_positive_kg=10.0, m_negative_kg=1.0)
    assert ff.signed_chirp_mass_kg < 0
    # And matches the helper's signed value when fed (m_pos, -m_neg).
    expected = chirp_mass(10.0, -1.0)
    assert ff.signed_chirp_mass_kg == pytest.approx(expected, rel=1e-12)


def test_df_dt_is_negative_for_anti_chirp() -> None:
    """Peters–Mathews df/dt < 0 when M_c < 0 (orbit expands, frequency falls)."""
    ff = _make()
    for f_hz in (1.0, 100.0, 1e4):
        assert ff.df_dt(f_hz) < 0


def test_df_dt_validates_positive_frequency() -> None:
    ff = _make()
    with pytest.raises(ValueError):
        ff.df_dt(0.0)
    with pytest.raises(ValueError):
        ff.df_dt(-1.0)


def test_force_attractive_in_yukawa_falloff_limit() -> None:
    """In λ → ∞ the field reduces to Newtonian inverse-square attraction on total mass."""
    m_pos, m_neg = 10.0, 1.0
    M_total = m_pos - m_neg  # = 9
    probe_mass = 1.0
    ff = _make(
        m_positive_kg=m_pos,
        m_negative_kg=m_neg,
        separation_m=1.0e6,
        probe_mass_kg=probe_mass,
        screening_m=1.0e30,
    )
    for R in (1.0, 10.0, 100.0):
        f = ff.force(0.0, np.array([R, 0.0, 0.0]))
        expected_mag = G * M_total * probe_mass / (R * R)
        # F is attractive (toward origin), so along -x.
        assert f[0] == pytest.approx(-expected_mag, rel=1e-6)
        assert f[1] == pytest.approx(0.0, abs=1e-30)


def test_force_is_negative_gradient_of_potential() -> None:
    ff = _make(screening_m=1.0e8)
    r0 = np.array([1.0e5, -3.0e4, 2.0e4])
    h = 1.0  # m — small relative to 1e5 m positions
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (ff.potential(rp) - ff.potential(rm)) / (2 * h)
    np.testing.assert_allclose(ff.force(0.0, r0), -grad, rtol=1e-4)
