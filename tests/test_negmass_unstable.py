"""Negative-total-mass binary: short-lived, raises at t_break."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce.negmass import NegativeTotalMassBinary

G = sc.G


def _make(
    m_positive_kg: float = 1.0,
    m_negative_kg: float = 5.0,
    separation_m: float = 1.0,
    craft_mass_kg: float = 1.0,
) -> NegativeTotalMassBinary:
    return NegativeTotalMassBinary(
        m_positive_kg=m_positive_kg,
        m_negative_kg=m_negative_kg,
        separation_m=separation_m,
        craft_mass_kg=craft_mass_kg,
    )


def test_protocol() -> None:
    ff = _make()
    assert isinstance(ff, ForceField)
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "negmass"
    assert ff.metadata["speculative_components"] == ["negative_mass_premise"]
    assert ff.metadata["applicable_for_trajectory"] is False


def test_validates_input() -> None:
    # Negative total mass requires |m_neg| > m_pos.
    with pytest.raises(ValueError):
        _make(m_positive_kg=5.0, m_negative_kg=1.0)
    with pytest.raises(ValueError):
        _make(m_positive_kg=0.0)
    with pytest.raises(ValueError):
        _make(m_negative_kg=-1.0)
    with pytest.raises(ValueError):
        _make(separation_m=-1.0)
    with pytest.raises(ValueError):
        _make(craft_mass_kg=0.0)


def test_t_break_matches_expected_formula() -> None:
    """t_break = π · sqrt(d³ / (G · |m_neg + m_pos_signed_net|))."""
    m_pos, m_neg, d = 1.0, 5.0, 2.0
    ff = _make(m_positive_kg=m_pos, m_negative_kg=m_neg, separation_m=d)
    expected = math.pi * math.sqrt(d**3 / (G * (m_neg - m_pos)))
    assert ff.t_break_s == pytest.approx(expected, rel=1e-12)


def test_force_repulsive_along_negative_axis() -> None:
    """Body force is along ``-axis`` (anti-Bondi)."""
    ff = _make()
    f = ff.force(0.0, np.zeros(3))
    assert f[0] < 0
    assert f[1] == 0.0 and f[2] == 0.0


def test_force_raises_at_break_time() -> None:
    ff = _make()
    t_break = ff.t_break_s
    # Force before break time is fine.
    _ = ff.force(0.5 * t_break, np.zeros(3))
    # At or after break time, raise.
    with pytest.raises(ValueError, match="disintegrated"):
        ff.force(t_break, np.zeros(3))
    with pytest.raises(ValueError, match="disintegrated"):
        ff.force(t_break * 1.1, np.zeros(3))


def test_potential_is_none() -> None:
    ff = _make()
    assert ff.potential(np.zeros(3)) is None
