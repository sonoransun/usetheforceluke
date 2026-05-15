"""NegativeMassPointSource: repulsive Newtonian gravity, protocol, validation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from usetheforce import ForceField
from usetheforce.negmass import NegativeMassPointSource

G = sc.G


def test_protocol() -> None:
    nm = NegativeMassPointSource(m_negative_kg=1.0, probe_mass_kg=1.0)
    assert isinstance(nm, ForceField)
    assert nm.metadata["speculative"] is True
    assert nm.metadata["avenue"] == "negmass"
    assert nm.metadata["applicable_for_trajectory"] is True


def test_validates_input() -> None:
    with pytest.raises(ValueError):
        NegativeMassPointSource(m_negative_kg=0.0, probe_mass_kg=1.0)
    with pytest.raises(ValueError):
        NegativeMassPointSource(m_negative_kg=-1.0, probe_mass_kg=1.0)
    with pytest.raises(ValueError):
        NegativeMassPointSource(m_negative_kg=1.0, probe_mass_kg=0.0)
    with pytest.raises(ValueError):
        NegativeMassPointSource(
            m_negative_kg=1.0, probe_mass_kg=1.0, source=(0.0, 0.0)  # type: ignore[arg-type]
        )


def test_force_is_repulsive() -> None:
    """Force on a positive-mass probe points *away* from the negative-mass source."""
    nm = NegativeMassPointSource(m_negative_kg=1.0e20, probe_mass_kg=1.0, source=(0, 0, 0))
    f = nm.force(0.0, np.array([1.0, 0.0, 0.0]))
    # Probe at +x ⇒ repulsion points along +x.
    assert f[0] > 0
    assert f[1] == 0.0 and f[2] == 0.0


def test_force_matches_repulsive_newtonian_law() -> None:
    """|F| = G · |m_neg| · m_probe / R²."""
    m_neg, m_probe = 1.0e20, 2.5
    nm = NegativeMassPointSource(m_negative_kg=m_neg, probe_mass_kg=m_probe)
    for R in (1.0, 10.0, 100.0):
        f = nm.force(0.0, np.array([R, 0.0, 0.0]))
        assert f[0] == pytest.approx(G * m_neg * m_probe / (R * R), rel=1e-12)


def test_potential_is_positive() -> None:
    """Repulsive potential is positive (sign-flipped vs. attractive gravity)."""
    nm = NegativeMassPointSource(m_negative_kg=1.0e20, probe_mass_kg=2.5)
    u = nm.potential(np.array([10.0, 0.0, 0.0]))
    assert u is not None and u > 0


def test_force_is_negative_gradient_of_potential() -> None:
    nm = NegativeMassPointSource(m_negative_kg=1.0e10, probe_mass_kg=2.5)
    r0 = np.array([5.0, -3.0, 2.0])
    h = 1e-4
    grad = np.empty(3)
    for i in range(3):
        rp = r0.copy()
        rm = r0.copy()
        rp[i] += h
        rm[i] -= h
        grad[i] = (nm.potential(rp) - nm.potential(rm)) / (2 * h)
    np.testing.assert_allclose(nm.force(0.0, r0), -grad, rtol=1e-6)


def test_source_property_returns_copy() -> None:
    nm = NegativeMassPointSource(m_negative_kg=1.0, probe_mass_kg=1.0, source=(1.0, 2.0, 3.0))
    s = nm.source
    s[0] = 99.0  # mutation should not affect the source
    np.testing.assert_array_equal(nm.source, np.array([1.0, 2.0, 3.0]))


def test_composite_buffer_against_schwarzschild_offsets_attraction() -> None:
    """A heavy enough buffer on the BH side flips the composite force outward."""
    from usetheforce import CompositeField
    from usetheforce.blackhole import SchwarzschildGravity

    M_SUN = 1.98892e30
    bh = SchwarzschildGravity(mass_kg=10 * M_SUN, probe_mass_kg=1.0)
    r_s = bh.schwarzschild_radius_m
    # Place a huge buffer between craft (50 r_s) and horizon (49 r_s).
    buf = NegativeMassPointSource(
        m_negative_kg=100 * M_SUN, probe_mass_kg=1.0, source=(49 * r_s, 0.0, 0.0)
    )
    comp = CompositeField(bh, buf)
    craft = np.array([50.0 * r_s, 0.0, 0.0])
    # With 100·M_SUN buffer at Δ = r_s vs 10·M_SUN BH at 50·r_s, buffer wins:
    # F_buf / F_BH = (100/10) · (50/1)² = 25000 ≫ 1. Net force +x (outward).
    f = comp.force(0.0, craft)
    assert f[0] > 0
