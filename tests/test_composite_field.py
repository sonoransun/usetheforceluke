"""CompositeField: vector summation, metadata aggregation, potential propagation."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import CompositeField, ForceField
from usetheforce.antimatter import AntimatterCounterGravity, AntimatterGravitonField
from usetheforce.casimir import ParallelPlateCasimir
from usetheforce.qfield import ShapedFieldAnsatz
from usetheforce.units import ureg


def _graviton(gamma: float = 1.0) -> AntimatterGravitonField:
    return AntimatterGravitonField(
        source=(0.0, 0.0, 0.0),
        gamma=gamma,
        coupling=1.0,
        screening=1.0e6,
        probe_mass=1.0,
    )


def _ansatz(amp: float = 0.5) -> ShapedFieldAnsatz:
    return ShapedFieldAnsatz(amplitude=amp, sigma=1.0, center=(0.0, 0.0, 0.0))


def test_protocol() -> None:
    comp = CompositeField(_graviton(), _ansatz())
    assert isinstance(comp, ForceField)


def test_force_is_vector_sum() -> None:
    g = _graviton(gamma=1.0)
    a = _ansatz(amp=0.5)
    comp = CompositeField(g, a)
    r = np.array([1.5, -0.7, 0.4])
    expected = g.force(0.0, r) + a.force(0.0, r)
    np.testing.assert_allclose(comp.force(0.0, r), expected, rtol=1e-12)


def test_potential_is_scalar_sum_when_both_conservative() -> None:
    g = _graviton()
    a = _ansatz()
    comp = CompositeField(g, a)
    r = np.array([2.0, 0.0, 0.0])
    u_g = g.potential(r)
    u_a = a.potential(r)
    assert u_g is not None and u_a is not None
    assert comp.potential(r) == pytest.approx(u_g + u_a, rel=1e-12)


def test_potential_none_when_any_component_nonconservative() -> None:
    g = _graviton()
    # AntimatterCounterGravity returns None potential by design.
    counter = AntimatterCounterGravity(
        mass=1.0, efficiency=0.5, background_g=lambda r: np.zeros(3)
    )
    comp = CompositeField(g, counter)
    assert comp.potential(np.array([1.0, 0.0, 0.0])) is None


def test_metadata_speculative_is_or_over_components() -> None:
    pp = ParallelPlateCasimir(area=1.0 * ureg("cm^2"), separation=100.0 * ureg("nm"))
    g = _graviton()
    assert pp.metadata["speculative"] is False
    assert g.metadata["speculative"] is True
    comp = CompositeField(pp, g)
    assert comp.metadata["speculative"] is True  # OR
    comp_anchored = CompositeField(pp)
    assert comp_anchored.metadata["speculative"] is False


def test_metadata_speculative_components_is_sorted_set_union() -> None:
    g1 = _graviton()  # ["gamma", "coupling", "screening"]
    a = _ansatz()  # ["amplitude", "sigma", "center"] per ShapedFieldAnsatz
    comp = CompositeField(g1, a)
    sc = comp.metadata["speculative_components"]
    assert isinstance(sc, list)
    assert sc == sorted(set(sc))  # sorted, deduplicated
    assert "gamma" in sc
    assert "coupling" in sc
    # Duplicate names from the same component should not be repeated.
    comp_dup = CompositeField(g1, _graviton(gamma=2.0))
    assert comp_dup.metadata["speculative_components"].count("gamma") == 1


def test_metadata_applicable_for_trajectory_is_and() -> None:
    pp = ParallelPlateCasimir(area=1.0 * ureg("cm^2"), separation=100.0 * ureg("nm"))
    g = _graviton()
    assert pp.metadata["applicable_for_trajectory"] is False
    assert g.metadata.get("applicable_for_trajectory", True) is True
    comp = CompositeField(pp, g)
    assert comp.metadata["applicable_for_trajectory"] is False  # AND


def test_metadata_components_retains_per_component_dicts() -> None:
    g = _graviton()
    a = _ansatz()
    comp = CompositeField(g, a)
    components_meta = comp.metadata["components"]
    assert len(components_meta) == 2
    assert components_meta[0]["avenue"] == "antimatter"
    assert components_meta[1]["avenue"] == "qfield"


def test_empty_tuple_raises() -> None:
    with pytest.raises(ValueError):
        CompositeField()


def test_non_forcefield_component_raises() -> None:
    with pytest.raises(TypeError):
        CompositeField(_graviton(), "not a force field")  # type: ignore[arg-type]


def test_single_component_is_permitted() -> None:
    g = _graviton()
    comp = CompositeField(g)
    r = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(comp.force(0.0, r), g.force(0.0, r), rtol=1e-12)


def test_avenue_is_composite() -> None:
    comp = CompositeField(_graviton())
    assert comp.metadata["avenue"] == "composite"
