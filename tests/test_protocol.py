"""Every avenue must satisfy ``ForceField`` and produce shape-(3,) finite forces."""

from __future__ import annotations

import numpy as np

from usetheforce import ForceField, ureg
from usetheforce.antimatter import AntimatterCounterGravity, AntimatterGravitonField
from usetheforce.casimir import ParallelPlateCasimir, ScaledCasimir
from usetheforce.qfield import HeavyElementLattice, ShapedFieldAnsatz, StimulatedEmissionArray


def _check(ff: ForceField) -> None:
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array([0.1, 0.2, 0.3]))
    assert f.shape == (3,)
    assert np.all(np.isfinite(f))
    assert "avenue" in ff.metadata
    assert "speculative" in ff.metadata


def test_parallel_plate_casimir_protocol() -> None:
    ff = ParallelPlateCasimir(area=1.0 * ureg.cm**2, separation=100.0 * ureg.nm)
    _check(ff)
    assert ff.metadata["speculative"] is False


def test_scaled_casimir_protocol() -> None:
    ff = ScaledCasimir(area=1.0 * ureg.cm**2, separation=100.0 * ureg.nm, geometry_factor=2.5)
    _check(ff)
    assert ff.metadata["speculative"] is True


def test_shaped_field_protocol() -> None:
    ff = ShapedFieldAnsatz(amplitude=1.0, sigma=0.5)
    _check(ff)
    assert ff.metadata["speculative"] is True


def test_antimatter_protocol() -> None:
    ff = AntimatterCounterGravity(
        mass=10.0,
        efficiency=0.9,
        background_g=lambda _r: np.array([0.0, 0.0, -9.81]),
    )
    _check(ff)
    assert ff.metadata["speculative"] is True


def test_heavy_element_lattice_protocol() -> None:
    ff = HeavyElementLattice(
        sites=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        strengths=[1.0, 0.5],
        coupling=1.0,
        softening=0.1,
    )
    _check(ff)
    assert ff.metadata["speculative"] is True


def test_stimulated_emission_protocol() -> None:
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        amplitudes=[1.0, 1.0],
        phases=[0.0, np.pi / 2],
        wavenumber=2.0,
    )
    _check(ff)
    assert ff.metadata["speculative"] is True


def test_graviton_field_protocol() -> None:
    ff = AntimatterGravitonField(
        source=(0.0, 0.0, 0.0),
        gamma=1.0,
        coupling=1.0,
        screening=5.0,
        probe_mass=1.0,
    )
    _check(ff)
    assert ff.metadata["speculative"] is True
