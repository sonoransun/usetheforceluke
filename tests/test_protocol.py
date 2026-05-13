"""Every avenue must satisfy ``ForceField`` and produce shape-(3,) finite forces."""

from __future__ import annotations

import numpy as np
import scipy.constants as sc

from usetheforce import ForceField, ureg
from usetheforce.antimatter import AntimatterCounterGravity, AntimatterGravitonField
from usetheforce.blackhole import BlackHoleCounterDrive, SchwarzschildGravity
from usetheforce.casimir import ParallelPlateCasimir, ScaledCasimir
from usetheforce.qfield import HeavyElementLattice, ShapedFieldAnsatz, StimulatedEmissionArray
from usetheforce.qgp import QGPGravitonField, QuarkGluonPlasmaSource

_MEV_TO_K = 1e6 * sc.e / sc.k


def _check(ff: ForceField, probe: tuple[float, float, float] = (0.1, 0.2, 0.3)) -> None:
    assert isinstance(ff, ForceField)
    f = ff.force(0.0, np.array(probe))
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


def test_qgp_graviton_field_protocol() -> None:
    src = QuarkGluonPlasmaSource(volume=1.0, temperature=200.0 * _MEV_TO_K)
    ff = QGPGravitonField(source=src, screening_length=5.0, probe_mass=1.0)
    _check(ff)
    assert ff.metadata["speculative"] is True
    assert ff.metadata["avenue"] == "qgp"


_M_SUN = 1.98892e30


def test_schwarzschild_gravity_protocol() -> None:
    ff = SchwarzschildGravity(mass_kg=_M_SUN, probe_mass_kg=1.0)
    r_s = ff.schwarzschild_radius_m
    # Probe must sit outside the horizon for SchwarzschildGravity.
    _check(ff, probe=(10 * r_s, 0.0, 0.0))
    assert ff.metadata["speculative"] is False
    assert ff.metadata["avenue"] == "blackhole"


def test_blackhole_counter_drive_protocol() -> None:
    drive = BlackHoleCounterDrive.from_schwarzschild(
        mass_kg=_M_SUN, probe_mass_kg=1.0, efficiency=1e-9
    )
    rs = SchwarzschildGravity(mass_kg=_M_SUN, probe_mass_kg=1.0).schwarzschild_radius_m
    _check(drive, probe=(10 * rs, 0.0, 0.0))
    assert drive.metadata["speculative"] is True
    assert drive.metadata["avenue"] == "blackhole"
