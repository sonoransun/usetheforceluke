"""BlackHoleCounterDrive: cancellation, capping, callable/ForceField wrapping."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce import ForceField
from usetheforce.blackhole import BlackHoleCounterDrive, SchwarzschildGravity

M_SUN = 1.98892e30


def _bh(use_gr: bool = False) -> SchwarzschildGravity:
    return SchwarzschildGravity(
        mass_kg=M_SUN,
        probe_mass_kg=1.0,
        use_gr_hover_correction=use_gr,
    )


def test_protocol_and_speculative_flag() -> None:
    bh = _bh()
    drive = BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=bh, efficiency=1.0)
    assert isinstance(drive, ForceField)
    assert drive.metadata["speculative"] is True
    assert "efficiency" in drive.metadata["speculative_components"]
    assert "counter_drive_mechanism" in drive.metadata["speculative_components"]
    assert drive.metadata["avenue"] == "blackhole"
    assert drive.metadata["applicable_for_trajectory"] is True


def test_validates_input() -> None:
    bh = _bh()
    with pytest.raises(ValueError):
        BlackHoleCounterDrive(probe_mass_kg=0.0, background_g=bh)
    with pytest.raises(ValueError):
        BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=bh, efficiency=-0.1)
    with pytest.raises(ValueError):
        BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=bh, efficiency=1.1)
    with pytest.raises(ValueError):
        BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=bh, max_thrust_n=0.0)
    with pytest.raises(TypeError):
        BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=42)  # type: ignore[arg-type]


def test_potential_is_none() -> None:
    drive = BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=_bh())
    assert drive.potential(np.array([1.0, 0.0, 0.0])) is None


def test_exact_cancellation_at_eta_one() -> None:
    """Drive force exactly cancels SchwarzschildGravity at η=1."""
    m = 1.0
    bh = SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=m)
    r_s = bh.schwarzschild_radius_m
    drive = BlackHoleCounterDrive(probe_mass_kg=m, background_g=bh, efficiency=1.0)
    r = np.array([5.0 * r_s, 0.0, 0.0])
    sum_force = bh.force(0.0, r) + drive.force(0.0, r)
    np.testing.assert_allclose(sum_force, np.zeros(3), atol=1e-10)


def test_zero_force_at_eta_zero() -> None:
    drive = BlackHoleCounterDrive(probe_mass_kg=1.0, background_g=_bh(), efficiency=0.0)
    bh = _bh()
    r = np.array([5.0 * bh.schwarzschild_radius_m, 0.0, 0.0])
    np.testing.assert_allclose(drive.force(0.0, r), np.zeros(3))


def test_max_thrust_cap_preserves_direction() -> None:
    """Cap reduces magnitude while keeping the unit vector."""
    bh = _bh()
    r_s = bh.schwarzschild_radius_m
    # At 5 r_s the Newtonian force is sizable; cap below it.
    raw = float(np.linalg.norm(bh.force(0.0, np.array([5 * r_s, 0.0, 0.0]))))
    cap = raw / 100.0
    drive = BlackHoleCounterDrive(
        probe_mass_kg=1.0, background_g=bh, efficiency=1.0, max_thrust_n=cap
    )
    F = drive.force(0.0, np.array([5 * r_s, 0.0, 0.0]))
    assert float(np.linalg.norm(F)) == pytest.approx(cap, rel=1e-12)
    # Direction is +x (outward, opposite to gravity).
    assert F[0] > 0 and F[1] == 0.0 and F[2] == 0.0


def test_accepts_bare_callable() -> None:
    """A plain g(r) callable is accepted in lieu of a ForceField."""

    def constant_g(r: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, -9.81])

    drive = BlackHoleCounterDrive(
        probe_mass_kg=10.0, background_g=constant_g, efficiency=1.0
    )
    F = drive.force(0.0, np.array([1.0, 2.0, 3.0]))
    # F = -1 · 10 · (0, 0, -9.81) = (0, 0, 98.1)
    np.testing.assert_allclose(F, np.array([0.0, 0.0, 98.1]))


def test_from_schwarzschild_classmethod_round_trip() -> None:
    """from_schwarzschild builds an internal SchwarzschildGravity equivalent to manual wiring."""
    direct = BlackHoleCounterDrive.from_schwarzschild(
        mass_kg=M_SUN, probe_mass_kg=2.0, efficiency=0.5
    )
    manual = BlackHoleCounterDrive(
        probe_mass_kg=2.0,
        background_g=SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=2.0),
        efficiency=0.5,
    )
    rs = SchwarzschildGravity(mass_kg=M_SUN, probe_mass_kg=2.0).schwarzschild_radius_m
    r = np.array([5 * rs, 0.0, 0.0])
    np.testing.assert_allclose(direct.force(0.0, r), manual.force(0.0, r), rtol=1e-12)
