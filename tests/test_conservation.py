"""Energy conservation across all conservative speculative ``ForceField``s."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.qfield import HeavyElementLattice, ShapedFieldAnsatz, StimulatedEmissionArray
from usetheforce.qgp import QGPGravitonField, QuarkGluonPlasmaSource
from usetheforce.trajectories import integrate


def _drift_fraction(energy: np.ndarray) -> float:
    span = float(np.max(energy) - np.min(energy))
    scale = max(abs(float(np.mean(energy))), 1e-9)
    return span / scale


@given(
    amplitude=st.floats(min_value=0.1, max_value=10.0),
    sigma=st.floats(min_value=0.5, max_value=3.0),
    v0=st.floats(min_value=-1.0, max_value=1.0),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_total_energy_conserved_shaped_field(amplitude: float, sigma: float, v0: float) -> None:
    ff = ShapedFieldAnsatz(amplitude=amplitude, sigma=sigma)
    traj = integrate(
        ff,
        mass=1.0,
        r0=[2.0, 0.0, 0.0],
        v0=[v0, v0, 0.0],
        t_span=(0.0, 5.0),
        n_eval=80,
    )
    drift = _drift_fraction(traj.total_energy(ff))
    assert drift < 1e-6, f"energy drift fraction {drift} too large"


@given(
    coupling=st.floats(min_value=0.5, max_value=5.0),
    softening=st.floats(min_value=0.3, max_value=1.5),
    v0=st.floats(min_value=-0.5, max_value=0.5),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_total_energy_conserved_heavy_element(coupling: float, softening: float, v0: float) -> None:
    ff = HeavyElementLattice(
        sites=[[0.0, 0.0, 0.0]],
        strengths=[1.0],
        coupling=coupling,
        softening=softening,
    )
    traj = integrate(
        ff,
        mass=1.0,
        r0=[3.0, 0.0, 0.0],
        v0=[v0, 0.5, 0.0],
        t_span=(0.0, 5.0),
        n_eval=80,
    )
    drift = _drift_fraction(traj.total_energy(ff))
    assert drift < 1e-6, f"energy drift fraction {drift} too large"


@given(
    amplitude=st.floats(min_value=0.5, max_value=2.0),
    wavenumber=st.floats(min_value=0.5, max_value=4.0),
    v0=st.floats(min_value=0.1, max_value=0.4),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_total_energy_conserved_stim_emission(
    amplitude: float, wavenumber: float, v0: float
) -> None:
    """Single-emitter stim-emission gives I ~ A²/R² — radial 1/R² potential admits
    no bound orbits, so we test conservation on a short outward trajectory that
    cannot reach the singularity within the integration window."""
    ff = StimulatedEmissionArray(
        positions=[[0.0, 0.0, 0.0]],
        amplitudes=[amplitude],
        phases=[0.0],
        wavenumber=wavenumber,
        coupling=1.0,
    )
    # Start at r=5 with outward velocity along x; even an attractive 1/R² potential
    # cannot reverse a positive-radial-velocity trajectory in 1 s with these magnitudes.
    traj = integrate(
        ff,
        mass=1.0,
        r0=[5.0, 0.0, 0.0],
        v0=[v0, 0.0, 0.0],
        t_span=(0.0, 1.0),
        n_eval=80,
    )
    drift = _drift_fraction(traj.total_energy(ff))
    assert drift < 1e-6, f"energy drift fraction {drift} too large"


@given(
    gamma=st.floats(min_value=0.5, max_value=5.0),
    coupling=st.floats(min_value=0.5, max_value=2.0),
    v0=st.floats(min_value=0.3, max_value=1.0),
)
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_total_energy_conserved_antimatter_graviton(
    gamma: float, coupling: float, v0: float
) -> None:
    ff = AntimatterGravitonField(
        source=(0.0, 0.0, 0.0),
        gamma=gamma,
        coupling=coupling,
        screening=1.0e15,  # large λ → effectively 1/r²
        probe_mass=1.0,
    )
    # Bound orbit-like initial condition: pick v0 below escape velocity.
    traj = integrate(
        ff,
        mass=1.0,
        r0=[1.0, 0.0, 0.0],
        v0=[0.0, v0, 0.0],
        t_span=(0.0, 4.0),
        n_eval=80,
    )
    drift = _drift_fraction(traj.total_energy(ff))
    assert drift < 1e-6, f"energy drift fraction {drift} too large"


def test_total_energy_conserved_qgp_graviton() -> None:
    """QGP graviton inherits the Yukawa form; large λ recovers conservative 1/r²."""
    import scipy.constants as sc  # noqa: PLC0415

    T_K = 200.0 * 1e6 * sc.e / sc.k
    src = QuarkGluonPlasmaSource(
        volume=1.0,
        temperature=T_K,
        containment_efficiency=1.0,
        graviton_yield=1e-50,  # tiny yield so |F| stays modest in our test units
    )
    ff = QGPGravitonField(
        source=src,
        screening_length=1.0e15,
        coupling_g=1.0,
        probe_mass=1.0,
    )
    # The cached Γ is finite; pick orbital initial conditions consistent with the
    # |F| this Γ produces.
    gamma = ff.gamma
    if gamma <= 0:
        return
    radius = 1.0
    v_circ = float(np.sqrt(gamma / radius))
    period = 2.0 * np.pi * radius / v_circ
    traj = integrate(
        ff,
        mass=1.0,
        r0=[radius, 0.0, 0.0],
        v0=[0.0, v_circ, 0.0],
        t_span=(0.0, period),
        n_eval=200,
    )
    drift = _drift_fraction(traj.total_energy(ff))
    assert drift < 1e-6, f"energy drift fraction {drift} too large"
