"""Energy conservation under the conservative ``ShapedFieldAnsatz``."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from usetheforce.qfield import ShapedFieldAnsatz
from usetheforce.trajectories import integrate


@given(
    amplitude=st.floats(min_value=0.1, max_value=10.0),
    sigma=st.floats(min_value=0.5, max_value=3.0),
    v0=st.floats(min_value=-1.0, max_value=1.0),
)
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_total_energy_conserved(amplitude: float, sigma: float, v0: float) -> None:
    ff = ShapedFieldAnsatz(amplitude=amplitude, sigma=sigma)
    traj = integrate(
        ff,
        mass=1.0,
        r0=[2.0, 0.0, 0.0],
        v0=[v0, v0, 0.0],
        t_span=(0.0, 5.0),
        n_eval=80,
    )
    energy = traj.total_energy(ff)
    span = float(np.max(energy) - np.min(energy))
    scale = max(abs(float(np.mean(energy))), 1e-9)
    assert span / scale < 1e-6, f"energy drift fraction {span / scale} too large"
