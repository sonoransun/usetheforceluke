"""Mission-planning helpers: shoot-and-correct toward a target position."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import root

from usetheforce.protocol import ForceField
from usetheforce.trajectories.integrator import integrate


def delta_v_for_target(
    ff: ForceField,
    mass: float,
    r0: Sequence[float],
    v0_guess: Sequence[float],
    r_target: Sequence[float],
    t_flight: float,
) -> np.ndarray:
    """Solve for the launch velocity ``v0`` that puts the craft at ``r_target`` after ``t_flight``.

    Returns the *delta-v* relative to ``v0_guess`` (so ``v0_guess + return == launch v0``).
    Single-shooting; assumes a smooth force field and a reasonable guess.
    """
    r0_arr = np.asarray(r0, dtype=float)
    rt_arr = np.asarray(r_target, dtype=float)
    v_guess = np.asarray(v0_guess, dtype=float)

    def residual(v: np.ndarray) -> np.ndarray:
        traj = integrate(ff, mass, r0_arr.tolist(), v.tolist(), (0.0, t_flight), n_eval=2)
        return traj.r[-1] - rt_arr

    sol = root(residual, v_guess, method="hybr")
    if not sol.success:
        raise RuntimeError(f"trajectory shooting failed: {sol.message}")
    return sol.x - v_guess
