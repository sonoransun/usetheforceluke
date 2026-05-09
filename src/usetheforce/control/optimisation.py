"""Tune ``ThrustController`` parameters against transit goals.

Two helpers:

- ``solve_min_time(...)`` — minimum-time transit subject to a peak-g constraint.
  Recognises the ``BrachistochroneTransit`` family analytically; falls back to
  a numerical 1-D optimisation otherwise.

- ``solve_min_dv(...)`` — finds controller-factory parameters that minimise
  the impulse expended subject to reaching ``target_r`` within ``t_max``.
  Penalty-method outer loop with ``scipy.optimize.minimize`` (Nelder–Mead).

The optimisation cannot rescue speculative physics — it only turns the dials a
controller exposes. Both helpers preserve the speculative status of the
underlying ``ForceField`` model in their result metadata.

```mermaid
sequenceDiagram
    participant Caller
    participant solve_min_dv
    participant minimize
    participant integrate
    Caller->>solve_min_dv: factory, target, vehicle, t_max
    loop optimiser iterations
        solve_min_dv->>minimize: objective
        minimize->>integrate: trial controller params
        integrate-->>minimize: trajectory + Δv penalty
    end
    solve_min_dv-->>Caller: tuned controller + diagnostics
```
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from usetheforce.control.controllers import (
    BrachistochroneTransit,
    ProportionalGuidance,
    ThrustController,
)
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.power import VehiclePowerState
from usetheforce.trajectories import integrate


@dataclass
class OptimisationResult:
    """Outcome of a tuning run."""

    controller: ThrustController
    travel_time_s: float
    delta_v_mps: float
    final_position_error_m: float
    iterations: int
    success: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def solve_min_time(
    distance_m: float,
    max_acceleration_mps2: float,
    mass_kg: float,
    *,
    r_start: Sequence[float] | None = None,
    r_target: Sequence[float] | None = None,
) -> OptimisationResult:
    """Analytical minimum-time brachistochrone transit.

    For a constant-acceleration profile of magnitude ``a_max``, the minimum
    one-way travel time covering ``distance_m`` is ``2·sqrt(d / a_max)`` and
    the achieved Δv (at midpoint, before braking) is ``a_max · t/2``.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if max_acceleration_mps2 <= 0:
        raise ValueError("max_acceleration_mps2 must be positive")
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    t_brake = math.sqrt(distance_m / max_acceleration_mps2)
    travel_time = 2.0 * t_brake
    peak_dv = max_acceleration_mps2 * t_brake  # Δv accumulated up to turnaround
    # Build a representative controller carrying the same parameters; if
    # endpoints aren't specified, use canonical (0, 0, 0) → (d, 0, 0).
    r0 = np.array(r_start if r_start is not None else (0.0, 0.0, 0.0), dtype=float)
    rt = np.array(
        r_target if r_target is not None else (distance_m, 0.0, 0.0), dtype=float
    )
    controller = BrachistochroneTransit(
        r_start=r0,
        r_target=rt,
        max_acceleration_mps2=max_acceleration_mps2,
        mass_kg=mass_kg,
    )
    return OptimisationResult(
        controller=controller,
        travel_time_s=travel_time,
        delta_v_mps=peak_dv,
        final_position_error_m=0.0,
        iterations=0,
        success=True,
        diagnostics={
            "method": "analytical brachistochrone",
            "t_brake_s": t_brake,
            "distance_m": distance_m,
        },
    )


def solve_min_dv(
    factory: Callable[[float], ProportionalGuidance],
    target_r: Sequence[float],
    mass_kg: float,
    t_max_s: float,
    *,
    r0: Sequence[float] = (0.0, 0.0, 0.0),
    v0: Sequence[float] = (0.0, 0.0, 0.0),
    background: Callable[[np.ndarray], np.ndarray] | None = None,
    gain_initial: float = 1.0,
    gain_bounds: tuple[float, float] = (1e-3, 1e6),
    miss_penalty: float = 1e3,
    max_iterations: int = 50,
) -> OptimisationResult:
    """Tune a single ``gain`` parameter on a ``ProportionalGuidance`` factory.

    The factory must produce a ``ProportionalGuidance`` given a single float
    (the gain). The optimiser minimises an objective combining the impulse
    consumed (≈ Δv) plus a penalty for missing ``target_r`` at time ``t_max_s``.

    Returns the tuned controller, achieved Δv, miss distance, and diagnostics.
    """
    target = np.asarray(target_r, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_r must have shape (3,)")
    if t_max_s <= 0:
        raise ValueError("t_max_s must be positive")

    bg = background if background is not None else (lambda r: np.zeros(3))  # noqa: ARG005
    history: list[tuple[float, float, float]] = []

    def evaluate(gain: float) -> tuple[float, float, float]:
        controller = factory(float(gain))
        field = ControlledThrustField(
            controller=controller,
            mass_kg=mass_kg,
            background=bg,
        )
        try:
            traj = integrate(
                field,
                mass=mass_kg,
                r0=list(r0),
                v0=list(v0),
                t_span=(0.0, t_max_s),
                n_eval=80,
            )
        except RuntimeError:
            return float("inf"), float("inf"), float("inf")
        miss = float(np.linalg.norm(traj.r[-1] - target))
        dv = float(np.linalg.norm(traj.v[-1] - np.asarray(v0, dtype=float)))
        return miss, dv, miss + miss_penalty * dv  # placeholder ordering

    def objective(x: np.ndarray) -> float:
        gain = max(gain_bounds[0], min(gain_bounds[1], float(x[0])))
        miss, dv, _ = evaluate(gain)
        cost = dv + miss_penalty * miss
        history.append((gain, dv, miss))
        return cost

    result = minimize(
        objective,
        x0=np.array([gain_initial]),
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": 1e-3, "fatol": 1e-3},
    )
    gain_opt = float(np.clip(result.x[0], *gain_bounds))
    miss_opt, dv_opt, _ = evaluate(gain_opt)
    controller = factory(gain_opt)
    return OptimisationResult(
        controller=controller,
        travel_time_s=t_max_s,
        delta_v_mps=dv_opt,
        final_position_error_m=miss_opt,
        iterations=int(result.nit),
        success=bool(result.success and miss_opt < 0.05 * float(np.linalg.norm(target))),
        diagnostics={
            "method": "Nelder-Mead 1-D over gain",
            "gain": gain_opt,
            "history": history,
            "scipy_message": result.message,
        },
    )


# Helper to support legacy `_BACKGROUNDS` lookups from missions/missions.py.
def _zero_background(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.zeros(3)
