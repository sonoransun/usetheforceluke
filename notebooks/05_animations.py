"""Generate the 2D animations embedded in README.md.

Deterministic — no RNG, no time-dependent inputs. Re-running this script alone
is sufficient to refresh both animated GIFs.

Outputs to ``assets/`` at the repo root:

- ``mission_dashboard.gif`` — animated 2×2 dashboard (trajectory + thrust +
  power + Δv) for one canonical heliocentric cruise.
- ``model_comparison.gif`` — same heliocentric cruise run with three different
  ``max_thrust_n`` levels, animated side by side to expose how thrust
  capability shapes the trajectory.

The cruises are composed inline rather than via ``missions.long_range`` —
those factories hard-code ``rtol=1e-10``, which interacts badly with the
controller framework's finite-difference velocity estimator on long timespans
(the integrator stalls on tiny stage-step sizes). Calling ``integrate()``
directly with relaxed tolerances produces visually identical curves in
seconds rather than hours.

Run with: ``.venv/bin/python notebooks/05_animations.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from usetheforce.control.controllers import ProportionalGuidance
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.power import VehiclePowerState
from usetheforce.missions.long_range import LongRangeMissionResult
from usetheforce.trajectories import integrate
from usetheforce.viz.control_animations import (
    animate_long_range_mission,
    animate_model_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

GM_SUN = 1.32712440018e20  # m³/s² — heliocentric gravitational parameter
AU = 1.495978707e11  # m

VEHICLE_MASS_KG = 1.0e5  # interplanetary-class
BURN_TIME_S = 200.0 * 86400.0  # 200 days
N_EVAL = 160


def _sun_bg(r: np.ndarray) -> np.ndarray:
    rn = float(np.linalg.norm(r))
    if rn == 0:
        return np.zeros(3)
    return -GM_SUN * r / rn**3


def _build_cruise(max_thrust_n: float, name: str) -> LongRangeMissionResult:
    """Heliocentric proportional-guidance cruise from 1 AU toward 1.524 AU.

    Bypasses ``missions.long_range.heliocentric_cruise`` so we can use a
    relaxed ``rtol`` — see this module's docstring.
    """
    target = np.array([1.524 * AU, 0.0, 0.0])
    ctrl = ProportionalGuidance(target_position=target, gain=1.0e-6, max_thrust_n=max_thrust_n)
    field = ControlledThrustField(
        controller=ctrl,
        mass_kg=VEHICLE_MASS_KG,
        background=_sun_bg,
        power=VehiclePowerState(initial_energy_j=1.0e16, instantaneous_power_w=1.0e10),
    )
    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, float(np.sqrt(GM_SUN / AU)), 0.0])
    traj = integrate(
        field,
        mass=VEHICLE_MASS_KG,
        r0=r0,
        v0=v0,
        t_span=(0.0, BURN_TIME_S),
        n_eval=N_EVAL,
        rtol=1.0e-6,
        atol=1.0e-3,
    )

    log_t = np.array([entry[0] for entry in field.thrust_log], dtype=float)
    log_F = np.array([entry[1] for entry in field.thrust_log], dtype=float)
    thrust_arr = np.empty((traj.t.size, 3))
    for i in range(3):
        thrust_arr[:, i] = np.interp(traj.t, log_t, log_F[:, i]) if log_t.size > 0 else 0.0
    mech = np.einsum("ij,ij->i", thrust_arr, traj.v)
    cum = np.concatenate(([0.0], np.cumsum(np.abs(mech[:-1]) * np.diff(traj.t))))
    power_hist = np.maximum(0.0, 1.0e16 - cum)

    return LongRangeMissionResult(
        name=name,
        trajectory=traj,
        delta_v_mps=float(np.linalg.norm(traj.v[-1] - v0)),
        burn_time_s=BURN_TIME_S,
        energy_used_j=float(cum[-1]),
        peak_g=max_thrust_n / (VEHICLE_MASS_KG * 9.80665),
        thrust_history_n=thrust_arr,
        power_history_j=power_hist,
        controller_metadata={
            "controller": "ProportionalGuidance",
            "max_thrust_n": max_thrust_n,
        },
        background="sun_central",
        target_metric={"start_au": 1.0, "target_au": 1.524},
    )


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    print(f"writing animations to {ASSETS}/")

    # 1. Dashboard — single mid-thrust cruise.
    dashboard = _build_cruise(max_thrust_n=1.0e4, name="heliocentric_cruise")
    animate_long_range_mission(
        dashboard,
        output=ASSETS / "mission_dashboard.gif",
        fps=20,
        every=2,
    )
    print("  mission_dashboard.gif")

    # 2. Comparison — three thrust capability levels of the same cruise.
    levels_n = [1.0e3, 1.0e4, 1.0e5]
    labels = [f"max thrust = {lvl / 1e3:g} kN" for lvl in levels_n]
    comparison = [_build_cruise(max_thrust_n=lvl, name=f"cruise_{lvl:.0e}") for lvl in levels_n]
    animate_model_comparison(
        comparison,
        output=ASSETS / "model_comparison.gif",
        fps=20,
        every=2,
        labels=labels,
    )
    print("  model_comparison.gif")

    print("done.")


if __name__ == "__main__":
    main()
