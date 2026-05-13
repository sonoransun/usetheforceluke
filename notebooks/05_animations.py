"""Generate the 2D animations embedded in README.md.

Deterministic — no RNG, no time-dependent inputs. Re-running this script alone
is sufficient to refresh both animated GIFs.

Outputs to ``assets/`` at the repo root:

- ``mission_dashboard.gif`` — animated 2×2 dashboard (trajectory + thrust +
  power + Δv) for one canonical heliocentric cruise.
- ``model_comparison.gif`` — same heliocentric cruise run with three different
  ``max_thrust_n`` levels, animated side by side to expose how thrust
  capability shapes the trajectory.
- ``blackhole_stationkeep.gif`` — dashboard animation of an
  ``event_horizon_stationkeep``-style hover (10 M_sun BH, city_ship). The
  controller is given a synthetic 10× the required hover thrust so the dashboard
  *visualises what a successful hover would look like*; the supertitle annotates
  the real-world shortfall against the vehicle's actual power budget.

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

from usetheforce._schwarzschild import G_NEWTON, gr_hover_factor, schwarzschild_radius
from usetheforce.control.controllers import ProportionalGuidance
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.power import VehiclePowerState
from usetheforce.missions import VEHICLES
from usetheforce.missions.long_range import LongRangeMissionResult
from usetheforce.trajectories import TrajectoryResult, integrate
from usetheforce.viz.control_animations import (
    animate_long_range_mission,
    animate_model_comparison,
)

M_SUN_KG = 1.98892e30

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


def _build_stationkeep(
    bh_mass_solar: float = 10.0,
    vehicle_key: str = "city_ship",
    hover_radius_factor: float = 1.5,
    duration_s: float = 60.0,
    n_eval: int = 120,
) -> LongRangeMissionResult:
    """Synthesised stationkeep ``LongRangeMissionResult`` for the dashboard.

    Live integration of Schwarzschild gravity at ``R = 1.5 r_s`` is intractable
    (acceleration ~ 10¹¹ m/s² blows up the DOP853 step controller). The
    dashboard's purpose is **visualising what a successful hover would look
    like** — so this helper *constructs* the result analytically: a probe
    pinned at the target radius, a slow tangential probe-frame motion for
    visible trajectory animation, and a thrust history equal to the required
    GR-corrected hover force. The supertitle then names the actual shortfall
    against the vehicle's real power budget, making the gap legible without
    pretending the integrator can resolve the dynamics.

    The narrative around this asset (see ``README.md``) calls out the
    visualisation fudge explicitly.
    """
    bh_mass_kg = bh_mass_solar * M_SUN_KG
    r_s = schwarzschild_radius(bh_mass_kg)
    target_radius = hover_radius_factor * r_s
    veh = VEHICLES[vehicle_key]

    required_newtonian = G_NEWTON * bh_mass_kg * veh.mass_kg / (target_radius * target_radius)
    required_gr = required_newtonian * gr_hover_factor(target_radius, r_s)
    supplied_actual = veh.power_w  # at V_REF = 1 m/s
    shortfall = required_gr / supplied_actual if supplied_actual > 0 else float("inf")

    # Synthesised trajectory: probe-frame "circular drift" at 5% of r_s in xy,
    # one full revolution over duration_s so the dashboard's trajectory panel
    # shows visible motion. Velocity magnitude is small enough that energy /
    # Δv panels stay readable.
    t = np.linspace(0.0, duration_s, n_eval)
    omega = 2.0 * np.pi / duration_s
    drift_radius = 0.05 * r_s
    rx = target_radius + drift_radius * np.cos(omega * t)
    ry = drift_radius * np.sin(omega * t)
    rz = np.zeros_like(t)
    vx = -drift_radius * omega * np.sin(omega * t)
    vy = drift_radius * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    r_arr = np.stack([rx, ry, rz], axis=1)
    v_arr = np.stack([vx, vy, vz], axis=1)
    traj = TrajectoryResult(
        t=t,
        r=r_arr,
        v=v_arr,
        mass=veh.mass_kg,
        field_metadata={
            "avenue": "blackhole",
            "model": "synthesised event_horizon_stationkeep",
            "speculative": True,
            "speculative_components": ["controller", "trajectory"],
        },
    )

    # Thrust history: required GR thrust pointing outward (radially); add a
    # tiny sinusoidal component along the drift direction so the thrust panel
    # has visible structure rather than a flat line.
    radial_unit = np.stack([rx, ry, rz], axis=1)
    radial_unit /= np.linalg.norm(radial_unit, axis=1, keepdims=True)
    perturb = 0.05 * required_gr * np.sin(2.0 * omega * t)
    thrust_arr = required_gr * radial_unit + perturb[:, None] * np.stack(
        [-np.sin(omega * t), np.cos(omega * t), np.zeros_like(t)], axis=1
    )

    # Power reserve: vehicle's actual reactor depleting at its rated power.
    initial_energy_j = veh.power_w * duration_s * 2.0
    cumulative = veh.power_w * t
    power_hist = np.maximum(0.0, initial_energy_j - cumulative)

    return LongRangeMissionResult(
        name=(
            f"event_horizon_stationkeep ({bh_mass_solar:g} M_sun, R = "
            f"{hover_radius_factor:g} r_s; supplied = {supplied_actual:.2e} N "
            f"vs. required = {required_gr:.2e} N → shortfall {shortfall:.1e}×)"
        ),
        trajectory=traj,
        delta_v_mps=0.0,  # closed drift, returns to origin
        burn_time_s=duration_s,
        energy_used_j=float(cumulative[-1]),
        peak_g=required_gr / (veh.mass_kg * 9.80665),
        thrust_history_n=thrust_arr,
        power_history_j=power_hist,
        controller_metadata={
            "controller": "ProportionalGuidance (synthesised — see notebook docstring)",
            "max_thrust_n": float(required_gr),
            "vehicle_key": vehicle_key,
        },
        background="schwarzschild_central",
        target_metric={
            "bh_mass_solar": bh_mass_solar,
            "schwarzschild_radius_m": r_s,
            "hover_radius_factor": hover_radius_factor,
            "target_radius_m": target_radius,
            "required_hover_newtonian_n": required_newtonian,
            "required_hover_gr_n": required_gr,
            "supplied_thrust_n": supplied_actual,
            "shortfall_ratio": shortfall,
        },
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

    # 3. Blackhole stationkeep dashboard.
    stationkeep = _build_stationkeep()
    animate_long_range_mission(
        stationkeep,
        output=ASSETS / "blackhole_stationkeep.gif",
        fps=20,
        every=2,
    )
    print("  blackhole_stationkeep.gif")

    print("done.")


if __name__ == "__main__":
    main()
