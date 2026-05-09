"""Long-range mission profiles using the ``control/`` framework.

Four named factories that build a ``ControlledThrustField``, integrate, and
return a ``LongRangeMissionResult`` carrying the trajectory plus thrust /
power histories sampled at the trajectory output times.

Mission catalogue:

1. ``interstellar_brachistochrone`` — constant-acceleration turnaround at midpoint.
2. ``heliocentric_cruise`` — Earth-orbit-relative start, target heliocentric radius.
3. ``lunar_stationkeep`` — proportional guidance against Earth-Moon gravity.
4. ``leo_orbit_modification`` — prograde burn under Earth central gravity.

Each factory returns a ``LongRangeMissionResult`` (extends ``MissionResult``);
the new visualisations in ``viz/control.py`` consume that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from usetheforce.control.controllers import (
    BrachistochroneTransit,
    ConstantThrust,
    ProportionalGuidance,
    ThrustController,
)
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.power import VehiclePowerState
from usetheforce.missions.missions import GM_EARTH, R_EARTH, MissionResult
from usetheforce.missions.vehicles import Vehicle
from usetheforce.trajectories import TrajectoryResult, integrate

# Astronomical constants (anchored).
AU: float = 1.495978707e11  # m
GM_SUN: float = 1.32712440018e20  # m³/s²
GM_MOON: float = 4.9048695e12  # m³/s²
EARTH_MOON_DISTANCE: float = 3.844e8  # m


# --------------------------------------------------------------------------------------
# Background gravity functions
# --------------------------------------------------------------------------------------


def _free_bg(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.zeros(3)


def _sun_central_bg(r: np.ndarray) -> np.ndarray:
    rn = float(np.linalg.norm(r))
    if rn == 0:
        return np.zeros(3)
    return -GM_SUN * r / rn**3


def _earth_central_bg(r: np.ndarray) -> np.ndarray:
    rn = float(np.linalg.norm(r))
    if rn == 0:
        return np.zeros(3)
    return -GM_EARTH * r / rn**3


def _earth_moon_two_body_bg(r: np.ndarray) -> np.ndarray:
    """Two-body gravity from Earth at origin and Moon at (D, 0, 0)."""
    moon_pos = np.array([EARTH_MOON_DISTANCE, 0.0, 0.0])
    rn_e = float(np.linalg.norm(r))
    rel_m = r - moon_pos
    rn_m = float(np.linalg.norm(rel_m))
    g = np.zeros(3)
    if rn_e > 0:
        g += -GM_EARTH * r / rn_e**3
    if rn_m > 0:
        g += -GM_MOON * rel_m / rn_m**3
    return g


# --------------------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LongRangeMissionResult:
    """Trajectory + thrust/power history + mission-specific outcome metrics."""

    name: str
    trajectory: TrajectoryResult
    delta_v_mps: float
    burn_time_s: float
    energy_used_j: float
    peak_g: float
    thrust_history_n: np.ndarray  # (N, 3) — sampled at trajectory.t
    power_history_j: np.ndarray  # (N,) — energy reserve remaining at each t
    controller_metadata: dict[str, Any]
    background: str
    target_metric: dict[str, Any] = field(default_factory=dict)

    def to_mission_result(self) -> MissionResult:
        thrust_mags = np.linalg.norm(self.thrust_history_n, axis=1)
        peak_thrust = float(thrust_mags.max()) if thrust_mags.size else 0.0
        return MissionResult(
            mission_key=self.name,
            vehicle_key=self.controller_metadata.get("vehicle_key", ""),
            model_key=self.controller_metadata.get("controller", ""),
            trajectory=self.trajectory,
            delta_v_mps=self.delta_v_mps,
            burn_time_s=self.burn_time_s,
            energy_j=self.energy_used_j,
            peak_accel_mps2=self.peak_g * 9.80665,
            peak_g=self.peak_g,
            thrust_n=peak_thrust,
            assumptions=dict(self.controller_metadata),
        )


# --------------------------------------------------------------------------------------
# Shared run helper
# --------------------------------------------------------------------------------------


def _run_with_field(
    name: str,
    field_obj: ControlledThrustField,
    mass_kg: float,
    r0: np.ndarray,
    v0: np.ndarray,
    t_span: tuple[float, float],
    n_eval: int,
    background_label: str,
    target_metric: dict[str, Any] | None = None,
) -> LongRangeMissionResult:
    field_obj.reset()
    traj = integrate(
        field_obj,
        mass=mass_kg,
        r0=list(r0),
        v0=list(v0),
        t_span=t_span,
        n_eval=n_eval,
    )
    # Sample thrust at the trajectory output times by linearly interpolating
    # the controller's logged calls.
    log_t = np.array([entry[0] for entry in field_obj.thrust_log], dtype=float)
    log_F = np.array([entry[1] for entry in field_obj.thrust_log], dtype=float)
    thrust_at_eval = np.empty((traj.t.size, 3), dtype=float)
    for i in range(3):
        if log_t.size > 0:
            thrust_at_eval[:, i] = np.interp(traj.t, log_t, log_F[:, i])
        else:
            thrust_at_eval[:, i] = 0.0
    # Approximate power-history reserve (integrate |F·v| over time).
    v_arr = traj.v
    mech_p = np.einsum("ij,ij->i", thrust_at_eval, v_arr)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(np.abs(mech_p[:-1]) * np.diff(traj.t))))
    initial_energy = field_obj.power.initial_energy_j
    power_history = np.maximum(0.0, initial_energy - cumulative_energy)
    delta_v = float(np.linalg.norm(traj.v[-1] - v0))
    energy_used = float(cumulative_energy[-1]) if cumulative_energy.size else 0.0
    peak_thrust = float(np.max(np.linalg.norm(thrust_at_eval, axis=1))) if thrust_at_eval.size else 0.0
    peak_g = peak_thrust / (mass_kg * 9.80665)
    return LongRangeMissionResult(
        name=name,
        trajectory=traj,
        delta_v_mps=delta_v,
        burn_time_s=t_span[1] - t_span[0],
        energy_used_j=energy_used,
        peak_g=peak_g,
        thrust_history_n=thrust_at_eval,
        power_history_j=power_history,
        controller_metadata=dict(field_obj.metadata),
        background=background_label,
        target_metric=target_metric or {},
    )


# --------------------------------------------------------------------------------------
# Mission factories
# --------------------------------------------------------------------------------------


def interstellar_brachistochrone(
    distance_ly: float,
    max_g: float,
    vehicle: Vehicle,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Constant-acceleration interstellar transit, turnaround at midpoint.

    Returns a ``LongRangeMissionResult`` with the analytical travel time
    (``trajectory.t[-1] = 2·sqrt(d / a_max)``) and the achieved Δv.
    """
    if distance_ly <= 0:
        raise ValueError("distance_ly must be positive")
    if max_g <= 0:
        raise ValueError("max_g must be positive")
    LIGHT_YEAR_M = 9.4607304725808e15
    distance_m = distance_ly * LIGHT_YEAR_M
    a_max = max_g * 9.80665
    r_start = np.zeros(3)
    r_target = np.array([distance_m, 0.0, 0.0])
    controller = BrachistochroneTransit(
        r_start=r_start,
        r_target=r_target,
        max_acceleration_mps2=a_max,
        mass_kg=vehicle.mass_kg,
    )
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_free_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * controller.analytical_travel_time_s,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    return _run_with_field(
        name="interstellar_brachistochrone",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r_start,
        v0=np.zeros(3),
        t_span=(0.0, controller.analytical_travel_time_s),
        n_eval=n_eval,
        background_label="free",
        target_metric={
            "distance_ly": distance_ly,
            "distance_m": distance_m,
            "max_g": max_g,
            "analytical_travel_time_s": controller.analytical_travel_time_s,
        },
    )


def heliocentric_cruise(
    start_au: float,
    target_au: float,
    vehicle: Vehicle,
    *,
    burn_time_days: float = 200.0,
    gain: float = 1.0e-9,
    max_thrust_n: float | None = None,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Heliocentric transfer using ``ProportionalGuidance`` against Sun gravity.

    Starts at ``start_au`` heliocentric on the +x axis with the local circular
    velocity (Keplerian); steers toward ``target_au`` with proportional gain
    over ``burn_time_days`` days.
    """
    if start_au <= 0 or target_au <= 0:
        raise ValueError("start_au and target_au must be positive")
    r0 = np.array([start_au * AU, 0.0, 0.0])
    v0 = np.array([0.0, float(np.sqrt(GM_SUN / r0[0])), 0.0])
    target_position = np.array([target_au * AU, 0.0, 0.0])
    controller = ProportionalGuidance(
        target_position=target_position,
        gain=gain,
        max_thrust_n=max_thrust_n if max_thrust_n is not None else 1e6,
    )
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_sun_central_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * burn_time_days * 86400.0,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    t_span = (0.0, burn_time_days * 86400.0)
    return _run_with_field(
        name="heliocentric_cruise",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r0,
        v0=v0,
        t_span=t_span,
        n_eval=n_eval,
        background_label="sun_central",
        target_metric={
            "start_au": start_au,
            "target_au": target_au,
            "gain": gain,
        },
    )


def lunar_stationkeep(
    target_position_m: tuple[float, float, float],
    duration_s: float,
    vehicle: Vehicle,
    *,
    gain: float = 1e-3,
    max_thrust_n: float | None = None,
    initial_offset_m: float = 1e4,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Proportional-guidance station-keeping against Earth-Moon gravity."""
    target = np.asarray(target_position_m, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_position_m must have shape (3,)")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    r0 = target + np.array([initial_offset_m, 0.0, 0.0])
    # Local Keplerian velocity (Earth-dominated regime) gives a sensible orbital v.
    rn0 = float(np.linalg.norm(r0))
    v_circ = float(np.sqrt(GM_EARTH / rn0))
    v0 = np.array([0.0, v_circ, 0.0])
    controller = ProportionalGuidance(
        target_position=target,
        gain=gain,
        max_thrust_n=max_thrust_n if max_thrust_n is not None else 1e6,
    )
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_earth_moon_two_body_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * duration_s,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    return _run_with_field(
        name="lunar_stationkeep",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r0,
        v0=v0,
        t_span=(0.0, duration_s),
        n_eval=n_eval,
        background_label="earth_moon_two_body",
        target_metric={
            "target_position_m": target.tolist(),
            "initial_offset_m": initial_offset_m,
            "gain": gain,
        },
    )


def leo_orbit_modification(
    initial_altitude_km: float,
    target_altitude_km: float,
    burn_time_s: float,
    vehicle: Vehicle,
    *,
    thrust_n: float | None = None,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Prograde burn from circular LEO to raise the apoapsis."""
    if initial_altitude_km <= 0 or target_altitude_km <= initial_altitude_km:
        raise ValueError("target_altitude_km must exceed initial_altitude_km > 0")
    r0_mag = R_EARTH + initial_altitude_km * 1000.0
    r0 = np.array([r0_mag, 0.0, 0.0])
    v_circ = float(np.sqrt(GM_EARTH / r0_mag))
    v0 = np.array([0.0, v_circ, 0.0])
    if thrust_n is None:
        # Naive estimate: Δv to circularise at the target altitude × m / burn_time.
        r_target = R_EARTH + target_altitude_km * 1000.0
        v_circ_target = float(np.sqrt(GM_EARTH / r_target))
        delta_v_estimate = abs(v_circ_target - v_circ) + 100.0
        thrust_n = vehicle.mass_kg * delta_v_estimate / burn_time_s
    direction = np.array([0.0, 1.0, 0.0])
    controller = ConstantThrust(magnitude_n=float(thrust_n), direction=direction)
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_earth_central_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * burn_time_s,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    result = _run_with_field(
        name="leo_orbit_modification",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r0,
        v0=v0,
        t_span=(0.0, burn_time_s),
        n_eval=n_eval,
        background_label="earth_central",
        target_metric={
            "initial_altitude_km": initial_altitude_km,
            "target_altitude_km": target_altitude_km,
            "thrust_n": thrust_n,
        },
    )
    # Compute apoapsis from the trajectory (max distance from Earth centre).
    radii = np.linalg.norm(result.trajectory.r, axis=1)
    achieved_apoapsis_km = float((radii.max() - R_EARTH) / 1000.0)
    result.target_metric["achieved_apoapsis_km"] = achieved_apoapsis_km
    return result
