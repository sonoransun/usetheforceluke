"""Long-range mission profiles using the ``control/`` framework.

Five named factories that build a ``ControlledThrustField``, integrate, and
return a ``LongRangeMissionResult`` carrying the trajectory plus thrust /
power histories sampled at the trajectory output times.

Mission catalogue:

1. ``interstellar_brachistochrone`` — constant-acceleration turnaround at midpoint.
2. ``heliocentric_cruise`` — Earth-orbit-relative start, target heliocentric radius.
3. ``lunar_stationkeep`` — proportional guidance against Earth-Moon gravity.
4. ``leo_orbit_modification`` — prograde burn under Earth central gravity.
5. ``event_horizon_stationkeep`` — proportional guidance hovering at
   ``hover_radius_factor · r_s`` of a Schwarzschild black hole. EXCEPTIONALLY
   SPECULATIVE: even with arbitrarily large supplied thrust, the required hover
   thrust diverges as ``R → r_s``.

Each factory returns a ``LongRangeMissionResult`` (extends ``MissionResult``);
the new visualisations in ``viz/control.py`` consume that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scipy.constants import c as C_LIGHT

from usetheforce._schwarzschild import G_NEWTON, gr_hover_factor, schwarzschild_radius
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
from usetheforce.negmass import BondiRunawayPair
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


def event_horizon_stationkeep(
    black_hole_mass_kg: float,
    duration_s: float,
    vehicle: Vehicle,
    *,
    hover_radius_factor: float = 1.10,
    use_gr_hover_correction: bool = False,
    gain: float = 1.0,
    max_thrust_n: float | None = None,
    initial_offset_m: float = 1.0,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Proportional-guidance station-keeping at ``hover_radius_factor · r_s``.

    EXCEPTIONALLY SPECULATIVE. The Schwarzschild gravity is anchored physics
    but the implicit assumption that arbitrary controllable thrust is available
    is not. The required hover thrust grows like ``G M m / R²`` (Newtonian) or
    diverges like ``1/sqrt(1 - r_s/R)`` (GR hover correction); near the horizon
    no realistic vehicle suffices. The mission factory exists to *visualise*
    that shortfall, not to claim feasibility.
    """
    if black_hole_mass_kg <= 0:
        raise ValueError("black_hole_mass_kg must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if hover_radius_factor <= 1.0:
        raise ValueError("hover_radius_factor must exceed 1.0 (must stay outside horizon)")
    r_s = schwarzschild_radius(black_hole_mass_kg)
    target_radius = hover_radius_factor * r_s
    target = np.array([target_radius, 0.0, 0.0])
    r0 = target + np.array([initial_offset_m, 0.0, 0.0])

    def _schwarzschild_bg(r: np.ndarray) -> np.ndarray:
        rn = float(np.linalg.norm(r))
        if rn <= r_s:
            # Inside or on the horizon — return a huge inward pull rather than NaN
            # so the integrator drifts back outward. (We don't claim physics here.)
            return -G_NEWTON * black_hole_mass_kg * r / max(r_s * r_s * r_s, 1e-300)
        return -G_NEWTON * black_hole_mass_kg * r / (rn**3)

    cap = max_thrust_n if max_thrust_n is not None else vehicle.power_w  # thrust at V_REF=1 m/s
    controller = ProportionalGuidance(
        target_position=target,
        gain=gain,
        max_thrust_n=float(cap),
    )
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_schwarzschild_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * duration_s,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    # Newtonian-limit required hover thrust at target_radius — the shortfall headline.
    required_hover_newtonian_n = (
        G_NEWTON * black_hole_mass_kg * vehicle.mass_kg / (target_radius * target_radius)
    )
    required_hover_gr_n = required_hover_newtonian_n * gr_hover_factor(target_radius, r_s)
    required_hover_n = required_hover_gr_n if use_gr_hover_correction else required_hover_newtonian_n
    return _run_with_field(
        name="event_horizon_stationkeep",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r0,
        v0=np.zeros(3),
        t_span=(0.0, duration_s),
        n_eval=n_eval,
        background_label="schwarzschild_central",
        target_metric={
            "black_hole_mass_kg": black_hole_mass_kg,
            "schwarzschild_radius_m": r_s,
            "hover_radius_factor": hover_radius_factor,
            "target_radius_m": target_radius,
            "required_hover_force_newtonian_n": required_hover_newtonian_n,
            "required_hover_force_gr_n": required_hover_gr_n,
            "required_hover_force_n": required_hover_n,
            "supplied_thrust_cap_n": float(cap),
            "use_gr_hover_correction": use_gr_hover_correction,
        },
    )


def event_horizon_stationkeep_with_buffer(
    black_hole_mass_kg: float,
    duration_s: float,
    vehicle: Vehicle,
    *,
    buffer_mass_neg_kg: float,
    buffer_radius_factor: float = 1.05,
    hover_radius_factor: float = 1.10,
    use_gr_hover_correction: bool = False,
    gain: float = 1.0,
    max_thrust_n: float | None = None,
    initial_offset_m: float = 1.0,
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Stationkeep with a *negative-mass buffer* placed between craft and horizon.

    EXCEPTIONALLY SPECULATIVE — see ``event_horizon_stationkeep`` for the
    underlying disclaimers. This variant places a point negative-mass element
    of magnitude ``buffer_mass_neg_kg`` at ``buffer_radius_factor · r_s``,
    strictly between the horizon at ``r_s`` and the craft's hover position at
    ``hover_radius_factor · r_s``. The buffer's *repulsive* gravity (the
    speculative leap; the math is sign-flipped Newtonian) pushes the craft
    outward, partially offsetting the Schwarzschild attraction.

    The background field passed to ``ControlledThrustField`` is the *sum* of
    Schwarzschild attraction and buffer repulsion. The reported
    ``target_metric`` augments the no-buffer version with:

    - ``buffer_position_radius_m`` — where the buffer sits.
    - ``buffer_repulsion_at_craft_n`` — magnitude of the buffer's outward force
      on the craft at the hover position.
    - ``net_required_hover_force_newtonian_n`` — ``max(0, F_BH − F_buffer)``.
    - ``net_required_hover_force_gr_n`` — same with the GR hover factor applied.
    - ``buffer_offset_ratio`` — ``F_buffer / F_BH`` (1.0 = perfect cancellation).
    - ``augmented_shortfall_ratio`` — ``net_required / supplied_cap``.

    A ``buffer_offset_ratio ≥ 1`` means the buffer alone can hold the craft up
    (negative-mass propulsion); ``augmented_shortfall_ratio`` then drops to
    zero. Read both numbers together: a small offset_ratio with a huge
    shortfall is the realistic case.
    """
    if black_hole_mass_kg <= 0:
        raise ValueError("black_hole_mass_kg must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if hover_radius_factor <= 1.0:
        raise ValueError("hover_radius_factor must exceed 1.0 (must stay outside horizon)")
    if not 1.0 < buffer_radius_factor < hover_radius_factor:
        raise ValueError(
            "buffer_radius_factor must lie strictly between 1.0 (the horizon) "
            f"and hover_radius_factor ({hover_radius_factor})"
        )
    if buffer_mass_neg_kg <= 0:
        raise ValueError("buffer_mass_neg_kg must be positive (it names the magnitude)")

    r_s = schwarzschild_radius(black_hole_mass_kg)
    target_radius = hover_radius_factor * r_s
    buffer_radius = buffer_radius_factor * r_s
    target = np.array([target_radius, 0.0, 0.0])
    buffer_pos = np.array([buffer_radius, 0.0, 0.0])
    r0 = target + np.array([initial_offset_m, 0.0, 0.0])
    m_buf = float(buffer_mass_neg_kg)

    def _schwarzschild_plus_buffer_bg(r: np.ndarray) -> np.ndarray:
        rn = float(np.linalg.norm(r))
        # Schwarzschild attraction (same clipped behaviour as the no-buffer factory).
        if rn <= r_s:
            g = -G_NEWTON * black_hole_mass_kg * r / max(r_s * r_s * r_s, 1e-300)
        else:
            g = -G_NEWTON * black_hole_mass_kg * r / (rn**3)
        # Buffer repulsion: positive-mass craft pushed *away* from the negative-mass point.
        d = r - buffer_pos
        d_norm = float(np.linalg.norm(d))
        if d_norm > 0:
            g = g + G_NEWTON * m_buf * d / (d_norm**3)
        return g

    cap = max_thrust_n if max_thrust_n is not None else vehicle.power_w
    controller = ProportionalGuidance(
        target_position=target,
        gain=gain,
        max_thrust_n=float(cap),
    )
    field_obj = ControlledThrustField(
        controller=controller,
        mass_kg=vehicle.mass_kg,
        background=_schwarzschild_plus_buffer_bg,
        power=VehiclePowerState(
            initial_energy_j=vehicle.power_w * duration_s,
            instantaneous_power_w=vehicle.power_w,
        ),
    )
    # Headline thrust budgets at the hover position.
    required_hover_newtonian_n = (
        G_NEWTON * black_hole_mass_kg * vehicle.mass_kg / (target_radius * target_radius)
    )
    delta = target_radius - buffer_radius  # >0 by construction
    buffer_repulsion_at_craft_n = G_NEWTON * m_buf * vehicle.mass_kg / (delta * delta)
    buffer_offset_ratio = buffer_repulsion_at_craft_n / required_hover_newtonian_n
    net_required_newtonian_n = max(
        0.0, required_hover_newtonian_n - buffer_repulsion_at_craft_n
    )
    gr_factor = gr_hover_factor(target_radius, r_s)
    required_hover_gr_n = required_hover_newtonian_n * gr_factor
    # GR correction applies to the Schwarzschild "hover effort" alone; subtract the
    # (Newtonian, flat-spacetime) buffer offset to get the net hover thrust under GR.
    net_required_gr_n = max(0.0, required_hover_gr_n - buffer_repulsion_at_craft_n)
    net_required_n = net_required_gr_n if use_gr_hover_correction else net_required_newtonian_n
    augmented_shortfall = (
        net_required_n / float(cap) if cap > 0 and net_required_n > 0 else 0.0
    )
    return _run_with_field(
        name="event_horizon_stationkeep_with_buffer",
        field_obj=field_obj,
        mass_kg=vehicle.mass_kg,
        r0=r0,
        v0=np.zeros(3),
        t_span=(0.0, duration_s),
        n_eval=n_eval,
        background_label="schwarzschild_plus_negmass_buffer",
        target_metric={
            "black_hole_mass_kg": black_hole_mass_kg,
            "schwarzschild_radius_m": r_s,
            "hover_radius_factor": hover_radius_factor,
            "target_radius_m": target_radius,
            "buffer_radius_factor": buffer_radius_factor,
            "buffer_position_radius_m": buffer_radius,
            "buffer_mass_neg_kg": m_buf,
            "buffer_repulsion_at_craft_n": buffer_repulsion_at_craft_n,
            "required_hover_force_newtonian_n": required_hover_newtonian_n,
            "required_hover_force_gr_n": required_hover_gr_n,
            "net_required_hover_force_newtonian_n": net_required_newtonian_n,
            "net_required_hover_force_gr_n": net_required_gr_n,
            "net_required_hover_force_n": net_required_n,
            "buffer_offset_ratio": buffer_offset_ratio,
            "supplied_thrust_cap_n": float(cap),
            "augmented_shortfall_ratio": augmented_shortfall,
            "use_gr_hover_correction": use_gr_hover_correction,
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


def bondi_runaway_cruise(
    vehicle: Vehicle,
    duration_s: float,
    *,
    m_negative_kg: float,
    separation_m: float,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    n_eval: int = 200,
) -> LongRangeMissionResult:
    """Bondi zero-net-mass pair as the *only* thrust source. EXCEPTIONALLY SPECULATIVE.

    The composite craft starts at rest at the origin and experiences the
    constant body force from ``BondiRunawayPair(m_negative_kg, separation_m,
    craft_mass_kg=vehicle.mass_kg, axis=axis)``. There is no background gravity,
    no power draw, and no controller — the Bondi pair self-accelerates from
    nothing, which is the load-bearing pathology.

    ``target_metric`` carries the headline triple:

    - ``terminal_velocity_mps`` — ``|v(duration_s)|``.
    - ``terminal_velocity_fraction_c`` — same divided by ``c``; flags ``> 1``
      if the classical Bondi solution has overshot ``c`` (it always does
      eventually).
    - ``energy_non_conservation_J`` — kinetic energy gained from rest. With no
      power input, this is the energy created out of nothing by the Bondi
      premise. Test suite asserts this grows monotonically; do *not* "fix"
      the model to make this vanish.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    bondi = BondiRunawayPair(
        m_negative_kg=m_negative_kg,
        separation_m=separation_m,
        craft_mass_kg=vehicle.mass_kg,
        axis=axis,
    )
    traj = integrate(
        bondi,
        mass=vehicle.mass_kg,
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        t_span=(0.0, duration_s),
        n_eval=n_eval,
    )
    constant_F = bondi.force(0.0, np.zeros(3))
    thrust_history = np.tile(constant_F, (traj.t.size, 1))
    # Bondi pair draws no power — keep the reserve full.
    initial_energy = vehicle.power_w * duration_s
    power_history = np.full(traj.t.size, initial_energy, dtype=float)

    v_terminal = float(np.linalg.norm(traj.v[-1]))
    ke_terminal = 0.5 * vehicle.mass_kg * float(np.dot(traj.v[-1], traj.v[-1]))
    ke_initial = 0.5 * vehicle.mass_kg * float(np.dot(traj.v[0], traj.v[0]))
    energy_nonconservation = ke_terminal - ke_initial
    peak_thrust = float(np.linalg.norm(constant_F))
    peak_g = peak_thrust / (vehicle.mass_kg * 9.80665)
    return LongRangeMissionResult(
        name="bondi_runaway_cruise",
        trajectory=traj,
        delta_v_mps=v_terminal,
        burn_time_s=duration_s,
        energy_used_j=0.0,
        peak_g=peak_g,
        thrust_history_n=thrust_history,
        power_history_j=power_history,
        controller_metadata={
            "controller": "bondi_runaway_pair",
            "vehicle_key": vehicle.key,
            **dict(bondi.metadata),
        },
        background="free",
        target_metric={
            "terminal_velocity_mps": v_terminal,
            "terminal_velocity_fraction_c": v_terminal / C_LIGHT,
            "energy_non_conservation_J": energy_nonconservation,
            "self_acceleration_mps2": bondi.self_acceleration_mps2,
            "m_negative_kg": m_negative_kg,
            "separation_m": separation_m,
            "duration_s": duration_s,
        },
    )
