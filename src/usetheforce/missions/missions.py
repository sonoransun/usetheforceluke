"""Canonical missions for the integrated trajectory runs.

Each mission specifies initial conditions, integration timespan, an optional
background gravity, and a thrust axis. The propulsion mechanism is applied as
a *constant body-frame thrust* whose magnitude is the force-from-power
delivered by the adapter at its reference radius — i.e. the snapshot's force
value, treated as available continuously throughout the burn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from usetheforce.missions.adapters import Adapter
from usetheforce.missions.vehicles import Vehicle
from usetheforce.protocol import ForceField
from usetheforce.trajectories import TrajectoryResult, integrate

# Standard gravitational parameter of Earth (m³/s²).
GM_EARTH: float = 3.986004418e14
# Earth radius (m); altitude added to this gives an LEO position.
R_EARTH: float = 6.371e6


@dataclass(frozen=True, slots=True)
class Mission:
    """Initial conditions and environment for an integrated burn."""

    key: str
    description: str
    r0_m: tuple[float, float, float]
    v0_mps: tuple[float, float, float]
    t_span_s: tuple[float, float]
    thrust_axis: tuple[float, float, float]
    background: str  # "free", "constant_g", "central"
    n_eval: int = 200


def _earth_central_gravity(r: np.ndarray) -> np.ndarray:
    """Central inverse-square gravity around Earth at origin."""
    rn = float(np.linalg.norm(r))
    return -GM_EARTH * r / rn**3


def _constant_g(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Constant 1g downward (-z)."""
    return np.array([0.0, 0.0, -9.80665])


def _free(r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.zeros(3)


_BACKGROUNDS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "free": _free,
    "constant_g": _constant_g,
    "central": _earth_central_gravity,
}


# Six canonical missions covering the regime of interest.
MISSIONS: dict[str, Mission] = {
    "free_burn_100s": Mission(
        key="free_burn_100s",
        description="100 s constant-thrust burn in free space",
        r0_m=(0.0, 0.0, 0.0),
        v0_mps=(0.0, 0.0, 0.0),
        t_span_s=(0.0, 100.0),
        thrust_axis=(1.0, 0.0, 0.0),
        background="free",
    ),
    "leo_raise_100s": Mission(
        key="leo_raise_100s",
        description="LEO prograde burn for 100 s under Earth central gravity",
        r0_m=(R_EARTH + 400e3, 0.0, 0.0),
        v0_mps=(0.0, 7669.0, 0.0),  # ~circular @ 400 km
        t_span_s=(0.0, 100.0),
        thrust_axis=(0.0, 1.0, 0.0),
        background="central",
        n_eval=300,
    ),
    "lunar_transfer_3d": Mission(
        key="lunar_transfer_3d",
        description="3-day prograde burn from LEO toward lunar distance",
        r0_m=(R_EARTH + 400e3, 0.0, 0.0),
        v0_mps=(0.0, 10800.0, 0.0),
        t_span_s=(0.0, 3 * 86400.0),
        thrust_axis=(0.0, 1.0, 0.0),
        background="central",
        n_eval=400,
    ),
    "year_burn_free": Mission(
        key="year_burn_free",
        description="1-year constant-thrust burn in free space (interstellar reach)",
        r0_m=(0.0, 0.0, 0.0),
        v0_mps=(0.0, 0.0, 0.0),
        t_span_s=(0.0, 365.25 * 86400.0),
        thrust_axis=(1.0, 0.0, 0.0),
        background="free",
        n_eval=400,
    ),
    "liftoff_60s": Mission(
        key="liftoff_60s",
        description="60 s vertical thrust against 1 g (lift-off feasibility)",
        r0_m=(0.0, 0.0, 0.0),
        v0_mps=(0.0, 0.0, 0.0),
        t_span_s=(0.0, 60.0),
        thrust_axis=(0.0, 0.0, 1.0),
        background="constant_g",
    ),
    "stationkeep_300s": Mission(
        key="stationkeep_300s",
        description="300 s low-thrust orbital stationkeeping in LEO",
        r0_m=(R_EARTH + 400e3, 0.0, 0.0),
        v0_mps=(0.0, 7669.0, 0.0),
        t_span_s=(0.0, 300.0),
        thrust_axis=(1.0, 0.0, 0.0),
        background="central",
        n_eval=300,
    ),
}


@dataclass(slots=True)
class MissionResult:
    """Outcome of an integrated mission run."""

    mission_key: str
    vehicle_key: str
    model_key: str
    trajectory: TrajectoryResult
    delta_v_mps: float
    burn_time_s: float
    energy_j: float
    peak_accel_mps2: float
    peak_g: float
    thrust_n: float
    assumptions: dict[str, Any]


class _ConstantThrustField:
    """Wraps a constant body-frame thrust together with a background gravity field.

    The propulsion model is reduced to its peak ``|F|`` (from the adapter at
    its reference radius); that thrust is applied along ``axis`` for the full
    burn. Background gravity is added as ``mass · g(r)``.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        thrust_n: float,
        axis: tuple[float, float, float],
        vehicle: Vehicle,
        background: Callable[[np.ndarray], np.ndarray],
        underlying_metadata: dict[str, Any],
    ) -> None:
        a = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(a))
        if norm == 0:
            raise ValueError("thrust axis must be non-zero")
        self._thrust_vec = (thrust_n / norm) * a
        self._mass = vehicle.mass_kg
        self._bg = background
        self.metadata = {
            "avenue": underlying_metadata.get("avenue", "missions"),
            "model": f"constant-thrust wrapper around {underlying_metadata.get('model', '?')}",
            "speculative": underlying_metadata.get("speculative", True),
            "citation": "constant-thrust idealization of the underlying model at its reference radius",
        }

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._thrust_vec + self._mass * self._bg(np.asarray(r, dtype=float))

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        return None


def run_mission(
    vehicle: Vehicle,
    model_key: str,
    adapter: Adapter,
    mission: Mission,
) -> MissionResult:
    """Run an integrated mission with constant-thrust idealization of the model."""
    adapter_result = adapter(vehicle, vehicle.power_w)
    if not adapter_result.applicable:
        raise ValueError(
            f"model {model_key!r} is not applicable for free-flight propulsion: "
            f"{adapter_result.reason}"
        )
    r_probe = np.array([adapter_result.r_ref_m, 0.0, 0.0])
    thrust_n = float(np.linalg.norm(adapter_result.field.force(0.0, r_probe)))
    bg = _BACKGROUNDS[mission.background]
    field: ForceField = _ConstantThrustField(
        thrust_n=thrust_n,
        axis=mission.thrust_axis,
        vehicle=vehicle,
        background=bg,
        underlying_metadata=adapter_result.field.metadata,
    )
    traj = integrate(
        field,
        mass=vehicle.mass_kg,
        r0=list(mission.r0_m),
        v0=list(mission.v0_mps),
        t_span=mission.t_span_s,
        n_eval=mission.n_eval,
    )
    v0 = np.asarray(mission.v0_mps, dtype=float)
    dv = float(np.linalg.norm(traj.v[-1] - v0))
    burn_time = mission.t_span_s[1] - mission.t_span_s[0]
    energy = vehicle.power_w * burn_time
    peak_accel = thrust_n / vehicle.mass_kg
    return MissionResult(
        mission_key=mission.key,
        vehicle_key=vehicle.key,
        model_key=model_key,
        trajectory=traj,
        delta_v_mps=dv,
        burn_time_s=burn_time,
        energy_j=energy,
        peak_accel_mps2=peak_accel,
        peak_g=peak_accel / 9.80665,
        thrust_n=thrust_n,
        assumptions=dict(adapter_result.assumptions),
    )
