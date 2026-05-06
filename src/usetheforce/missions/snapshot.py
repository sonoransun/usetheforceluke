"""Static (vehicle × model) snapshot evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from usetheforce.missions.adapters import Adapter, AdapterResult
from usetheforce.missions.vehicles import Vehicle

G_EARTH: float = 9.80665  # m/s²


# Long-range probe distance used to differentiate models on falloff laws.
R_FAR_M: float = 1000.0


@dataclass(slots=True)
class Snapshot:
    """Capability snapshot for one (vehicle, model) cell."""

    vehicle_key: str
    model_key: str
    applicable: bool
    reason: str
    force_n: float
    accel_mps2: float
    g_load: float
    twr_1g: float
    liftoff_capable: bool
    energy_per_dv_jspm: float  # J·s/m, = power / accel for constant-thrust idealization
    range_scale_m: float
    force_at_1km_n: float  # |F| at 1 km — exposes the model's falloff law
    falloff_ratio: float  # force_at_1km / force_n (lower = faster decay with distance)
    assumptions: dict


SnapshotMatrix = dict[str, dict[str, Snapshot]]


def _evaluate_cell(
    vehicle: Vehicle,
    model_key: str,
    adapter: Adapter,
) -> Snapshot:
    result: AdapterResult = adapter(vehicle, vehicle.power_w)
    r_probe = np.array([result.r_ref_m, 0.0, 0.0])
    f_vec = result.field.force(0.0, r_probe)
    f_mag = float(np.linalg.norm(f_vec))
    # Force at r=1 km exposes the model's falloff law (Gaussian/1-over-r²/Yukawa).
    r_far = np.array([R_FAR_M, 0.0, 0.0])
    try:
        f_far = float(np.linalg.norm(result.field.force(0.0, r_far)))
    except (ValueError, ZeroDivisionError):
        f_far = 0.0
    accel = f_mag / vehicle.mass_kg
    g_load = accel / G_EARTH
    twr = f_mag / (vehicle.mass_kg * G_EARTH)
    energy_per_dv = vehicle.power_w / accel if accel > 0 else float("inf")
    return Snapshot(
        vehicle_key=vehicle.key,
        model_key=model_key,
        applicable=result.applicable,
        reason=result.reason,
        force_n=f_mag,
        accel_mps2=accel,
        g_load=g_load,
        twr_1g=twr,
        liftoff_capable=twr >= 1.0 and result.applicable,
        energy_per_dv_jspm=energy_per_dv,
        range_scale_m=result.r_ref_m,
        force_at_1km_n=f_far,
        falloff_ratio=f_far / f_mag if f_mag > 0 else 0.0,
        assumptions=dict(result.assumptions),
    )


def evaluate_snapshot(
    vehicles: Mapping[str, Vehicle],
    adapters: Mapping[str, Adapter],
) -> SnapshotMatrix:
    """Evaluate every (vehicle, model) cell. Returns ``matrix[vehicle_key][model_key]``."""
    matrix: SnapshotMatrix = {}
    for vk, vehicle in vehicles.items():
        row: dict[str, Snapshot] = {}
        for mk, adapter in adapters.items():
            row[mk] = _evaluate_cell(vehicle, mk, adapter)
        matrix[vk] = row
    return matrix
