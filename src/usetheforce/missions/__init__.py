"""Propulsion evaluation across vehicle scales.

A vehicle catalogue (CubeSat → city ship), per-model power→force adapters that
make speculative coupling explicit at the call site, a static snapshot
evaluator producing the (vehicle × model) matrix, and an ODE-integrated
mission runner for the most propulsively interesting pairs.

All speculative parameters are *stated* alongside every result; nothing in
this subpackage claims real efficiency.
"""

from usetheforce.missions.adapters import ALL_ADAPTERS, AdapterResult
from usetheforce.missions.long_range import (
    LongRangeMissionResult,
    bondi_runaway_cruise,
    event_horizon_stationkeep,
    event_horizon_stationkeep_with_buffer,
    heliocentric_cruise,
    interstellar_brachistochrone,
    leo_orbit_modification,
    lunar_stationkeep,
)
from usetheforce.missions.missions import MISSIONS, MissionResult, run_mission
from usetheforce.missions.snapshot import Snapshot, SnapshotMatrix, evaluate_snapshot
from usetheforce.missions.vehicles import VEHICLES, Vehicle, power_budget

__all__ = [
    "ALL_ADAPTERS",
    "MISSIONS",
    "VEHICLES",
    "AdapterResult",
    "LongRangeMissionResult",
    "MissionResult",
    "Snapshot",
    "SnapshotMatrix",
    "Vehicle",
    "bondi_runaway_cruise",
    "evaluate_snapshot",
    "event_horizon_stationkeep",
    "event_horizon_stationkeep_with_buffer",
    "heliocentric_cruise",
    "interstellar_brachistochrone",
    "leo_orbit_modification",
    "lunar_stationkeep",
    "power_budget",
    "run_mission",
]
