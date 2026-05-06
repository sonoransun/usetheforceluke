"""Propulsion evaluation across vehicle scales.

A vehicle catalogue (CubeSat → city ship), per-model power→force adapters that
make speculative coupling explicit at the call site, a static snapshot
evaluator producing the (vehicle × model) matrix, and an ODE-integrated
mission runner for the most propulsively interesting pairs.

All speculative parameters are *stated* alongside every result; nothing in
this subpackage claims real efficiency.
"""

from usetheforce.missions.adapters import ALL_ADAPTERS, AdapterResult
from usetheforce.missions.missions import MISSIONS, MissionResult, run_mission
from usetheforce.missions.snapshot import Snapshot, SnapshotMatrix, evaluate_snapshot
from usetheforce.missions.vehicles import VEHICLES, Vehicle, power_budget

__all__ = [
    "ALL_ADAPTERS",
    "MISSIONS",
    "VEHICLES",
    "AdapterResult",
    "MissionResult",
    "Snapshot",
    "SnapshotMatrix",
    "Vehicle",
    "evaluate_snapshot",
    "power_budget",
    "run_mission",
]
