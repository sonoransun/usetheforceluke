"""Algorithmic thrust control for trajectories.

The ``ThrustController`` protocol decouples *what thrust to apply* from
*how to apply it*: controllers compute thrust as a function of (t, r, v,
power_state); the ``ControlledThrustField`` wraps a controller into a
``ForceField`` that the integrator can consume.

Public surface:

- ``ThrustController`` — protocol
- ``VehiclePowerState`` — onboard energy bookkeeping
- ``ControlledThrustField`` — wraps a controller as a ``ForceField``
- Concrete controllers: ``ConstantThrust``, ``ScheduledThrust``,
  ``ConstantAcceleration``, ``BrachistochroneTransit``,
  ``ProportionalGuidance``, ``BangBangAltitude``
- Optimisation helpers: ``solve_min_time``, ``solve_min_dv``
"""

from usetheforce.control.controllers import (
    BangBangAltitude,
    BrachistochroneTransit,
    ConstantAcceleration,
    ConstantThrust,
    ProportionalGuidance,
    ScheduledThrust,
    ThrustController,
)
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.optimisation import (
    OptimisationResult,
    solve_min_dv,
    solve_min_time,
)
from usetheforce.control.power import VehiclePowerState

__all__ = [
    "BangBangAltitude",
    "BrachistochroneTransit",
    "ConstantAcceleration",
    "ConstantThrust",
    "ControlledThrustField",
    "OptimisationResult",
    "ProportionalGuidance",
    "ScheduledThrust",
    "ThrustController",
    "VehiclePowerState",
    "solve_min_dv",
    "solve_min_time",
]
