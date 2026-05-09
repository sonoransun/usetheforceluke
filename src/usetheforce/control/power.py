"""Onboard energy bookkeeping for closed-loop control.

The integrator's RHS does not have access to a true wall-clock dt — it sees
the current ``t`` only. ``VehiclePowerState`` is therefore *state-driven*: the
caller (``ControlledThrustField``) is responsible for telling it ``consume(dt,
power)`` once per step. With ``solve_ivp`` this happens between successive RHS
calls and the dt heuristic is the difference of consecutive ``t`` values.

Conventions:
- ``initial_energy_j`` is the onboard energy reserve (J).
- ``instantaneous_power_w`` is the maximum continuous power output the vehicle
  can sustain (W). When energy is exhausted, ``available_power`` returns 0
  and controllers must throttle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VehiclePowerState:
    """Tracks energy reserve consumption over a mission."""

    initial_energy_j: float
    instantaneous_power_w: float
    elapsed_energy_j: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_energy_j <= 0:
            raise ValueError("initial_energy_j must be positive")
        if self.instantaneous_power_w <= 0:
            raise ValueError("instantaneous_power_w must be positive")
        if self.elapsed_energy_j < 0:
            raise ValueError("elapsed_energy_j must be non-negative")

    @property
    def remaining_energy_j(self) -> float:
        return max(0.0, self.initial_energy_j - self.elapsed_energy_j)

    def is_depleted(self) -> bool:
        return self.elapsed_energy_j >= self.initial_energy_j

    def available_power(self) -> float:
        """Maximum power output right now (W). Zero when reserve is depleted."""
        return 0.0 if self.is_depleted() else self.instantaneous_power_w

    def consume(self, dt: float, power_w: float) -> None:
        """Record dt seconds of operation at ``power_w`` watts."""
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if power_w < 0:
            raise ValueError("power_w must be non-negative")
        self.elapsed_energy_j += dt * power_w

    def reset(self) -> None:
        """Restore the reserve to its initial value (used between optimiser iterations)."""
        self.elapsed_energy_j = 0.0
