"""``ControlledThrustField`` — wraps a ``ThrustController`` as a ``ForceField``.

The integrator's RHS sees only ``(t, r)``; this class reconstructs an
estimate of ``v`` from finite differences between successive RHS calls and
forwards everything (plus a power state) to the controller. Background
gravity is added on top and treated as conservative.

Energy bookkeeping: each RHS evaluation debits ``dt × |F·v|`` from the
``VehiclePowerState`` reserve (mechanical-power approximation). When the
reserve is depleted, ``power.available_power()`` returns 0 and well-behaved
controllers throttle accordingly.

```mermaid
sequenceDiagram
    participant solve_ivp
    participant ControlledThrustField
    participant ThrustController
    participant VehiclePowerState
    solve_ivp->>ControlledThrustField: force(t, r)
    ControlledThrustField->>ControlledThrustField: v ≈ (r − r_prev)/(t − t_prev)
    ControlledThrustField->>ThrustController: thrust(t, r, v, power)
    ThrustController-->>ControlledThrustField: F_thrust
    ControlledThrustField->>VehiclePowerState: consume(dt, P)
    ControlledThrustField-->>solve_ivp: F_thrust + m·bg(r)
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from usetheforce.control.controllers import ThrustController
from usetheforce.control.power import VehiclePowerState


class ControlledThrustField:
    """Composes a thrust controller with a background gravity into a ``ForceField``."""

    metadata: dict[str, Any]

    def __init__(
        self,
        controller: ThrustController,
        mass_kg: float,
        background: Callable[[np.ndarray], np.ndarray],
        power: VehiclePowerState | None = None,
    ) -> None:
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        self._controller = controller
        self._mass = float(mass_kg)
        self._bg = background
        # If no power state supplied, inject a "infinite-energy" stub so
        # controllers that consult available_power() don't see zero.
        self._power = power if power is not None else VehiclePowerState(
            initial_energy_j=1e300,
            instantaneous_power_w=1e300,
        )
        # State for finite-difference velocity estimation.
        self._t_prev: float | None = None
        self._r_prev: np.ndarray | None = None
        # History buffers — populated externally after integration completes
        # for the reporting layer (``LongRangeMissionResult``).
        self.thrust_log: list[tuple[float, np.ndarray]] = []
        self.metadata = {
            "model": f"controlled-thrust ({controller.metadata.get('controller', '?')})",
            "speculative": True,
            "speculative_components": list(
                controller.metadata.get("speculative_components", [])
            ),
            "controller": dict(controller.metadata),
            "mass_kg": self._mass,
        }

    @property
    def power(self) -> VehiclePowerState:
        return self._power

    @property
    def controller(self) -> ThrustController:
        return self._controller

    def reset(self) -> None:
        """Clear FD-velocity state and any log between repeated runs."""
        self._t_prev = None
        self._r_prev = None
        self.thrust_log.clear()
        self._power.reset()

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        if self._t_prev is None or self._r_prev is None:
            v_est = np.zeros(3)
            dt = 0.0
        else:
            dt = max(t - self._t_prev, 0.0)
            v_est = (r_arr - self._r_prev) / dt if dt > 0 else np.zeros(3)
        self._t_prev = t
        self._r_prev = r_arr.copy()

        thrust_vec = np.asarray(
            self._controller.thrust(t, r_arr, v_est, self._power), dtype=float
        )
        # Power-limit cap: |F · v| is the mechanical power being delivered;
        # if greater than available, scale thrust magnitude proportionally.
        avail = self._power.available_power()
        if avail < float("inf") and dt > 0:
            mech_p = float(abs(np.dot(thrust_vec, v_est)))
            if mech_p > avail and avail > 0:
                thrust_vec = thrust_vec * (avail / mech_p)
                mech_p = avail
            self._power.consume(dt, mech_p)
        # Log thrust for later reporting.
        self.thrust_log.append((t, thrust_vec.copy()))
        # Background gravity (mass-weighted force).
        return thrust_vec + self._mass * self._bg(r_arr)

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        # Controlled fields are not generally conservative — they inject energy
        # via thrust. Total-energy bookkeeping is tracked via VehiclePowerState
        # rather than ``-∇U``.
        return None
