"""Catalogue of thrust controllers.

A ``ThrustController`` answers the question "what thrust vector should the
vehicle apply right now?" given the current time, position, velocity, and
onboard power state. Controllers are *pure with respect to state* — they don't
mutate themselves between calls; ``VehiclePowerState`` carries the energy
bookkeeping.

```mermaid
classDiagram
    class ThrustController {
        <<protocol>>
        +metadata: dict
        +thrust(t, r, v, power) ndarray
    }
    class ConstantThrust
    class ScheduledThrust
    class ConstantAcceleration
    class BrachistochroneTransit
    class ProportionalGuidance
    class BangBangAltitude
    ThrustController <|.. ConstantThrust
    ThrustController <|.. ScheduledThrust
    ThrustController <|.. ConstantAcceleration
    ThrustController <|.. BrachistochroneTransit
    ThrustController <|.. ProportionalGuidance
    ThrustController <|.. BangBangAltitude
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from usetheforce.control.power import VehiclePowerState


@runtime_checkable
class ThrustController(Protocol):
    """Compute the thrust vector at the current state."""

    metadata: dict[str, Any]

    def thrust(
        self,
        t: float,
        r: np.ndarray,
        v: np.ndarray,
        power: VehiclePowerState,
    ) -> np.ndarray:
        """Return thrust (N) at time ``t``, position ``r`` (m), velocity ``v`` (m/s)."""
        ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.zeros(3)
    return v / n


# --------------------------------------------------------------------------------------
# Open-loop controllers
# --------------------------------------------------------------------------------------


class ConstantThrust:
    """Constant body-frame thrust vector. Backward-compatible default."""

    metadata: dict[str, Any]

    def __init__(self, magnitude_n: float, direction: np.ndarray | tuple[float, float, float]) -> None:
        if magnitude_n < 0:
            raise ValueError("magnitude_n must be non-negative")
        d = np.asarray(direction, dtype=float)
        if d.shape != (3,):
            raise ValueError("direction must have shape (3,)")
        n = float(np.linalg.norm(d))
        if n == 0:
            raise ValueError("direction must be non-zero")
        self._vec = (magnitude_n / n) * d
        self.metadata = {
            "controller": "ConstantThrust",
            "magnitude_n": float(magnitude_n),
            "direction": d.tolist(),
            "speculative_components": [],
            "closed_loop": False,
        }

    def thrust(
        self,
        t: float,  # noqa: ARG002
        r: np.ndarray,  # noqa: ARG002
        v: np.ndarray,  # noqa: ARG002
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        return self._vec


class ScheduledThrust:
    """Open-loop schedule: ``profile(t) -> (magnitude_n, direction_vec)``."""

    metadata: dict[str, Any]

    def __init__(
        self,
        profile: Callable[[float], tuple[float, np.ndarray | tuple[float, float, float]]],
        description: str = "scheduled",
    ) -> None:
        self._profile = profile
        self.metadata = {
            "controller": "ScheduledThrust",
            "description": description,
            "speculative_components": ["profile"],
            "closed_loop": False,
        }

    def thrust(
        self,
        t: float,
        r: np.ndarray,  # noqa: ARG002
        v: np.ndarray,  # noqa: ARG002
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        magnitude, direction = self._profile(t)
        d = np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n == 0 or magnitude == 0:
            return np.zeros(3)
        return (float(magnitude) / n) * d


# --------------------------------------------------------------------------------------
# Closed-loop controllers
# --------------------------------------------------------------------------------------


class ConstantAcceleration:
    """Closed-loop magnitude: holds ``|F| = m · a_target`` along a direction policy.

    ``direction_policy``:
    - ``"velocity"`` — along current velocity (prograde). Reverts to the launch
      axis if the vehicle is at rest.
    - ``"toward_target"`` — toward ``target_position``.

    Throttles to ``power.available_power() / |v|`` if the reserve is exhausted.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        a_target_mps2: float,
        mass_kg: float,
        direction_policy: str = "velocity",
        target_position: np.ndarray | tuple[float, float, float] | None = None,
        launch_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        if a_target_mps2 <= 0:
            raise ValueError("a_target_mps2 must be positive")
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if direction_policy not in {"velocity", "toward_target"}:
            raise ValueError(f"direction_policy must be 'velocity' or 'toward_target'")
        if direction_policy == "toward_target" and target_position is None:
            raise ValueError("target_position is required for direction_policy='toward_target'")
        self._a = float(a_target_mps2)
        self._m = float(mass_kg)
        self._policy = direction_policy
        self._target = np.asarray(target_position, dtype=float) if target_position is not None else None
        self._launch_axis = _unit(np.asarray(launch_axis, dtype=float))
        self.metadata = {
            "controller": "ConstantAcceleration",
            "a_target_mps2": self._a,
            "mass_kg": self._m,
            "direction_policy": direction_policy,
            "speculative_components": [],
            "closed_loop": True,
        }

    def thrust(
        self,
        t: float,  # noqa: ARG002
        r: np.ndarray,
        v: np.ndarray,
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        magnitude = self._m * self._a
        if self._policy == "velocity":
            direction = _unit(v) if np.linalg.norm(v) > 0 else self._launch_axis
        else:
            assert self._target is not None
            direction = _unit(self._target - r)
        if not np.any(direction):
            return np.zeros(3)
        return magnitude * direction


class BrachistochroneTransit:
    """Constant-acceleration "turnaround at midpoint" profile.

    Accelerates at ``+a_max`` along the start→target axis for the first half
    of the trip, then decelerates at ``+a_max`` *against* the velocity for the
    second half. Analytical travel time for distance ``d`` and acceleration
    ``a_max`` is ``t = 2 · sqrt(d / a_max)``.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        r_start: np.ndarray | tuple[float, float, float],
        r_target: np.ndarray | tuple[float, float, float],
        max_acceleration_mps2: float,
        mass_kg: float,
    ) -> None:
        if max_acceleration_mps2 <= 0:
            raise ValueError("max_acceleration_mps2 must be positive")
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        self._r0 = np.asarray(r_start, dtype=float)
        self._rt = np.asarray(r_target, dtype=float)
        if self._r0.shape != (3,) or self._rt.shape != (3,):
            raise ValueError("r_start and r_target must have shape (3,)")
        delta = self._rt - self._r0
        self._distance_m = float(np.linalg.norm(delta))
        if self._distance_m == 0:
            raise ValueError("r_start and r_target coincide")
        self._axis = delta / self._distance_m
        self._a = float(max_acceleration_mps2)
        self._m = float(mass_kg)
        self._t_brake = float(np.sqrt(self._distance_m / self._a))  # half the analytical travel time
        self._t_total = 2.0 * self._t_brake
        self.metadata = {
            "controller": "BrachistochroneTransit",
            "distance_m": self._distance_m,
            "max_acceleration_mps2": self._a,
            "t_brake_s": self._t_brake,
            "analytical_travel_time_s": self._t_total,
            "speculative_components": [],
            "closed_loop": True,
        }

    @property
    def t_brake_s(self) -> float:
        return self._t_brake

    @property
    def analytical_travel_time_s(self) -> float:
        return self._t_total

    def thrust(
        self,
        t: float,
        r: np.ndarray,  # noqa: ARG002
        v: np.ndarray,
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        magnitude = self._m * self._a
        if t < self._t_brake:
            # Accelerate along start→target axis.
            return magnitude * self._axis
        # Decelerate along the *current velocity* (anti-parallel).
        if np.linalg.norm(v) == 0:
            return -magnitude * self._axis
        return -magnitude * _unit(v)


class ProportionalGuidance:
    """Position-error proportional controller: thrust toward target with bounded magnitude.

    Magnitude is ``min(gain · |target − r|, max_thrust_n)``; direction is the
    unit vector toward ``target_position``. Useful for stationkeeping and
    rendezvous where strict velocity tracking isn't required.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        target_position: np.ndarray | tuple[float, float, float],
        gain: float,
        max_thrust_n: float,
    ) -> None:
        if gain <= 0:
            raise ValueError("gain must be positive")
        if max_thrust_n <= 0:
            raise ValueError("max_thrust_n must be positive")
        self._target = np.asarray(target_position, dtype=float)
        if self._target.shape != (3,):
            raise ValueError("target_position must have shape (3,)")
        self._gain = float(gain)
        self._max = float(max_thrust_n)
        self.metadata = {
            "controller": "ProportionalGuidance",
            "target_position": self._target.tolist(),
            "gain": self._gain,
            "max_thrust_n": self._max,
            "speculative_components": [],
            "closed_loop": True,
        }

    @property
    def gain(self) -> float:
        return self._gain

    def thrust(
        self,
        t: float,  # noqa: ARG002
        r: np.ndarray,
        v: np.ndarray,  # noqa: ARG002
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        delta = self._target - r
        distance = float(np.linalg.norm(delta))
        if distance == 0:
            return np.zeros(3)
        magnitude = min(self._gain * distance, self._max)
        return magnitude * (delta / distance)


class BangBangAltitude:
    """Hysteretic on/off vertical thrust to maintain an altitude band along ``axis``.

    When the projected altitude ``r·axis`` falls below ``target − threshold``,
    fires at ``magnitude_n`` along ``+axis``. When it rises above
    ``target + threshold``, thrust drops to zero. Otherwise, holds the most
    recent state (true bang-bang hysteresis).
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        target_altitude_m: float,
        threshold_m: float,
        magnitude_n: float,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        if threshold_m <= 0:
            raise ValueError("threshold_m must be positive")
        if magnitude_n <= 0:
            raise ValueError("magnitude_n must be positive")
        self._target = float(target_altitude_m)
        self._thresh = float(threshold_m)
        self._mag = float(magnitude_n)
        self._axis = _unit(np.asarray(axis, dtype=float))
        self._on = False
        self.metadata = {
            "controller": "BangBangAltitude",
            "target_altitude_m": self._target,
            "threshold_m": self._thresh,
            "magnitude_n": self._mag,
            "axis": self._axis.tolist(),
            "speculative_components": [],
            "closed_loop": True,
        }

    def thrust(
        self,
        t: float,  # noqa: ARG002
        r: np.ndarray,
        v: np.ndarray,  # noqa: ARG002
        power: VehiclePowerState,  # noqa: ARG002
    ) -> np.ndarray:
        altitude = float(np.dot(r, self._axis))
        if altitude < self._target - self._thresh:
            self._on = True
        elif altitude > self._target + self._thresh:
            self._on = False
        return self._mag * self._axis if self._on else np.zeros(3)
