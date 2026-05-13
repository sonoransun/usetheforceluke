"""Speculative counter-drive that cancels a supplied gravitational background.

EXCEPTIONALLY SPECULATIVE. No known mechanism couples to the local
gravitational acceleration ``g(r)`` to produce an equal-and-opposite reaction
force on the vehicle's centre of mass. This model exists solely to quantify
the propulsion *shortfall* near a Schwarzschild horizon: given a power budget,
how much of the required hover thrust can a "perfect counter-drive" actually
supply?

The force is::

    F_drive(r) = -η · m_probe · g_background(r)

with conversion efficiency ``η ∈ [0, 1]`` and an optional magnitude cap
``max_thrust_n``. ``η=0`` is a passive vehicle; ``η=1`` is exact local cancellation.

``potential()`` returns ``None``: a generic supplied ``g`` may be
non-conservative, so we decline to claim a scalar potential — same escape hatch
as ``AntimatterCounterGravity``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from usetheforce.blackhole.metric import SchwarzschildGravity
from usetheforce.protocol import ForceField

# Type alias: a background gravitational *acceleration* field g(r).
BackgroundG = Callable[[np.ndarray], np.ndarray]


def _gravity_callable_from_field(field: ForceField, probe_mass_kg: float) -> BackgroundG:
    """Wrap any ``ForceField`` so it returns acceleration (force / probe_mass)."""

    def g(r: np.ndarray) -> np.ndarray:
        f = np.asarray(field.force(0.0, np.asarray(r, dtype=float)), dtype=float)
        return f / probe_mass_kg

    return g


class BlackHoleCounterDrive:
    """Counter-thrust drive: ``F = -η · m · g(r)`` with optional thrust cap.

    Parameters
    ----------
    probe_mass_kg:
        Probe mass (kg). Force is mass-weighted to mirror ``-m · g``.
    background_g:
        Either a callable ``r -> g(r)`` returning gravitational acceleration
        (m/s²) or a ``ForceField`` representing the background gravity (the
        wrapper divides by ``probe_mass_kg`` to recover ``g``).
    efficiency:
        ``η ∈ [0, 1]``. ``0`` → passive (zero counter-thrust); ``1`` → exact
        local cancellation.
    max_thrust_n:
        Optional cap on the thrust magnitude. ``None`` means uncapped.
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        probe_mass_kg: float,
        background_g: BackgroundG | ForceField,
        efficiency: float = 1.0,
        max_thrust_n: float | None = None,
    ) -> None:
        if probe_mass_kg <= 0:
            raise ValueError("probe_mass_kg must be positive")
        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("efficiency must lie in [0, 1]")
        if max_thrust_n is not None and max_thrust_n <= 0:
            raise ValueError("max_thrust_n must be positive (or None)")
        self._mp = float(probe_mass_kg)
        self._eps = float(efficiency)
        self._cap = float(max_thrust_n) if max_thrust_n is not None else None
        if isinstance(background_g, ForceField):
            self._g = _gravity_callable_from_field(background_g, self._mp)
            self._bg_source = "ForceField"
        elif callable(background_g):
            self._g = background_g
            self._bg_source = "callable"
        else:
            raise TypeError("background_g must be a ForceField or a callable")
        self.metadata = {
            "avenue": "blackhole",
            "model": f"speculative counter-drive (η={self._eps})",
            "speculative": True,
            "speculative_components": [
                "efficiency",
                "counter_drive_mechanism",
                "max_thrust_n",
            ],
            "applicable_for_trajectory": True,
            "citation": (
                "EXCEPTIONALLY SPECULATIVE: no known mechanism couples directly to "
                "local gravitational acceleration to produce an equal-and-opposite "
                "force; this drive is included solely to quantify the propulsion "
                "shortfall near r_s."
            ),
            "background_source": self._bg_source,
            "max_thrust_n": self._cap,
        }

    @classmethod
    def from_schwarzschild(
        cls,
        mass_kg: float,
        probe_mass_kg: float,
        *,
        center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_gr_hover_correction: bool = False,
        horizon_softening_m: float = 0.0,
        efficiency: float = 1.0,
        max_thrust_n: float | None = None,
    ) -> BlackHoleCounterDrive:
        """Convenience: build the counter-drive against a Schwarzschild background."""
        bh = SchwarzschildGravity(
            mass_kg=mass_kg,
            probe_mass_kg=probe_mass_kg,
            center=center,
            use_gr_hover_correction=use_gr_hover_correction,
            horizon_softening_m=horizon_softening_m,
        )
        return cls(
            probe_mass_kg=probe_mass_kg,
            background_g=bh,
            efficiency=efficiency,
            max_thrust_n=max_thrust_n,
        )

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        g = np.asarray(self._g(np.asarray(r, dtype=float)), dtype=float)
        F = -self._eps * self._mp * g
        if self._cap is not None:
            mag = float(np.linalg.norm(F))
            if mag > self._cap and mag > 0.0:
                F = F * (self._cap / mag)
        return F

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        return None
