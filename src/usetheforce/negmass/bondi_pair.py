"""Bondi zero-net-mass runaway pair. EXCEPTIONALLY SPECULATIVE.

A rigid composite of a positive-mass element ``+|m_neg|`` and a negative-mass
element ``-|m_neg|`` separated by ``d`` accelerates indefinitely with constant
proper acceleration::

    a = G · |m_neg| / d²

in the direction from the negative-mass element toward the positive-mass
element (the "axis" of the pair, as specified at construction). The composite
craft of total positive payload mass ``m_craft`` therefore experiences a
constant body force::

    F = m_craft · a · n̂

independent of probe position. The whole point of the model — Forward's 2015
"propulsion without fuel" pitch — is that this force does not require any
exhaust, working fluid, or external field. It also violates conservation of
energy and momentum for the centre of mass; the test suite asserts this
explicitly so that nobody "fixes" the model later by silently zeroing the
runaway.

``applicable_for_trajectory`` is ``False`` because the ``force(t, r)`` is
constant in ``r`` — this is a rigid body force on the composite, not a probe
field. Trajectory integration *is* meaningful (you get a uniformly accelerating
craft), but the field is not a ``-∇U`` quantity for a probe at arbitrary
position; the mission adapter for this model accordingly exposes
``applicable=False``, just like the parallel-plate Casimir adapter.

References
----------
- Bondi, H. (1957). Rev. Mod. Phys. 29:423.
- Forward, R. (2015). J. Propulsion & Power 6:1.
- Loeb, A. (2024). Medium.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce._negmass import (
    bondi_self_acceleration,
    validate_neg_mass,
    validate_separation,
)


class BondiRunawayPair:
    """Constant body force on a composite craft from a zero-net-mass pair."""

    metadata: dict[str, Any]

    def __init__(
        self,
        m_negative_kg: float,
        separation_m: float,
        craft_mass_kg: float,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        validate_neg_mass(m_negative_kg)
        validate_separation(separation_m)
        if craft_mass_kg <= 0:
            raise ValueError("craft_mass_kg must be positive")
        axis_arr = np.asarray(axis, dtype=float)
        if axis_arr.shape != (3,):
            raise ValueError("axis must have shape (3,)")
        norm = float(np.linalg.norm(axis_arr))
        if norm == 0:
            raise ValueError("axis must be non-zero")
        self._axis: np.ndarray = axis_arr / norm
        self._m_neg = float(m_negative_kg)
        self._d = float(separation_m)
        self._mc = float(craft_mass_kg)
        self._a = bondi_self_acceleration(self._m_neg, self._d)
        # Force on the composite craft along +axis. Cached at construction.
        self._force_vec: np.ndarray = self._mc * self._a * self._axis
        self.metadata = {
            "avenue": "negmass",
            "model": (
                f"Bondi runaway pair (|m_neg|={self._m_neg:g} kg, d={self._d:g} m, "
                f"a={self._a:g} m/s²)"
            ),
            "speculative": True,
            "speculative_components": ["m_negative_kg", "negative_mass_premise"],
            "applicable_for_trajectory": False,
            "citation": "Bondi 1957 Rev. Mod. Phys. 29:423; Forward 2015 propulsion claim",
            "self_acceleration_mps2": self._a,
        }

    @property
    def self_acceleration_mps2(self) -> float:
        return self._a

    @property
    def axis(self) -> np.ndarray:
        return self._axis.copy()

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Constant body force ``m_craft · a · n̂``; ignores position and time."""
        return self._force_vec

    def potential(self, r: np.ndarray) -> float | None:  # noqa: ARG002
        """Linear potential ``-F · r`` is mathematically defined but physically
        bogus for a self-propulsive pair (the work-energy theorem does not apply
        — that is the Bondi pathology). We return ``None`` to keep
        ``TrajectoryResult.total_energy`` from producing a misleading value."""
        return None
