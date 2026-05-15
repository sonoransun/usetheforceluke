"""Sum of two or more ``ForceField`` instances as a single ``ForceField``.

The framework defines exactly one structural seam (``protocol.ForceField``):
every avenue produces such an object, ``trajectories.integrate`` consumes one.
Until now, *combining* two of them — e.g. bolting a Bondi-pair negative-mass
appendage onto an existing antimatter-graviton drive on the same craft —
required a hand-coded composite class. ``CompositeField`` fills that gap with
a single avenue-agnostic class that itself satisfies the protocol.

Semantics
---------

- ``force(t, r)`` returns the vector sum of each component's ``force(t, r)``.
- ``potential(r)`` returns the scalar sum if *every* component provides a
  finite potential. If any component returns ``None`` (non-conservative
  field, e.g. ``AntimatterCounterGravity``), the composite also returns
  ``None`` — mirroring that existing precedent. This keeps
  ``TrajectoryResult.total_energy`` honest: a composite is conservative
  iff each of its parts is.
- ``metadata`` aggregates: ``speculative`` is the OR over components,
  ``speculative_components`` is the sorted set-union,
  ``applicable_for_trajectory`` is the AND. The full per-component metadata
  is retained under ``metadata["components"]`` so adapters and snapshot
  reports can still introspect each piece.

Intentional non-features
------------------------

- No weighted sum. ``CompositeField(f1, f2)`` is the only composition surface;
  weighted sums are easy to add later (``CompositeField(f1, f2, weights=...)``)
  if a need arises.
- No operator overloading on ``ForceField``. ``f1 + f2`` is not supported —
  callers must construct the composite explicitly to keep field arithmetic
  visible at the call site.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from usetheforce.protocol import ForceField


class CompositeField:
    """Vector sum of two or more ``ForceField`` instances.

    Parameters
    ----------
    *components:
        Two or more ``ForceField`` instances. An empty tuple raises
        ``ValueError``; a single component is permitted (it's just a thin
        wrapper, useful for uniform handling at call sites that always
        construct composites).
    label:
        Optional short label for ``metadata["model"]``. Defaults to
        ``"composite"``.

    Raises
    ------
    ValueError:
        If ``components`` is empty.
    TypeError:
        If any component does not satisfy the ``ForceField`` protocol.
    """

    metadata: dict[str, Any]

    def __init__(self, *components: ForceField, label: str = "composite") -> None:
        if len(components) == 0:
            raise ValueError("CompositeField requires at least one component")
        for i, c in enumerate(components):
            if not isinstance(c, ForceField):
                raise TypeError(
                    f"component {i} does not satisfy the ForceField protocol"
                )
        self._components: tuple[ForceField, ...] = tuple(components)
        self._label = str(label)

        speculative = any(bool(c.metadata.get("speculative", False)) for c in components)
        spec_components_union = sorted(
            {
                name
                for c in components
                for name in c.metadata.get("speculative_components", [])
            }
        )
        applicable = all(
            bool(c.metadata.get("applicable_for_trajectory", True)) for c in components
        )
        component_metas = [dict(c.metadata) for c in components]
        citation = "; ".join(
            str(c.metadata.get("citation", "")) for c in components if c.metadata.get("citation")
        )
        self.metadata = {
            "avenue": "composite",
            "model": f"{self._label} ({len(components)} components)",
            "speculative": speculative,
            "speculative_components": spec_components_union,
            "applicable_for_trajectory": applicable,
            "citation": citation,
            "components": component_metas,
        }

    @property
    def components(self) -> tuple[ForceField, ...]:
        return self._components

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        total = np.zeros(3, dtype=float)
        for c in self._components:
            total = total + np.asarray(c.force(t, r), dtype=float)
        return total

    def potential(self, r: np.ndarray) -> float | None:
        # Conservative iff every component declares a finite potential.
        total = 0.0
        for c in self._components:
            u = c.potential(r)
            if u is None:
                return None
            total += float(u)
        return total
