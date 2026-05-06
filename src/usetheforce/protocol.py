"""The ``ForceField`` protocol — the single shared seam between propulsion avenues.

Each avenue (Casimir, qfield, antimatter, qgp, …) produces an object satisfying
this protocol; ``trajectories.integrate`` consumes it. Adding a new avenue is
one class. Keep this contract minimal — every method here is paid for in every
implementation.

Writing a new ``ForceField``
----------------------------

A minimal example::

    import numpy as np
    from typing import Any

    class ConstantForce:
        '''Apply a constant body-frame force; useful for sanity tests.'''
        metadata: dict[str, Any]

        def __init__(self, force_n: tuple[float, float, float]) -> None:
            self._f = np.asarray(force_n, dtype=float)
            self.metadata = {
                "avenue": "test",
                "model": "constant force",
                "speculative": False,
                "speculative_components": [],
                "citation": "test only",
            }

        def force(self, t: float, r: np.ndarray) -> np.ndarray:
            return self._f

        def potential(self, r: np.ndarray) -> float | None:
            # Constant force = -∇(linear potential); we could expose that here, but
            # a position-independent body-force model is easier to leave as None.
            return None

Conventions
-----------

- ``force(t, r)`` returns Newtons (shape ``(3,)``); units are SI throughout.
- ``potential(r)`` is optional. Return ``None`` if the field is non-conservative
  or the model genuinely doesn't have a probe-position potential. Returning a
  ``float`` lets ``TrajectoryResult.total_energy`` compute conservation; the
  integrator will reject ``NaN`` or ``±inf``.
- ``metadata`` carries:
  - ``"avenue"``: top-level avenue (``"casimir"``, ``"qfield"``, ``"antimatter"``, ``"qgp"`` …)
  - ``"model"``: short human-readable description
  - ``"speculative"``: ``True`` for any model whose mechanism is not derived
    from established physics. Tests assert this flag is set; treat it as
    load-bearing.
  - ``"speculative_components"``: list of parameter names that carry the
    speculative leap. Empty for purely-anchored models.
  - ``"citation"``: one-line reference (paper, arXiv id, or "speculative ansatz").
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ForceField(Protocol):
    """A propulsion-relevant force field over (t, r) in SI.

    See the module docstring for conventions and a worked example.
    """

    metadata: dict[str, Any]

    def force(self, t: float, r: np.ndarray) -> np.ndarray:
        """Force in Newtons at time ``t`` (s) and position ``r`` (m)."""
        ...

    def potential(self, r: np.ndarray) -> float | None:
        """Potential energy in Joules at ``r``, or ``None`` if not conservative.

        ``None`` declares the model has no probe-position-derivable scalar
        potential. ``TrajectoryResult.total_energy`` raises if a model returns
        ``None`` (or non-finite floats), so callers can rely on the return
        value being either a finite float or an explicit ``None``.
        """
        ...
