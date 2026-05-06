"""ODE integration of Newton's 2nd law through a ``ForceField``.

Wraps ``scipy.integrate.solve_ivp`` with high-accuracy defaults (DOP853,
rtol=1e-10) suitable for orbit-class problems.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from usetheforce.protocol import ForceField


@dataclass(slots=True)
class TrajectoryResult:
    """Time-series of position and velocity."""

    t: np.ndarray  # (N,)
    r: np.ndarray  # (N, 3)
    v: np.ndarray  # (N, 3)
    mass: float
    field_metadata: dict[str, Any]

    def kinetic_energy(self) -> np.ndarray:
        return 0.5 * self.mass * np.einsum("ij,ij->i", self.v, self.v)

    def total_energy(self, ff: ForceField) -> np.ndarray:
        """Kinetic + potential energy along the trajectory (J).

        Requires ``ff.potential(r)`` to return finite floats at every sampled
        point. Raises ``ValueError`` if the field doesn't expose a potential
        (returns ``None``) or returns non-finite values (NaN, ±∞).
        """
        pe_list = [ff.potential(self.r[i]) for i in range(len(self.t))]
        if any(p is None for p in pe_list):
            raise ValueError("ForceField does not provide a potential; total_energy is undefined")
        pe = np.asarray(pe_list, dtype=float)
        if not np.all(np.isfinite(pe)):
            raise ValueError(
                "ForceField.potential returned non-finite values; total_energy is undefined"
            )
        return self.kinetic_energy() + pe


def integrate(
    ff: ForceField,
    mass: float,
    r0: Sequence[float],
    v0: Sequence[float],
    t_span: tuple[float, float],
    n_eval: int = 200,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    **solve_ivp_kwargs: Any,
) -> TrajectoryResult:
    """Integrate ``m r̈ = ff.force(t, r)`` from ``t_span[0]`` to ``t_span[1]``.

    All inputs are SI: ``mass`` in kg, ``r0`` in m (shape (3,)), ``v0`` in m/s
    (shape (3,)), ``t_span`` in seconds. ``n_eval`` is the number of evenly-spaced
    output samples (the integrator may take many internal steps between them).
    Defaults are tuned for orbital-class problems: 8th-order Dormand–Prince
    (``DOP853``) with ``rtol=1e-10`` and ``atol=1e-12``.
    """
    if mass <= 0:
        raise ValueError("mass must be positive")
    r0_arr = np.asarray(r0, dtype=float)
    v0_arr = np.asarray(v0, dtype=float)
    if r0_arr.shape != (3,) or v0_arr.shape != (3,):
        raise ValueError("r0 and v0 must have shape (3,)")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        a = ff.force(t, r) / mass
        return np.concatenate([v, a])

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    y0 = np.concatenate([r0_arr, v0_arr])
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **solve_ivp_kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return TrajectoryResult(
        t=sol.t,
        r=sol.y[:3].T,
        v=sol.y[3:].T,
        mass=mass,
        field_metadata=dict(ff.metadata),
    )
