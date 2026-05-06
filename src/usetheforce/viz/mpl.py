"""Static matplotlib plots for trajectories and energies. Requires ``[viz]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from usetheforce.protocol import ForceField
from usetheforce.trajectories.integrator import TrajectoryResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure


_AXIS = {"x": 0, "y": 1, "z": 2}


def plot_trajectory_2d(result: TrajectoryResult, axes: str = "xy") -> Figure:
    """2D projection of a trajectory onto the chosen plane (``'xy'``, ``'xz'``, ``'yz'``)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if len(axes) != 2 or any(a not in _AXIS for a in axes):
        raise ValueError(f"axes must be a 2-letter string from x/y/z, got {axes!r}")
    i, j = _AXIS[axes[0]], _AXIS[axes[1]]
    fig, ax = plt.subplots()
    ax.plot(result.r[:, i], result.r[:, j], lw=1.5)
    ax.scatter(result.r[0, i], result.r[0, j], c="green", label="start", zorder=3)
    ax.scatter(result.r[-1, i], result.r[-1, j], c="red", label="end", zorder=3)
    ax.set_xlabel(f"{axes[0]} (m)")
    ax.set_ylabel(f"{axes[1]} (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.set_title(result.field_metadata.get("model", "trajectory"))
    return fig


def plot_energy(result: TrajectoryResult, ff: ForceField | None = None) -> Figure:
    """Plot kinetic energy (and total, if ``ff.potential`` is available)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots()
    ke = result.kinetic_energy()
    ax.plot(result.t, ke, label="kinetic")
    if ff is not None:
        try:
            te: np.ndarray | None = result.total_energy(ff)
        except ValueError:
            te = None
        if te is not None:
            ax.plot(result.t, te, label="total")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("energy (J)")
    ax.legend()
    return fig


def render_field_slice(field_xyz: np.ndarray, axis: str = "z", index: int | None = None) -> Any:
    """Plot the magnitude of a (3, Nx, Ny, Nz) sampled vector field on a slice."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if axis not in _AXIS:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    mag = np.linalg.norm(field_xyz, axis=0)
    a = _AXIS[axis]
    if index is None:
        index = mag.shape[a] // 2
    s: list[slice | int] = [slice(None), slice(None), slice(None)]
    s[a] = index
    fig, ax = plt.subplots()
    ax.imshow(mag[tuple(s)].T, origin="lower")
    ax.set_title(f"|F| slice along {axis}={index}")
    return fig
