"""Visualisations for ``LongRangeMissionResult`` — thrust profiles, power reserves, summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from usetheforce.missions.long_range import LongRangeMissionResult


def plot_thrust_profile(result: LongRangeMissionResult) -> Figure:
    """|F_thrust(t)| and Cartesian components vs t."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = result.trajectory.t
    F = result.thrust_history_n
    ax.plot(t, np.linalg.norm(F, axis=1), color="black", linewidth=2, label="|F|")
    ax.plot(t, F[:, 0], color="#1f77b4", alpha=0.7, label="Fx")
    ax.plot(t, F[:, 1], color="#ff7f0e", alpha=0.7, label="Fy")
    ax.plot(t, F[:, 2], color="#2ca02c", alpha=0.7, label="Fz")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title(f"Thrust profile — {result.name}")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_power_reserve(result: LongRangeMissionResult) -> Figure:
    """Energy reserve remaining vs t (log-y)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(result.trajectory.t, result.power_history_j + 1.0, color="#d62728", linewidth=2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Energy reserve (J), +1 floor for log axis")
    ax.set_title(f"Power reserve — {result.name}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_trajectory_with_thrust_arrows(
    result: LongRangeMissionResult,
    every: int = 20,
) -> Figure:
    """2D trajectory in xy with thrust quiver overlay."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7, 7))
    r = result.trajectory.r
    F = result.thrust_history_n
    ax.plot(r[:, 0], r[:, 1], color="#1f77b4", linewidth=1.8, label="trajectory")
    sample = slice(None, None, max(1, every))
    ax.quiver(
        r[sample, 0],
        r[sample, 1],
        F[sample, 0],
        F[sample, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
        color="#d4a017",
        width=0.003,
        label="thrust",
    )
    ax.scatter(r[0, 0], r[0, 1], c="green", s=80, zorder=3, label="start")
    ax.scatter(r[-1, 0], r[-1, 1], c="red", s=80, zorder=3, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory + thrust quivers — {result.name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_long_range_summary(result: LongRangeMissionResult) -> Figure:
    """2×2 composite: trajectory, thrust magnitude, power reserve, accumulated Δv."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # Top-left: trajectory in xy.
    r = result.trajectory.r
    ax = axs[0, 0]
    ax.plot(r[:, 0], r[:, 1], color="#1f77b4", linewidth=1.5)
    ax.scatter(r[0, 0], r[0, 1], c="green", s=60, zorder=3, label="start")
    ax.scatter(r[-1, 0], r[-1, 1], c="red", s=60, zorder=3, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory (xy)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")

    # Top-right: thrust magnitude.
    ax = axs[0, 1]
    F = result.thrust_history_n
    ax.plot(result.trajectory.t, np.linalg.norm(F, axis=1), color="black", linewidth=1.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|F_thrust| (N)")
    ax.set_title("Thrust magnitude")
    ax.grid(True, ls=":", alpha=0.4)

    # Bottom-left: power reserve.
    ax = axs[1, 0]
    ax.plot(result.trajectory.t, result.power_history_j, color="#d62728", linewidth=1.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Energy reserve (J)")
    ax.set_title("Power reserve depletion")
    ax.grid(True, ls=":", alpha=0.4)

    # Bottom-right: accumulated Δv.
    ax = axs[1, 1]
    v = result.trajectory.v
    v0 = v[0]
    dv = np.linalg.norm(v - v0, axis=1)
    ax.plot(result.trajectory.t, dv, color="#9467bd", linewidth=1.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δv from launch (m/s)")
    ax.set_title("Accumulated Δv")
    ax.grid(True, ls=":", alpha=0.4)

    fig.suptitle(
        f"{result.name} — controller: {result.controller_metadata.get('controller', '?')} | "
        f"Δv = {result.delta_v_mps:.3e} m/s | peak g = {result.peak_g:.3g}"
    )
    fig.tight_layout()
    return fig
