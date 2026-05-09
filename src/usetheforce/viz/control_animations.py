"""2D animations of ``LongRangeMissionResult`` — thrust + trajectory dashboards.

Animated counterparts of the static helpers in ``viz/control.py``. Two helpers:

- ``animate_long_range_mission`` — animated 2×2 dashboard for one mission
  (trajectory + thrust magnitude + power reserve + accumulated Δv, all synced
  by a moving "now" cursor).
- ``animate_model_comparison`` — side-by-side trajectories and thrust
  magnitudes for several missions on a normalised time axis, useful for
  contrasting propulsion capabilities.

Both follow the convention from ``viz/pyvista_3d.py``: when ``output`` is a
path ending in ``.gif`` the animation is written via the pillow writer and
``None`` is returned; otherwise the live ``FuncAnimation`` is returned for
interactive display. Matplotlib is lazy-imported per-function (``[viz]`` extra).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

    from usetheforce.missions.long_range import LongRangeMissionResult


def _save_or_return(
    anim: FuncAnimation,
    fig,
    output: str | Path | None,
    fps: int,
) -> FuncAnimation | None:
    if output is None:
        return anim
    out = Path(output)
    if out.suffix.lower() != ".gif":
        raise ValueError(f"output must end with .gif (got {out.suffix!r})")
    anim.save(str(out), writer="pillow", fps=fps)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
    return None


def animate_long_range_mission(
    result: LongRangeMissionResult,
    output: str | Path | None = None,
    fps: int = 20,
    every: int = 1,
) -> FuncAnimation | None:
    """Animated 2×2 dashboard for one ``LongRangeMissionResult``.

    Layout mirrors ``viz/control.py:plot_long_range_summary``:

    - Top-left: xy trajectory (full path drawn faintly + growing line +
      spacecraft marker + a yellow line indicating current thrust direction).
    - Top-right: ``|F_thrust|(t)`` with a "now" cursor and dot.
    - Bottom-left: power reserve vs t, same idiom.
    - Bottom-right: accumulated Δv vs t, same idiom.

    ``every`` subsamples the trajectory output times — pass ``every=4`` etc.
    for long missions to keep the produced GIF small.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.animation import FuncAnimation  # noqa: PLC0415

    if every < 1:
        raise ValueError("every must be >= 1")

    t = result.trajectory.t
    r = result.trajectory.r
    v = result.trajectory.v
    F = result.thrust_history_n
    P = result.power_history_j

    if t.size < 2:
        raise ValueError("trajectory must contain at least two samples")

    Fmag = np.linalg.norm(F, axis=1)
    dv = np.linalg.norm(v - v[0], axis=1)

    frame_indices = np.arange(0, len(t), every)
    if frame_indices[-1] != len(t) - 1:
        frame_indices = np.append(frame_indices, len(t) - 1)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # Top-left: trajectory in xy.
    ax_traj = axs[0, 0]
    ax_traj.plot(r[:, 0], r[:, 1], color="#1f77b4", linewidth=1, alpha=0.2)
    ax_traj.scatter([r[0, 0]], [r[0, 1]], c="green", s=60, zorder=3, label="start")
    ax_traj.scatter([r[-1, 0]], [r[-1, 1]], c="red", s=60, zorder=3, label="end")
    (traj_line,) = ax_traj.plot([], [], color="#1f77b4", linewidth=1.8)
    (craft,) = ax_traj.plot([], [], "o", color="black", markersize=7, zorder=4)
    (thrust_arrow,) = ax_traj.plot([], [], color="#d4a017", linewidth=2.5, zorder=4)
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory (xy)")
    ax_traj.grid(True, ls=":", alpha=0.4)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(fontsize=8, loc="best")

    x_span = float(np.ptp(r[:, 0])) or 1.0
    y_span = float(np.ptp(r[:, 1])) or 1.0
    arrow_len = 0.08 * max(x_span, y_span)

    # Top-right: |F|(t).
    ax_F = axs[0, 1]
    ax_F.plot(t, Fmag, color="black", linewidth=1, alpha=0.25)
    ax_F.set_xlabel("t (s)")
    ax_F.set_ylabel("|F_thrust| (N)")
    ax_F.set_title("Thrust magnitude")
    ax_F.grid(True, ls=":", alpha=0.4)
    cursor_F = ax_F.axvline(t[0], color="black", linewidth=1, linestyle="--", alpha=0.5)
    (dot_F,) = ax_F.plot([], [], "o", color="black", markersize=6)

    # Bottom-left: power reserve.
    ax_P = axs[1, 0]
    ax_P.plot(t, P, color="#d62728", linewidth=1, alpha=0.25)
    ax_P.set_xlabel("t (s)")
    ax_P.set_ylabel("Energy reserve (J)")
    ax_P.set_title("Power reserve depletion")
    ax_P.grid(True, ls=":", alpha=0.4)
    cursor_P = ax_P.axvline(t[0], color="#d62728", linewidth=1, linestyle="--", alpha=0.5)
    (dot_P,) = ax_P.plot([], [], "o", color="#d62728", markersize=6)

    # Bottom-right: accumulated Δv.
    ax_dv = axs[1, 1]
    ax_dv.plot(t, dv, color="#9467bd", linewidth=1, alpha=0.25)
    ax_dv.set_xlabel("t (s)")
    ax_dv.set_ylabel("Δv from launch (m/s)")
    ax_dv.set_title("Accumulated Δv")
    ax_dv.grid(True, ls=":", alpha=0.4)
    cursor_dv = ax_dv.axvline(t[0], color="#9467bd", linewidth=1, linestyle="--", alpha=0.5)
    (dot_dv,) = ax_dv.plot([], [], "o", color="#9467bd", markersize=6)

    fig.suptitle(
        f"{result.name} — controller: {result.controller_metadata.get('controller', '?')} | "
        f"Δv = {result.delta_v_mps:.3e} m/s | peak g = {result.peak_g:.3g}"
    )
    fig.tight_layout()

    def update(frame_idx: int):
        i = int(frame_indices[frame_idx])
        ti = float(t[i])
        traj_line.set_data(r[: i + 1, 0], r[: i + 1, 1])
        craft.set_data([float(r[i, 0])], [float(r[i, 1])])
        F_xy_norm = float(np.linalg.norm(F[i, :2]))
        if F_xy_norm > 0:
            ux = F[i, 0] / F_xy_norm
            uy = F[i, 1] / F_xy_norm
            thrust_arrow.set_data(
                [float(r[i, 0]), float(r[i, 0] + arrow_len * ux)],
                [float(r[i, 1]), float(r[i, 1] + arrow_len * uy)],
            )
        else:
            thrust_arrow.set_data([], [])
        cursor_F.set_xdata([ti, ti])
        dot_F.set_data([ti], [float(Fmag[i])])
        cursor_P.set_xdata([ti, ti])
        dot_P.set_data([ti], [float(P[i])])
        cursor_dv.set_xdata([ti, ti])
        dot_dv.set_data([ti], [float(dv[i])])
        return (
            traj_line,
            craft,
            thrust_arrow,
            cursor_F,
            dot_F,
            cursor_P,
            dot_P,
            cursor_dv,
            dot_dv,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
        cache_frame_data=False,
    )
    return _save_or_return(anim, fig, output, fps)


def animate_model_comparison(
    results: Sequence[LongRangeMissionResult],
    output: str | Path | None = None,
    fps: int = 20,
    labels: Sequence[str] | None = None,
    every: int = 1,
) -> FuncAnimation | None:
    """Side-by-side animation contrasting several missions on shared axes.

    Top panel: xy trajectories (one growing line per result, distinct colour,
    one marker per result at its current position). Bottom panel: ``|F|(t)``
    plotted against *normalised time* ``t / t_total`` so missions of different
    durations advance in lockstep visually. A single vertical "now" cursor on
    the bottom panel ties all results to the same frame fraction.

    ``labels`` overrides the per-result legend names (default: each result's
    ``controller_metadata["controller"]``).

    Use this to contrast propulsion-capability variants on the same task —
    e.g., the same ``heliocentric_cruise`` mission with three different
    ``max_thrust_n`` values.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.animation import FuncAnimation  # noqa: PLC0415

    if not results:
        raise ValueError("results must be non-empty")
    if every < 1:
        raise ValueError("every must be >= 1")

    if labels is None:
        labels = [
            r.controller_metadata.get("controller", f"run {i}") for i, r in enumerate(results)
        ]
    if len(labels) != len(results):
        raise ValueError(f"labels length {len(labels)} != results length {len(results)}")

    n_max = max(r.trajectory.t.size for r in results)
    n_frames = max(2, n_max // every)

    fractions: list[np.ndarray] = []
    Fmags: list[np.ndarray] = []
    for rr in results:
        tt = rr.trajectory.t
        if tt.size > 1 and tt[-1] > tt[0]:
            fractions.append((tt - tt[0]) / (tt[-1] - tt[0]))
        else:
            fractions.append(np.zeros_like(tt))
        Fmags.append(np.linalg.norm(rr.thrust_history_n, axis=1))

    fig, (ax_traj, ax_F) = plt.subplots(2, 1, figsize=(10, 9))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    traj_lines = []
    craft_markers = []
    F_growing = []
    F_dots = []

    for idx, (rr, lab) in enumerate(zip(results, labels, strict=True)):
        col = palette[idx % len(palette)]
        r = rr.trajectory.r
        Fm = Fmags[idx]
        frac = fractions[idx]

        ax_traj.plot(r[:, 0], r[:, 1], color=col, linewidth=1, alpha=0.18)
        (line,) = ax_traj.plot([], [], color=col, linewidth=1.8, label=lab)
        (mk,) = ax_traj.plot([], [], "o", color=col, markersize=7, zorder=4)
        traj_lines.append(line)
        craft_markers.append(mk)

        ax_F.plot(frac, Fm, color=col, linewidth=1, alpha=0.18)
        (gline,) = ax_F.plot([], [], color=col, linewidth=1.6, label=lab)
        (dot,) = ax_F.plot([], [], "o", color=col, markersize=6)
        F_growing.append(gline)
        F_dots.append(dot)

    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectories — capability comparison")
    ax_traj.grid(True, ls=":", alpha=0.4)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(fontsize=9, loc="best")

    ax_F.set_xlabel("normalised time (t / t_total)")
    ax_F.set_ylabel("|F_thrust| (N)")
    ax_F.set_title("Thrust magnitude vs normalised time")
    ax_F.grid(True, ls=":", alpha=0.4)
    ax_F.set_xlim(0.0, 1.0)
    ax_F.legend(fontsize=9, loc="best")

    cursor = ax_F.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    fig.suptitle("Propulsion capability comparison")
    fig.tight_layout()

    def update(frame_idx: int):
        frac_now = min(1.0, (frame_idx * every) / max(1, n_max - 1))
        artists = []
        for idx, rr in enumerate(results):
            r = rr.trajectory.r
            frac = fractions[idx]
            Fm = Fmags[idx]
            i = int(np.searchsorted(frac, frac_now))
            i = min(i, r.shape[0] - 1)
            traj_lines[idx].set_data(r[: i + 1, 0], r[: i + 1, 1])
            craft_markers[idx].set_data([float(r[i, 0])], [float(r[i, 1])])
            F_growing[idx].set_data(frac[: i + 1], Fm[: i + 1])
            F_dots[idx].set_data([float(frac[i])], [float(Fm[i])])
            artists.extend([traj_lines[idx], craft_markers[idx], F_growing[idx], F_dots[idx]])
        cursor.set_xdata([frac_now, frac_now])
        artists.append(cursor)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        cache_frame_data=False,
    )
    return _save_or_return(anim, fig, output, fps)
