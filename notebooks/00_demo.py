"""End-to-end demo: build a ShapedFieldAnsatz, integrate a trajectory, render with all viz tiers.

Run as a script (``python notebooks/00_demo.py``) or convert to a notebook with
``jupytext --to ipynb 00_demo.py``. Kept as ``.py`` so diffs are reviewable.

Requires the ``[viz]``, ``[interactive]``, and ``[3d]`` extras for the full set
of renders. Each block is guarded — missing extras are reported but don't abort
the script.
"""

# %% Imports
from __future__ import annotations

import numpy as np

from usetheforce.fields import RegularGrid3D
from usetheforce.qfield import ShapedFieldAnsatz
from usetheforce.trajectories import integrate

# %% Build a force field and integrate a trajectory through it.
ff = ShapedFieldAnsatz(amplitude=2.0, sigma=1.0, center=(0.0, 0.0, 0.0))
traj = integrate(
    ff,
    mass=1.0,
    r0=[2.5, 0.0, 0.0],
    v0=[0.0, 0.7, 0.0],
    t_span=(0.0, 12.0),
    n_eval=300,
)
energy = traj.total_energy(ff)
print(f"Energy drift: {np.std(energy) / abs(np.mean(energy)):.2e}")

# %% Tier 1 — matplotlib (static, [viz] extra).
try:
    import matplotlib

    matplotlib.use("Agg")
    from usetheforce.viz.mpl import plot_energy, plot_trajectory_2d

    fig = plot_trajectory_2d(traj, axes="xy")
    fig.savefig("trajectory_xy.png", dpi=120)
    fig2 = plot_energy(traj, ff)
    fig2.savefig("energy.png", dpi=120)
    print("matplotlib: wrote trajectory_xy.png, energy.png")
except ImportError as exc:
    print(f"matplotlib skipped: {exc}")

# %% Tier 2 — plotly (browser-interactive, [interactive] extra).
try:
    from usetheforce.viz.plotly_3d import trajectory_3d

    fig = trajectory_3d(traj)
    fig.write_html("trajectory_3d.html")
    print("plotly: wrote trajectory_3d.html")
except ImportError as exc:
    print(f"plotly skipped: {exc}")

# %% Tier 3 — PyVista (3D animation, [3d] extra).
try:
    from usetheforce.viz.pyvista_3d import animate_trajectory_in_field

    grid = RegularGrid3D(origin=(-3.0, -3.0, -1.0), spacing=(0.5, 0.5, 0.5), shape=(13, 13, 5))
    animate_trajectory_in_field(traj, ff, grid, output="trajectory.gif", fps=20)
    print("pyvista: wrote trajectory.gif")
except ImportError as exc:
    print(f"pyvista skipped: {exc}")
except Exception as exc:  # rendering may fail without a working GL stack
    print(f"pyvista render skipped (likely no GL): {exc}")
