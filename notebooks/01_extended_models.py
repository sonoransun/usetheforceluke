"""End-to-end demo of the three new theoretical force options.

Each section: build the model → integrate a sample trajectory → render with
matplotlib + plotly + (optionally) PyVista. All viz tiers are guarded so a
missing extra reports cleanly without aborting.

Run as ``.venv/bin/python notebooks/01_extended_models.py``.
"""

from __future__ import annotations

import numpy as np

from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.fields import RegularGrid3D
from usetheforce.qfield import HeavyElementLattice, StimulatedEmissionArray
from usetheforce.trajectories import integrate


def _save_viz(label: str, ff, traj, grid: RegularGrid3D) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from usetheforce.viz.mpl import plot_energy, plot_trajectory_2d

        plot_trajectory_2d(traj, axes="xy").savefig(f"{label}_xy.png", dpi=120)
        plot_energy(traj, ff).savefig(f"{label}_energy.png", dpi=120)
        print(f"  matplotlib: wrote {label}_xy.png, {label}_energy.png")
    except ImportError as exc:
        print(f"  matplotlib skipped: {exc}")

    try:
        from usetheforce.viz.plotly_3d import trajectory_3d

        trajectory_3d(traj).write_html(f"{label}_3d.html")
        print(f"  plotly: wrote {label}_3d.html")
    except ImportError as exc:
        print(f"  plotly skipped: {exc}")

    try:
        from usetheforce.viz.pyvista_3d import animate_trajectory_in_field

        animate_trajectory_in_field(traj, ff, grid, output=f"{label}.gif", fps=20)
        print(f"  pyvista: wrote {label}.gif")
    except ImportError as exc:
        print(f"  pyvista skipped: {exc}")
    except Exception as exc:  # GL/VTK availability varies on headless machines
        print(f"  pyvista render skipped: {exc}")


# %% Heavy-element lattice — three softened sources arranged in a triangle.
print("\n[1] HeavyElementLattice")
he = HeavyElementLattice(
    sites=np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]]),
    strengths=np.array([1.0, 1.0, 1.0]),
    coupling=0.5,
    softening=0.2,
)
he_traj = integrate(
    he, mass=1.0, r0=[2.5, 0.0, 0.1], v0=[0.0, 0.4, 0.0], t_span=(0.0, 20.0), n_eval=300
)
print(
    f"  energy drift: {np.std(he_traj.total_energy(he)) / abs(np.mean(he_traj.total_energy(he))):.2e}"
)
he_grid = RegularGrid3D(origin=(-3.0, -3.0, -0.5), spacing=(0.5, 0.5, 0.5), shape=(13, 13, 3))
_save_viz("heavy_elements", he, he_traj, he_grid)


# %% Stimulated-emission phased array — four emitters on a square with rotating phase.
print("\n[2] StimulatedEmissionArray")
square = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
em = StimulatedEmissionArray(
    positions=square,
    amplitudes=[1.0, 1.0, 1.0, 1.0],
    phases=[0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
    wavenumber=2.0,
    coupling=0.3,
)
# Keep the probe well above the emitter plane (z=0) so it never passes through a source.
em_traj = integrate(
    em, mass=1.0, r0=[0.0, 0.0, 2.5], v0=[0.05, 0.05, 0.0], t_span=(0.0, 8.0), n_eval=200
)
print(
    f"  energy drift: {np.std(em_traj.total_energy(em)) / abs(np.mean(em_traj.total_energy(em))):.2e}"
)
em_grid = RegularGrid3D(origin=(-2.0, -2.0, -1.0), spacing=(0.4, 0.4, 0.4), shape=(11, 11, 7))
_save_viz("stimulated_emission", em, em_traj, em_grid)


# %% Antimatter graviton field — single Yukawa source.
print("\n[3] AntimatterGravitonField")
gv = AntimatterGravitonField(
    source=(0.0, 0.0, 0.0),
    gamma=1.0,
    coupling=1.0,
    screening=20.0,
    probe_mass=1.0,
)
gv_traj = integrate(
    gv, mass=1.0, r0=[2.0, 0.0, 0.0], v0=[0.0, 0.6, 0.0], t_span=(0.0, 25.0), n_eval=400
)
print(
    f"  energy drift: {np.std(gv_traj.total_energy(gv)) / abs(np.mean(gv_traj.total_energy(gv))):.2e}"
)
# Grid origin offset so no sample point lands on the singular source at (0,0,0).
gv_grid = RegularGrid3D(origin=(-3.1, -3.1, -0.6), spacing=(0.5, 0.5, 0.5), shape=(13, 13, 3))
_save_viz("graviton", gv, gv_traj, gv_grid)
