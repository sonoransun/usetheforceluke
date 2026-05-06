"""Smoke tests for the visualization tiers — each guarded by its optional extra."""

from __future__ import annotations

import os

import numpy as np
import pytest

from usetheforce.qfield import ShapedFieldAnsatz
from usetheforce.trajectories import integrate


@pytest.fixture(scope="module")
def trajectory():
    ff = ShapedFieldAnsatz(amplitude=1.0, sigma=1.0)
    return ff, integrate(
        ff,
        mass=1.0,
        r0=[2.0, 0.0, 0.0],
        v0=[0.0, 0.5, 0.0],
        t_span=(0.0, 4.0),
        n_eval=40,
    )


def test_mpl_trajectory_2d(trajectory) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from usetheforce.viz.mpl import plot_energy, plot_trajectory_2d  # noqa: PLC0415

    ff, traj = trajectory
    fig1 = plot_trajectory_2d(traj, axes="xy")
    fig2 = plot_energy(traj, ff)
    assert fig1 is not None and fig2 is not None


def test_plotly_trajectory_3d(trajectory) -> None:
    pytest.importorskip("plotly")
    from usetheforce.viz.plotly_3d import trajectory_3d  # noqa: PLC0415

    _, traj = trajectory
    fig = trajectory_3d(traj)
    assert fig is not None
    assert len(fig.data) >= 1


@pytest.mark.skipif(
    os.environ.get("USETHEFORCE_SKIP_PYVISTA") == "1",
    reason="PyVista smoke disabled (no GL / off-screen).",
)
def test_pyvista_imports() -> None:
    """Just verify the module imports and the public function is callable.

    Rendering requires a working VTK/GL stack, which CI without OSMesa cannot
    guarantee. Set ``USETHEFORCE_SKIP_PYVISTA=1`` to skip on those machines.
    """
    pytest.importorskip("pyvista")
    from usetheforce.viz.pyvista_3d import animate_trajectory_in_field  # noqa: F401, PLC0415


def test_render_field_slice() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from usetheforce.viz.mpl import render_field_slice  # noqa: PLC0415

    field = np.random.default_rng(0).normal(size=(3, 5, 5, 5))
    fig = render_field_slice(field, axis="z")
    assert fig is not None
