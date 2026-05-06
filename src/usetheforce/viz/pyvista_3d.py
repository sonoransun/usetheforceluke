"""3D interactive animation of trajectories through shaped fields. Requires ``[3d]`` extra.

The canonical "watch the field evolve while the craft moves through it" view.
Renders glyph vectors of a sampled force field, an optional energy isosurface,
and an animated marker for the craft along ``result.r``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from usetheforce.fields.grid import RegularGrid3D
from usetheforce.fields.sample import sample_force_field
from usetheforce.protocol import ForceField
from usetheforce.trajectories.integrator import TrajectoryResult

if TYPE_CHECKING:
    import pyvista as pv


def _build_field_glyphs(ff: ForceField, grid: RegularGrid3D, t: float, factor: float) -> pv.DataSet:
    import pyvista as pv  # noqa: PLC0415

    field = sample_force_field(ff, grid, t)  # (3, Nx, Ny, Nz)
    points = grid.points()  # (N, 3) in ij ordering
    vectors = field.reshape(3, -1).T  # (N, 3)
    # Avoid div-by-zero in glyph scaling.
    norms = np.linalg.norm(vectors, axis=1)
    max_norm = float(np.max(norms)) if np.max(norms) > 0 else 1.0
    cloud = pv.PolyData(points)
    cloud["F"] = vectors
    cloud["|F|"] = norms
    glyphs = cloud.glyph(orient="F", scale="|F|", factor=factor / max_norm)
    return glyphs


def animate_trajectory_in_field(
    result: TrajectoryResult,
    ff: ForceField,
    grid: RegularGrid3D,
    output: str | Path | None = None,
    glyph_factor: float = 0.5,
    fps: int = 20,
    show_isosurface: bool = True,
) -> None:
    """Animate the trajectory through a glyph-rendered field.

    If ``output`` ends with ``.gif`` or ``.mp4`` an animation file is written;
    otherwise an interactive window is shown (or a trame server in Jupyter).
    """
    import pyvista as pv  # noqa: PLC0415

    plotter = pv.Plotter(off_screen=output is not None)
    glyphs = _build_field_glyphs(ff, grid, result.t[0], glyph_factor)
    plotter.add_mesh(glyphs, scalars="|F|", cmap="viridis", show_scalar_bar=True)

    if show_isosurface and getattr(ff, "potential", None) is not None:
        try:
            scalar = np.array(
                [ff.potential(p) for p in grid.points()],
                dtype=float,
            ).reshape(grid.shape)
            mesh = pv.ImageData(
                dimensions=grid.shape,
                spacing=grid.spacing,
                origin=grid.origin,
            )
            mesh["U"] = scalar.flatten(order="F")
            iso = mesh.contour(isosurfaces=5, scalars="U")
            if iso.n_points > 0:
                plotter.add_mesh(iso, opacity=0.3, cmap="plasma", show_scalar_bar=False)
        except (TypeError, ValueError):
            pass  # potential not numerically usable; skip isosurface

    craft = pv.Sphere(radius=glyph_factor * 0.2, center=tuple(result.r[0]))
    craft_actor = plotter.add_mesh(craft, color="red")

    if output is not None:
        out = Path(output)
        if out.suffix.lower() == ".gif":
            plotter.open_gif(str(out), fps=fps)
        else:
            plotter.open_movie(str(out), framerate=fps)

    for i in range(len(result.t)):
        plotter.remove_actor(craft_actor)
        craft = pv.Sphere(radius=glyph_factor * 0.2, center=tuple(result.r[i]))
        craft_actor = plotter.add_mesh(craft, color="red")
        if output is not None:
            plotter.write_frame()
        else:
            plotter.update()

    if output is not None:
        plotter.close()
    else:
        plotter.show()
