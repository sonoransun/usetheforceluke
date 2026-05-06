"""Browser-interactive 3D plots. Requires ``[interactive]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

from usetheforce.trajectories.integrator import TrajectoryResult

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def trajectory_3d(result: TrajectoryResult) -> Figure:
    """Interactive 3D line + endpoint markers for a trajectory."""
    import plotly.graph_objects as go  # noqa: PLC0415

    r = result.r
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            mode="lines",
            line={"width": 4},
            name="path",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[r[0, 0], r[-1, 0]],
            y=[r[0, 1], r[-1, 1]],
            z=[r[0, 2], r[-1, 2]],
            mode="markers+text",
            text=["start", "end"],
            textposition="top center",
            marker={"size": 6, "color": ["green", "red"]},
            name="endpoints",
        )
    )
    fig.update_layout(
        title=result.field_metadata.get("model", "trajectory"),
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"},
    )
    return fig
