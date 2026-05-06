"""Renderers for the snapshot matrix and mission results.

Markdown table writer (no deps), matplotlib bar chart (`[viz]` extra), plotly
3D trajectory (`[interactive]` extra). Heavy deps are lazy-imported.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from usetheforce.missions.snapshot import Snapshot, SnapshotMatrix
from usetheforce.missions.vehicles import Vehicle

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _fmt(x: float) -> str:
    if x == 0 or not (-1e30 < x < 1e30):
        return f"{x:.2e}"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.2e}"
    return f"{x:.3g}"


def snapshot_to_markdown(matrix: SnapshotMatrix, vehicles: dict[str, Vehicle]) -> str:
    """Render the (vehicle × model) matrix as a Markdown report.

    Includes a per-cell assumptions footer so the speculative coupling is
    visible in the output, not buried in code.
    """
    if not matrix:
        return "# Snapshot\n\n_(empty)_\n"

    first_row = next(iter(matrix.values()))
    model_keys = list(first_row.keys())

    lines: list[str] = []
    lines.append("# Propulsion evaluation — capability snapshot")
    lines.append("")
    lines.append(
        "Each cell reports the steady-state acceleration (m/s²) and the corresponding"
        " thrust-to-weight ratio at 1 g. Cells flagged `n/a` are not net-propulsive in the"
        " constant-thrust regime (e.g. Casimir cavity force is internal). All speculative"
        " parameter choices are listed at the end of this report."
    )
    lines.append("")
    lines.append("## Vehicle catalogue")
    lines.append("")
    lines.append("| Vehicle | Description | Mass (kg) | Power (W) |")
    lines.append("| --- | --- | ---: | ---: |")
    for vk, vehicle in vehicles.items():
        lines.append(
            f"| `{vk}` | {vehicle.description} | {_fmt(vehicle.mass_kg)} | {_fmt(vehicle.power_w)} |"
        )
    lines.append("")

    lines.append("## Acceleration (m/s²)")
    lines.append("")
    header = "| Vehicle | " + " | ".join(f"`{k}`" for k in model_keys) + " |"
    sep = "| " + " | ".join(["---"] * (len(model_keys) + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for vk, row in matrix.items():
        cells = []
        for mk in model_keys:
            cell = row[mk]
            if not cell.applicable:
                cells.append("n/a")
            else:
                cells.append(_fmt(cell.accel_mps2))
        lines.append(f"| `{vk}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Thrust-to-weight (×1 g)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for vk, row in matrix.items():
        cells = []
        for mk in model_keys:
            cell = row[mk]
            if not cell.applicable:
                cells.append("n/a")
            else:
                marker = " ⬆" if cell.liftoff_capable else ""
                cells.append(f"{_fmt(cell.twr_1g)}{marker}")
        lines.append(f"| `{vk}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("⬆ = thrust-to-weight ≥ 1 (lift-off feasible against Earth gravity)")
    lines.append("")

    lines.append("## Range scale (m)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for vk, row in matrix.items():
        cells = []
        for mk in model_keys:
            cell = row[mk]
            cells.append(_fmt(cell.range_scale_m))
        lines.append(f"| `{vk}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Falloff: |F(r=1 km)| / |F(r=r_ref)|")
    lines.append("")
    lines.append(
        "Lower ratios mean faster decay with distance. This is where the four"
        " applicable models actually differ — the snapshot acceleration is matched"
        " by construction (same power, same V_REF), but the *shape* of the field"
        " around the vehicle is model-specific."
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for vk, row in matrix.items():
        cells = []
        for mk in model_keys:
            cell = row[mk]
            if not cell.applicable:
                cells.append("n/a")
            else:
                cells.append(_fmt(cell.falloff_ratio))
        lines.append(f"| `{vk}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Speculative parameter assumptions")
    lines.append("")
    lines.append(
        "These are the speculative coupling choices that drive the numbers above. They"
        " are *stated*, not derived from any physical theory."
    )
    lines.append("")
    seen: set[tuple[str, str]] = set()
    for row in matrix.values():
        for mk, cell in row.items():
            if mk in seen:
                continue
            seen.add((mk,) if False else (mk, ""))
            lines.append(f"### `{mk}`")
            lines.append("")
            for k, v in cell.assumptions.items():
                lines.append(f"- **{k}**: {v}")
            if not cell.applicable:
                lines.append(f"- **applicability**: not propulsive — {cell.reason}")
            lines.append("")
    return "\n".join(lines) + "\n"


_AXIS_LABELS = {
    "cubesat_6u": "CubeSat (12 kg)",
    "smallsat": "Smallsat (500 kg)",
    "crewed": "Crewed (12 t)",
    "interplanetary": "Interplanetary (100 t)",
    "generation_ship": "Gen ship (10⁴ t)",
    "city_ship": "City ship (10⁶ t)",
}


def plot_acceleration_vs_mass(
    matrix: SnapshotMatrix,
    vehicles: dict[str, Vehicle],
) -> Figure:
    """Log–log scatter of acceleration vs vehicle mass, one trace per applicable model."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 5))
    first_row = next(iter(matrix.values()))
    model_keys = list(first_row.keys())
    masses = [vehicles[vk].mass_kg for vk in matrix]
    for mk in model_keys:
        ys = []
        xs = []
        for vk, row in matrix.items():
            cell = row[mk]
            if cell.applicable:
                xs.append(vehicles[vk].mass_kg)
                ys.append(cell.accel_mps2)
        if xs:
            ax.loglog(xs, ys, marker="o", label=mk)
    ax.set_xlabel("Vehicle mass (kg)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Constant-power acceleration vs vehicle mass")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(masses)
    ax.set_xticklabels(
        [_AXIS_LABELS.get(vk, vk) for vk in matrix], rotation=30, ha="right", fontsize=8
    )
    fig.tight_layout()
    return fig


def render_mission_table(results: Iterable) -> str:
    """Render a list of MissionResult objects as a Markdown table."""
    lines = []
    lines.append("| Mission | Vehicle | Model | Δv (m/s) | Peak g | Thrust (N) | Energy (J) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| `{r.mission_key}` | `{r.vehicle_key}` | `{r.model_key}` | "
            f"{_fmt(r.delta_v_mps)} | {_fmt(r.peak_g)} | {_fmt(r.thrust_n)} | {_fmt(r.energy_j)} |"
        )
    return "\n".join(lines) + "\n"


def _used(snapshot: Snapshot) -> Snapshot:
    return snapshot
