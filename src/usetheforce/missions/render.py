"""Renderers for the snapshot matrix and mission results.

Markdown table writer (no deps), matplotlib bar chart (`[viz]` extra), plotly
3D trajectory (`[interactive]` extra). Heavy deps are lazy-imported.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from usetheforce.missions.snapshot import SnapshotMatrix
from usetheforce.missions.vehicles import Vehicle
from usetheforce.protocol import ForceField

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
    seen: set[str] = set()
    for row in matrix.values():
        for mk, cell in row.items():
            if mk in seen:
                continue
            seen.add(mk)
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


def plot_vehicle_scale_strip(vehicles: dict[str, Vehicle]) -> Figure:
    """Hero-style horizontal log-mass strip with the six vehicles labelled.

    Labels are staggered above/below the marker line so adjacent decades don't
    crowd each other.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(13, 3.6))
    masses = [v.mass_kg for v in vehicles.values()]
    keys = list(vehicles.keys())
    descriptions = [v.description for v in vehicles.values()]
    ys = [0.5] * len(masses)
    ax.plot(masses, ys, color="#cccccc", linewidth=2, zorder=1)
    ax.scatter(masses, ys, s=180, c="#1f77b4", edgecolor="white", linewidth=2, zorder=3)
    for i, (x, key, desc, mass) in enumerate(zip(masses, keys, descriptions, masses, strict=True)):
        above = i % 2 == 0
        offset = 38 if above else -52
        va = "bottom" if above else "top"
        ax.annotate(
            f"{key}\n{desc}\n{mass:g} kg",
            xy=(x, 0.5),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            linespacing=1.25,
        )
    ax.set_xscale("log")
    ax.set_xlim(masses[0] * 0.2, masses[-1] * 5)
    ax.set_ylim(-1.0, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("Dry mass (kg, log scale)")
    ax.set_title("Vehicle scales evaluated — CubeSat to metropolitan city ship")
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_twr_heatmap(matrix: SnapshotMatrix, vehicles: dict[str, Vehicle]) -> Figure:
    """Vehicle × model thrust-to-weight heatmap (log10), with `n/a` for inapplicable cells."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    first_row = next(iter(matrix.values()))
    model_keys = list(first_row.keys())
    vehicle_keys = list(matrix.keys())

    data = np.full((len(vehicle_keys), len(model_keys)), np.nan)
    for i, vk in enumerate(vehicle_keys):
        for j, mk in enumerate(model_keys):
            cell = matrix[vk][mk]
            if cell.applicable and cell.twr_1g > 0:
                data[i, j] = float(np.log10(cell.twr_1g))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(model_keys, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(vehicle_keys)))
    ax.set_yticklabels([f"{vk}\n({vehicles[vk].mass_kg:g} kg)" for vk in vehicle_keys], fontsize=8)
    for i in range(len(vehicle_keys)):
        for j in range(len(model_keys)):
            cell = matrix[vehicle_keys[i]][model_keys[j]]
            if not cell.applicable:
                ax.text(j, i, "n/a", ha="center", va="center", color="#666", fontsize=8)
            else:
                ax.text(
                    j,
                    i,
                    f"{cell.twr_1g:.2g}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] < -1.0 else "black",
                    fontsize=8,
                )
    cbar = fig.colorbar(im, ax=ax, label="log₁₀(TWR @ 1 g)")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Thrust-to-weight ratio across (vehicle × model)")
    fig.tight_layout()
    return fig


def plot_falloff_vs_distance(
    field_per_model: dict[str, ForceField],
    r_min_m: float = 0.1,
    r_max_m: float = 1e4,
    n_points: int = 200,
) -> Figure:
    """|F(r)| vs distance on log–log axes, one line per applicable model.

    ``field_per_model`` maps model_key → an object that exposes ``force(t, r)``.
    The caller supplies adapter-built ``ForceField`` instances for one chosen
    vehicle so the curves are directly comparable.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    rs = np.logspace(np.log10(r_min_m), np.log10(r_max_m), n_points)
    fig, ax = plt.subplots(figsize=(9, 6))
    probe = np.zeros(3)
    for mk, ff in field_per_model.items():
        mags = np.empty(rs.size)
        for i, r in enumerate(rs):
            probe[0] = r
            try:
                f = ff.force(0.0, probe)
                mags[i] = float(np.linalg.norm(f))
            except (ValueError, ZeroDivisionError):
                mags[i] = np.nan
        ax.loglog(rs, mags, label=mk, linewidth=2)
    ax.set_xlabel("Probe distance r (m)")
    ax.set_ylabel("|F(r)| (N)")
    ax.set_title("Force vs distance — model falloff laws under matched-power scaling")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_mission_delta_v_bar(results: Iterable) -> Figure:
    """Horizontal log-x Δv bars for each integrated mission, coloured by model."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    items = list(results)
    if not items:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "(no mission results)", ha="center", va="center")
        return fig

    labels = [f"{r.vehicle_key}\n× {r.model_key}\n[{r.mission_key}]" for r in items]
    dvs = [r.delta_v_mps for r in items]
    models = [r.model_key for r in items]
    palette = plt.get_cmap("tab10")
    unique_models = list(dict.fromkeys(models))
    colors = {m: palette(i) for i, m in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.85 * len(items))))
    bars = ax.barh(
        range(len(items)),
        dvs,
        color=[colors[m] for m in models],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Achieved Δv (m/s, log scale)")
    ax.set_title("Integrated-mission Δv across (vehicle × model)")
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.4)
    for bar, dv in zip(bars, dvs, strict=True):
        ax.text(
            dv * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{dv:.2e}",
            va="center",
            fontsize=8,
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in unique_models]
    ax.legend(handles, unique_models, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
