"""Propulsion evaluation — vehicles × models snapshot, then targeted missions.

Writes everything to ``results/``:

- ``snapshot.md`` — Markdown report with the (vehicle × model) matrix and the
  speculative parameter assumptions used.
- ``accel_vs_mass.png`` — log–log scatter of acceleration vs vehicle mass.
- ``mission_<name>.html`` and ``mission_<name>.png`` — per-mission artifacts.

Run with: ``.venv/bin/python notebooks/02_evaluation.py``.
"""

from __future__ import annotations

from pathlib import Path

from usetheforce.missions import (
    ALL_ADAPTERS,
    MISSIONS,
    VEHICLES,
    evaluate_snapshot,
    run_mission,
)
from usetheforce.missions.render import (
    plot_acceleration_vs_mass,
    render_mission_table,
    snapshot_to_markdown,
)

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def main() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    md = snapshot_to_markdown(matrix, VEHICLES)
    (OUT / "snapshot.md").write_text(md)
    print(f"wrote {OUT / 'snapshot.md'} ({len(md)} chars)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        fig = plot_acceleration_vs_mass(matrix, VEHICLES)
        fig.savefig(OUT / "accel_vs_mass.png", dpi=120)
        print(f"wrote {OUT / 'accel_vs_mass.png'}")
    except ImportError as exc:
        print(f"matplotlib skipped: {exc}")

    # Targeted missions: pick the four most propulsively interesting (vehicle, model) pairs.
    pairs = [
        ("smallsat", "shaped_field_ansatz", "free_burn_100s"),
        ("crewed", "stimulated_emission_array", "leo_raise_100s"),
        ("interplanetary", "heavy_element_lattice", "lunar_transfer_3d"),
        ("generation_ship", "antimatter_graviton", "stationkeep_300s"),
        ("city_ship", "antimatter_graviton", "year_burn_free"),
        ("cubesat_6u", "stimulated_emission_array", "free_burn_100s"),
    ]
    results = []
    for vk, mk, miss_key in pairs:
        result = run_mission(VEHICLES[vk], mk, ALL_ADAPTERS[mk], MISSIONS[miss_key])
        results.append(result)
        label = f"mission_{vk}_{mk}_{miss_key}"
        print(
            f"\n{label}\n  thrust = {result.thrust_n:.3e} N\n"
            f"  Δv = {result.delta_v_mps:.3e} m/s\n"
            f"  peak g = {result.peak_g:.3e}\n"
            f"  energy = {result.energy_j:.3e} J"
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            from usetheforce.viz.mpl import plot_trajectory_2d

            plot_trajectory_2d(result.trajectory, axes="xy").savefig(OUT / f"{label}.png", dpi=120)
            print(f"  wrote {OUT / (label + '.png')}")
        except ImportError as exc:
            print(f"  matplotlib skipped: {exc}")
        try:
            from usetheforce.viz.plotly_3d import trajectory_3d

            trajectory_3d(result.trajectory).write_html(OUT / f"{label}.html")
            print(f"  wrote {OUT / (label + '.html')}")
        except ImportError as exc:
            print(f"  plotly skipped: {exc}")

    table = render_mission_table(results)
    (OUT / "missions.md").write_text("# Propulsion evaluation — integrated missions\n\n" + table)
    print(f"\nwrote {OUT / 'missions.md'}")


if __name__ == "__main__":
    main()
