"""Generate the curated figure set embedded in README.md.

Deterministic — no RNG, no time-dependent inputs. Re-running this script alone
is sufficient to refresh every image referenced from the README.

Outputs to ``assets/`` at the repo root:

- ``vehicle_scale_strip.png`` — hero image of the six vehicle scales
- ``twr_heatmap.png`` — thrust-to-weight grid across (vehicle × model)
- ``falloff_comparison.png`` — |F(r)| vs distance for the four applicable models
- ``mission_dv_bar.png`` — Δv per integrated mission
- ``accel_vs_mass.png`` — copied from results/ if present (fallback: regenerate)
- ``mission_trajectory.png`` — single trajectory plot, copied from results/

Run with: ``.venv/bin/python notebooks/03_documentation_figures.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from usetheforce.missions import (
    ALL_ADAPTERS,
    MISSIONS,
    VEHICLES,
    evaluate_snapshot,
    run_mission,
)
from usetheforce.missions.render import (
    plot_acceleration_vs_mass,
    plot_falloff_vs_distance,
    plot_mission_delta_v_bar,
    plot_twr_heatmap,
    plot_vehicle_scale_strip,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    print(f"writing figures to {ASSETS}/")

    # 1. Vehicle scale strip — hero image
    fig = plot_vehicle_scale_strip(VEHICLES)
    fig.savefig(ASSETS / "vehicle_scale_strip.png", dpi=144, bbox_inches="tight")
    print("  vehicle_scale_strip.png")

    # 2. Snapshot matrix → TWR heatmap
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    fig = plot_twr_heatmap(matrix, VEHICLES)
    fig.savefig(ASSETS / "twr_heatmap.png", dpi=144, bbox_inches="tight")
    print("  twr_heatmap.png")

    # 3. Falloff comparison — uses smallsat-scaled adapter outputs across r-range.
    smallsat = VEHICLES["smallsat"]
    applicable = [
        "shaped_field_ansatz",
        "heavy_element_lattice",
        "stimulated_emission_array",
        "antimatter_graviton",
        "qgp_graviton",
    ]
    fields = {mk: ALL_ADAPTERS[mk](smallsat, smallsat.power_w).field for mk in applicable}
    fig = plot_falloff_vs_distance(fields, r_min_m=0.1, r_max_m=1e4, n_points=200)
    fig.savefig(ASSETS / "falloff_comparison.png", dpi=144, bbox_inches="tight")
    print("  falloff_comparison.png")

    # 4. Mission Δv bar — same six pairs as in 02_evaluation.py.
    pairs = [
        ("smallsat", "shaped_field_ansatz", "free_burn_100s"),
        ("crewed", "stimulated_emission_array", "leo_raise_100s"),
        ("interplanetary", "heavy_element_lattice", "lunar_transfer_3d"),
        ("generation_ship", "antimatter_graviton", "stationkeep_300s"),
        ("city_ship", "antimatter_graviton", "year_burn_free"),
        ("cubesat_6u", "stimulated_emission_array", "free_burn_100s"),
    ]
    results = [
        run_mission(VEHICLES[vk], mk, ALL_ADAPTERS[mk], MISSIONS[mission])
        for vk, mk, mission in pairs
    ]
    fig = plot_mission_delta_v_bar(results)
    fig.savefig(ASSETS / "mission_dv_bar.png", dpi=144, bbox_inches="tight")
    print("  mission_dv_bar.png")

    # 5. Reuse accel-vs-mass from results/, or regenerate if missing.
    src = RESULTS / "accel_vs_mass.png"
    if src.exists():
        shutil.copy2(src, ASSETS / "accel_vs_mass.png")
        print("  accel_vs_mass.png (copied from results/)")
    else:
        fig = plot_acceleration_vs_mass(matrix, VEHICLES)
        fig.savefig(ASSETS / "accel_vs_mass.png", dpi=144, bbox_inches="tight")
        print("  accel_vs_mass.png (generated)")

    # 6. Pick a representative mission trajectory and copy it as `mission_trajectory.png`.
    candidates = [
        "mission_interplanetary_heavy_element_lattice_lunar_transfer_3d.png",
        "mission_city_ship_antimatter_graviton_year_burn_free.png",
        "mission_generation_ship_antimatter_graviton_stationkeep_300s.png",
    ]
    for name in candidates:
        cand = RESULTS / name
        if cand.exists():
            shutil.copy2(cand, ASSETS / "mission_trajectory.png")
            print(f"  mission_trajectory.png (copied from results/{name})")
            break
    else:
        print("  mission_trajectory.png skipped (no candidate in results/)")

    print("done.")


if __name__ == "__main__":
    main()
