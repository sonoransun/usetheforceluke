"""Generate the technical-diagnostic figures embedded in README.md.

Deterministic — no RNG, no time-dependent inputs. Re-runnable; outputs land in
``assets/`` alongside the marketing-shaped figures from
``03_documentation_figures.py``.

Outputs:

- ``g_eff_crossover.png`` — anchored deconfinement curve
- ``yukawa_screening.png`` — λ-sweep showing 1/r² recovery
- ``conservation_drift.png`` — proof of <1e-6 energy drift
- ``field_heatmaps.png`` — 2×2 spatial |F| panels

Run with: ``.venv/bin/python notebooks/04_technical_figures.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from usetheforce.viz.diagnostics import (
    plot_conservation_drift,
    plot_field_heatmap_grid,
    plot_g_effective_crossover,
    plot_yukawa_screening_sweep,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    print(f"writing technical figures to {ASSETS}/")

    fig = plot_g_effective_crossover()
    fig.savefig(ASSETS / "g_eff_crossover.png", dpi=144, bbox_inches="tight")
    print("  g_eff_crossover.png")

    fig = plot_yukawa_screening_sweep()
    fig.savefig(ASSETS / "yukawa_screening.png", dpi=144, bbox_inches="tight")
    print("  yukawa_screening.png")

    fig = plot_conservation_drift()
    fig.savefig(ASSETS / "conservation_drift.png", dpi=144, bbox_inches="tight")
    print("  conservation_drift.png")

    fig = plot_field_heatmap_grid()
    fig.savefig(ASSETS / "field_heatmaps.png", dpi=144, bbox_inches="tight")
    print("  field_heatmaps.png")

    print("done.")


if __name__ == "__main__":
    main()
