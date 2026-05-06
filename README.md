# Use the Force, Luke!

This speculative effort envisions a method of generating artificial gravity fields for propulsion, and calculating the trajectories possible.

As Avi Loeb mentioned in https://avi-loeb.medium.com/can-the-vacuum-drive-fuel-free-propulsion-e4983a86419c the holy grail for space travel lies in nullifying gravity.

## Design goals

This project provides three main avenues for novel propulsion mechanisms:
- Casimir effect scaled to useful force ranges, trading fuel efficiency for propulsion power
- Exotic quantum field shaping to generate anti-gravity forces using radioactive decay, stimulated emissions, and fine nano structure emitters
- Anti-matter conversion of energy directly into counter gravity forces with perfect efficiency

## Getting started

```bash
pip install -e ".[dev,viz,interactive,3d]"
pytest
```

Optional extras:
- `viz` — matplotlib (static figures)
- `interactive` — plotly (browser-interactive 2D/3D)
- `3d` — pyvista + trame (3D animation of fields and trajectories)

See `CLAUDE.md` for stack conventions and `src/usetheforce/` for the package layout.

## Theoretical force options

The package implements six `ForceField` models across the three avenues. Each one exposes the same protocol (`force(t, r)`, `potential(r)`, `metadata`) and is tested against an analytic limit.

| Avenue | Model | Speculative? | Anchor |
| --- | --- | --- | --- |
| `casimir` | `ParallelPlateCasimir` | no | textbook Casimir 1948 formula `F/A = -π²ℏc/(240a⁴)` |
| `casimir` | `ScaledCasimir` | yes | parallel-plate × geometry factor `g` |
| `qfield` | `ShapedFieldAnsatz` | yes | Gaussian-well potential |
| `qfield` | `HeavyElementLattice` | yes | Plummer-softened multi-site dipole sum; single-site limit ⇒ 1/r² |
| `qfield` | `StimulatedEmissionArray` | yes | coherent phased-array intensity `I=|Σₙ Aₙ e^(i(kRₙ+φₙ))/Rₙ|²`; single-emitter limit ⇒ A²/r² |
| `antimatter` | `AntimatterCounterGravity` | yes | local cancellation of supplied background g(r) |
| `antimatter` | `AntimatterGravitonField` | yes | Yukawa potential `-gΓ e^(-r/λ)/r`; λ→∞ limit ⇒ 1/r² |

Speculative models all carry `metadata["speculative"] = True`; tests assert the marker is never silently dropped.

## Propulsion evaluation

Run `python notebooks/02_evaluation.py` to generate a (vehicle × model) capability snapshot and a set of integrated missions across six vehicle scales (`cubesat_6u` → `city_ship`, ~10⁸× in mass). Outputs land in `results/`:

- `snapshot.md` — acceleration, thrust-to-weight, range scale, and the **falloff ratio at 1 km** for each (vehicle, model) cell. The four free-flight models are normalized to the same input power (so peak acceleration is matched by construction); the **falloff column is where they differ** — heavy-element lattice (1/r²) reaches farthest, then graviton Yukawa (≈1/r² inside its screening length), then stimulated emission (1/r³), then the Gaussian shaped field (essentially zero past a few σ).
- `accel_vs_mass.png` — log–log scatter of acceleration vs vehicle mass.
- `mission_*.{png,html}` — per-mission trajectory artifacts for six selected (vehicle, model, mission) triples spanning a 100 s smallsat free-burn up to a 1-year city-ship interstellar boost.
- `missions.md` — Δv, peak g, thrust, and energy for each integrated mission.

Both Casimir variants are flagged `applicable=False` for free-flight propulsion (cavity force is internal in steady state); the report shows the static physics for completeness. All speculative parameter choices used to derive the numbers are listed at the bottom of `snapshot.md`.

