"""Technical diagnostic figures: anchored physics, Yukawa screening, conservation, field shapes.

These are *physics* figures (not snapshot/mission marketing) intended to make
the framework's anchored physics and numerical rigour visible. Each helper is
a stateless function that returns a ``Figure`` so it can be unit-tested without
disk IO. ``notebooks/04_technical_figures.py`` is the canonical driver that
calls each one and writes the PNGs to ``assets/``.

Heavy deps (matplotlib) are lazy-imported per function — same convention as the
rest of ``viz/``. Requires the ``[viz]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from usetheforce._schwarzschild import G_NEWTON, gr_hover_factor, schwarzschild_radius
from usetheforce.antimatter import AntimatterGravitonField
from usetheforce.blackhole import SchwarzschildGravity
from usetheforce.fields import RegularGrid3D
from usetheforce.missions.vehicles import VEHICLES, Vehicle
from usetheforce.protocol import ForceField
from usetheforce.qgp.source import (
    DELTA_T_MEV,
    G_HRG,
    G_QGP,
    T_C_MEV,
    g_effective,
)
from usetheforce.trajectories import integrate

# Solar mass (kg) — match the adapter default.
_M_SUN_KG: float = 1.98892e30

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# 1 MeV → K (anchored constant for the QGP figures).
_MEV_TO_K: float = 1e6 * 1.602176634e-19 / 1.380649e-23


def plot_g_effective_crossover(
    T_min_mev: float = 10.0,
    T_max_mev: float = 500.0,
    n: int = 200,
) -> Figure:
    """Plot the lattice-QCD-flavoured ``g_eff(T)`` deconfinement crossover.

    Anchored physics: the curve interpolates between the hadron-resonance-gas
    plateau ``g_HRG ≈ 3`` (T ≪ T_c) and the QGP plateau ``g_QGP ≈ 47.5``
    (T ≫ T_c) via a tanh of width ``ΔT ≈ 20 MeV`` centred on the lattice value
    ``T_c ≈ 155 MeV``.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    T_mev = np.geomspace(T_min_mev, T_max_mev, n)
    T_K = T_mev * _MEV_TO_K
    g = np.array([g_effective(t) for t in T_K])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.semilogx(T_mev, g, color="#1f77b4", linewidth=2.5, label="g_eff(T)")
    ax.axhline(G_HRG, color="#aaaaaa", linestyle=":", label=f"g_HRG = {G_HRG}")
    ax.axhline(G_QGP, color="#aaaaaa", linestyle=":", label=f"g_QGP = {G_QGP}")
    ax.axvline(
        T_C_MEV, color="#d4a017", linestyle="--", linewidth=1.5, label=f"T_c = {T_C_MEV} MeV"
    )
    ax.axvspan(T_C_MEV - DELTA_T_MEV, T_C_MEV + DELTA_T_MEV, color="#d4a017", alpha=0.15)
    ax.set_xlabel("Temperature (MeV, log scale)")
    ax.set_ylabel("g_eff(T)")
    ax.set_title("Lattice-QCD-flavoured deconfinement crossover (anchored physics)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_yukawa_screening_sweep(
    lambdas_m: list[float] | None = None,
    r_min_m: float = 0.1,
    r_max_m: float = 1e4,
    n: int = 200,
) -> Figure:
    """Plot ``|F(r)|`` for the antimatter graviton model across a sweep of λ.

    All curves use the same ``Γ = g = m_probe = 1`` so the only difference is
    the screening length λ. As λ → ∞ the Yukawa shape collapses to a clean
    1/r² (the Newtonian-like limit). Anchored: the Yukawa form. Speculative:
    graviton coupling itself.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    lambdas = lambdas_m if lambdas_m is not None else [1.0, 10.0, 100.0, 1000.0, 1e6]
    rs = np.geomspace(r_min_m, r_max_m, n)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(lambdas):
        ff = AntimatterGravitonField(
            source=(0.0, 0.0, 0.0),
            gamma=1.0,
            coupling=1.0,
            screening=lam,
            probe_mass=1.0,
        )
        mags = np.empty(rs.size)
        probe = np.zeros(3)
        for k, r in enumerate(rs):
            probe[0] = r
            mags[k] = float(np.linalg.norm(ff.force(0.0, probe)))
        color = cmap(i / max(len(lambdas) - 1, 1))
        label = f"λ = {lam:g} m"
        ax.loglog(rs, mags, color=color, linewidth=2, label=label)
    # Reference 1/r² line (the λ→∞ limit shape).
    ref = rs[0] ** 2 / rs**2
    ax.loglog(rs, ref, color="#888888", linestyle="--", linewidth=1, label="1/r² reference")
    ax.set_xlabel("Distance r (m)")
    ax.set_ylabel("|F(r)| (N), Γ=g=m=1")
    ax.set_title("Yukawa graviton screening sweep — λ→∞ recovers inverse-square")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return fig


def plot_conservation_drift(
    models: dict[str, ForceField] | None = None,
    radius_m: float = 2.0,
    t_periods: float = 1.0,
    n_eval: int = 400,
) -> Figure:
    """Plot |ΔE/E₀| over time for several conservative speculative models.

    For each model, integrate a circular orbit (radius ``radius_m``) for
    ``t_periods`` periods using the model's natural circular speed; plot the
    relative energy drift on a log-y axis. Demonstrates the framework's
    <1e-6 conservation claim concretely.

    If ``models`` is None, builds a sensible default trio from the conservative
    speculative models (``ShapedFieldAnsatz``, ``HeavyElementLattice``,
    ``AntimatterGravitonField``).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from usetheforce.qfield import HeavyElementLattice, ShapedFieldAnsatz  # noqa: PLC0415

    if models is None:
        # Each model is calibrated so a circular orbit at r=2 has v_circ ≈ 1 m/s
        # — keeps the magnitudes comparable across the panel.
        models = {
            "shaped_field": ShapedFieldAnsatz(amplitude=2.0, sigma=2.0),
            "heavy_element": HeavyElementLattice(
                sites=[[0.0, 0.0, 0.0]], strengths=[1.0], coupling=4.0, softening=0.5
            ),
            "antimatter_graviton": AntimatterGravitonField(
                source=(0.0, 0.0, 0.0),
                gamma=4.0,
                coupling=1.0,
                screening=1e15,  # effectively 1/r²
                probe_mass=1.0,
            ),
        }

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, (name, ff) in enumerate(models.items()):
        # Probe at radius_m; numerical |F(r)| → v_circ via |F| = m·v²/r.
        probe = np.array([radius_m, 0.0, 0.0])
        f_mag = float(np.linalg.norm(ff.force(0.0, probe)))
        if f_mag <= 0:
            continue
        v_circ = float(np.sqrt(f_mag * radius_m))  # m=1
        period = 2.0 * np.pi * radius_m / v_circ
        traj = integrate(
            ff,
            mass=1.0,
            r0=[radius_m, 0.0, 0.0],
            v0=[0.0, v_circ, 0.0],
            t_span=(0.0, t_periods * period),
            n_eval=n_eval,
        )
        try:
            energy = traj.total_energy(ff)
        except ValueError:
            continue
        e0 = float(energy[0])
        if e0 == 0:
            continue
        drift = np.abs((energy - e0) / e0) + 1e-18  # add tiny floor for log axis
        ax.semilogy(traj.t / period, drift, color=cmap(i), linewidth=1.8, label=name)
    ax.axhline(1e-6, color="#aaaaaa", linestyle="--", linewidth=1, label="1e-6 floor (claim)")
    ax.set_xlabel(f"t / orbital period (radius = {radius_m} m)")
    ax.set_ylabel("|ΔE / E₀|  (log)")
    ax.set_title("Energy conservation drift — DOP853 + rtol=1e-10 keeps drift < 1e-6")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_blackhole_required_thrust(
    bh_mass_solar: float = 10.0,
    vehicle_keys: tuple[str, ...] = ("cubesat_6u", "interplanetary", "city_ship"),
    ratio_min: float = 1.001,
    ratio_max: float = 1000.0,
    n: int = 200,
    v_ref_mps: float = 1.0,
) -> Figure:
    """Required hover thrust vs. R/r_s for several vehicles, Newtonian + GR.

    Each vehicle contributes three curves: Newtonian required thrust
    ``G M m_probe / R²`` (solid), GR-corrected ``× 1/√(1 − r_s/R)`` (dashed),
    and the horizontal supplied thrust ``power / V_REF`` (dotted). The GR curve
    diverges as ``R → r_s⁺`` — the headline of the blackhole-explorer mode.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    bh_mass_kg = bh_mass_solar * _M_SUN_KG
    r_s = schwarzschild_radius(bh_mass_kg)
    ratios = np.geomspace(ratio_min, ratio_max, n)
    radii = ratios * r_s

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    palette = ("#1f77b4", "#d4a017", "#d62728", "#2ca02c", "#9467bd", "#8c564b")
    for i, vk in enumerate(vehicle_keys):
        veh: Vehicle = VEHICLES[vk]
        f_newt = G_NEWTON * bh_mass_kg * veh.mass_kg / (radii * radii)
        f_gr = np.array(
            [f_newt[k] * gr_hover_factor(radii[k], r_s) for k in range(radii.size)]
        )
        supplied = veh.power_w / v_ref_mps
        color = palette[i % len(palette)]
        ax.loglog(ratios, f_newt, color=color, linewidth=2.0, label=f"{veh.key} — Newtonian")
        ax.loglog(ratios, f_gr, color=color, linewidth=1.6, linestyle="--", label=f"{veh.key} — GR")
        ax.axhline(supplied, color=color, linewidth=1, linestyle=":", alpha=0.6)
    ax.set_xlabel("R / r_s  (log)")
    ax.set_ylabel("Hover thrust required [N]  (log) — dotted = supplied (power / V_REF)")
    ax.set_title(
        f"Required hover thrust vs. radius — {bh_mass_solar:g} M_sun BH "
        f"(r_s = {r_s / 1000.0:.2f} km)"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_blackhole_field_heatmap(
    bh_mass_solar: float = 10.0,
    span_rs: float = 4.0,
    probe_mass_kg: float = 1.0,
    grid_points: int = 121,
) -> Figure:
    """2D ``|F(x, y, 0)|`` heatmap for a Schwarzschild source with the horizon overlaid.

    The probe is treated as ``probe_mass_kg = 1 kg`` for absolute Newton labelling.
    The event horizon ``R = r_s`` is drawn as a red circle. Inside the horizon the
    helper paints a sentinel (NaN) so the singular region is visually distinct.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LogNorm  # noqa: PLC0415
    from matplotlib.patches import Circle  # noqa: PLC0415

    bh_mass_kg = bh_mass_solar * _M_SUN_KG
    r_s = schwarzschild_radius(bh_mass_kg)
    half = span_rs * r_s
    spacing = (2.0 * half) / (grid_points - 1)
    grid = RegularGrid3D(
        origin=(-half, -half, -0.5 * spacing),
        spacing=(spacing, spacing, spacing),
        shape=(grid_points, grid_points, 3),
    )
    ff = SchwarzschildGravity(
        mass_kg=bh_mass_kg,
        probe_mass_kg=probe_mass_kg,
        horizon_softening_m=0.0,
    )
    # Direct sampling — _slice_field_xy raises on probes inside the horizon; we
    # populate manually so the inner region becomes NaN rather than blowing up.
    nx, ny = grid.shape[0], grid.shape[1]
    out = np.full((nx, ny), np.nan, dtype=float)
    points = grid.points()
    nz = grid.shape[2]
    z_index = nz // 2
    for i in range(nx):
        for j in range(ny):
            flat = (i * ny + j) * nz + z_index
            probe = points[flat]
            try:
                out[i, j] = float(np.linalg.norm(ff.force(0.0, probe)))
            except ValueError:
                out[i, j] = np.nan

    finite = out[np.isfinite(out) & (out > 0)]
    vmin = float(finite.min()) if finite.size else 1e-12
    vmax = float(finite.max()) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = (-half, half, -half, half)
    im = ax.imshow(
        out.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    horizon = Circle((0.0, 0.0), r_s, fill=False, color="red", linewidth=2.0, label="event horizon r_s")
    ax.add_patch(horizon)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Schwarzschild |F| on a (x, y, 0) slice — {bh_mass_solar:g} M_sun "
        f"(r_s = {r_s / 1000.0:.2f} km)"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label="|F| (N, probe = 1 kg)")
    fig.tight_layout()
    return fig


def plot_blackhole_shortfall_matrix(
    bh_mass_solar: float = 10.0,
    ratios: tuple[float, ...] = (1.001, 1.01, 1.1, 2.0, 10.0, 100.0),
    v_ref_mps: float = 1.0,
) -> Figure:
    """Heatmap of ``log10(required / supplied)`` over (vehicle, R/r_s).

    Rows are the six entries in ``VEHICLES`` in declared order (cubesat → city_ship);
    columns are the supplied R/r_s ratios. Required thrust is the Newtonian
    hover thrust ``G M m / R²``; supplied is ``power / V_REF``. The colour map
    is sequential — every cell here is positive (a shortfall), often by many
    orders of magnitude.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    bh_mass_kg = bh_mass_solar * _M_SUN_KG
    r_s = schwarzschild_radius(bh_mass_kg)
    veh_keys = list(VEHICLES.keys())
    matrix = np.empty((len(veh_keys), len(ratios)), dtype=float)
    for i, vk in enumerate(veh_keys):
        veh = VEHICLES[vk]
        supplied = veh.power_w / v_ref_mps
        for j, ratio in enumerate(ratios):
            R = ratio * r_s
            required = G_NEWTON * bh_mass_kg * veh.mass_kg / (R * R)
            matrix[i, j] = np.log10(required / supplied)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(matrix, cmap="inferno", aspect="auto")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_yticks(range(len(veh_keys)))
    ax.set_yticklabels(veh_keys)
    ax.set_xlabel("R / r_s")
    ax.set_ylabel("vehicle")
    ax.set_title(
        f"log₁₀(required_hover / supplied) — {bh_mass_solar:g} M_sun BH "
        f"(every cell is a shortfall)"
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.1f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.mean() else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, shrink=0.85, label="log₁₀(shortfall ratio)")
    fig.tight_layout()
    return fig


def _slice_field_xy(ff: ForceField, grid: RegularGrid3D, t: float = 0.0) -> np.ndarray:
    """Return |F(x, y, 0)| as a (Nx, Ny) array — z is taken at the central slice."""
    nx, ny, nz = grid.shape
    points = grid.points()  # (N, 3)
    z_index = nz // 2
    out = np.empty((nx, ny), dtype=float)
    probe = np.empty(3, dtype=float)
    for i in range(nx):
        for j in range(ny):
            flat = (i * ny + j) * nz + z_index
            probe[:] = points[flat]
            out[i, j] = float(np.linalg.norm(ff.force(t, probe)))
    return out


def plot_field_heatmap_grid(
    models: dict[str, ForceField] | None = None,
    grid: RegularGrid3D | None = None,
) -> Figure:
    """Plot a 2×2 grid of |F(x, y, 0)| heatmaps — one per applicable model.

    Shared logarithmic colour scale across all four panels so the relative
    spatial structures are directly comparable. The point is to make the
    snapshot's ``falloff_ratio`` column visceral.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LogNorm  # noqa: PLC0415

    from usetheforce.qfield import (  # noqa: PLC0415
        HeavyElementLattice,
        ShapedFieldAnsatz,
        StimulatedEmissionArray,
    )

    if grid is None:
        grid = RegularGrid3D(
            origin=(-3.05, -3.05, -0.05),
            spacing=(0.15, 0.15, 0.1),
            shape=(41, 41, 3),
        )
    if models is None:
        models = {
            "ShapedFieldAnsatz": ShapedFieldAnsatz(amplitude=2.0, sigma=1.0),
            "HeavyElementLattice": HeavyElementLattice(
                sites=[[0.0, 0.0, 0.0]],
                strengths=[1.0],
                coupling=2.0,
                softening=0.3,
            ),
            "StimulatedEmissionArray": StimulatedEmissionArray(
                positions=[[0.0, 0.0, 0.0]],
                amplitudes=[1.5],
                phases=[0.0],
                wavenumber=2.0,
                coupling=1.0,
                min_distance_m=0.05,
            ),
            "AntimatterGravitonField": AntimatterGravitonField(
                source=(0.0, 0.0, 0.0),
                gamma=1.0,
                coupling=1.0,
                screening=10.0,
                probe_mass=1.0,
            ),
        }

    # Sample once per model on the same xy-slice.
    fields = {}
    for name, ff in models.items():
        try:
            fields[name] = _slice_field_xy(ff, grid)
        except ValueError:
            # Probe drifted into a singular point — fill with a small sentinel
            # so the figure still renders even if one model can't be sampled.
            fields[name] = np.full(grid.shape[:2], np.nan)

    valid = [arr[np.isfinite(arr) & (arr > 0)] for arr in fields.values()]
    valid = [v for v in valid if v.size > 0]
    vmin = max(min((v.min() for v in valid), default=1e-12), 1e-12)
    vmax = max((v.max() for v in valid), default=1.0)

    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    items = list(fields.items())
    extent = (
        grid.origin[0],
        grid.origin[0] + grid.spacing[0] * (grid.shape[0] - 1),
        grid.origin[1],
        grid.origin[1] + grid.spacing[1] * (grid.shape[1] - 1),
    )
    norm = LogNorm(vmin=vmin, vmax=vmax)
    last_im = None
    for ax, (name, arr) in zip(axs.flat, items, strict=True):
        last_im = ax.imshow(
            arr.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            norm=norm,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle("Spatial |F(x, y, 0)| for each applicable speculative model (shared log colour)")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axs.ravel().tolist(), shrink=0.85, label="|F| (N)")
        cbar.ax.tick_params(labelsize=8)
    return fig
