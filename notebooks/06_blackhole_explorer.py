"""Blackhole explorer demo — what propulsion is needed to hover near r_s?

EXCEPTIONALLY SPECULATIVE. The Schwarzschild gravity is anchored, but the
counter-drive itself is not — there is no known mechanism. This notebook
quantifies the *shortfall* between the thrust a representative vehicle's power
budget supplies and the thrust required to hover at a series of radii outside
the event horizon. Headline: as ``R → r_s``, every realistic vehicle loses.

Run as a script (``python notebooks/06_blackhole_explorer.py``) or convert to
a notebook with ``jupytext --to ipynb 06_blackhole_explorer.py``.
"""

# %% Imports
from __future__ import annotations

import os

import numpy as np
import scipy.constants as sc

from usetheforce._schwarzschild import gr_hover_factor, schwarzschild_radius
from usetheforce.missions import (
    VEHICLES,
    event_horizon_stationkeep,
    event_horizon_stationkeep_with_buffer,
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
M_SUN = 1.98892e30

# %% Choose a stellar-mass black hole and two extreme vehicles.
BH_MASS_KG = 10.0 * M_SUN
r_s = schwarzschild_radius(BH_MASS_KG)
print(f"Black hole mass: 10 M_sun = {BH_MASS_KG:.3e} kg")
print(f"Schwarzschild radius r_s = {r_s:.3e} m ({r_s / 1000.0:.2f} km)")
print()

VEHICLE_KEYS = ("cubesat_6u", "city_ship")
RATIOS = (1.001, 1.01, 1.1, 2.0, 10.0, 100.0)


def supplied_thrust_n(power_w: float, v_ref_mps: float = 1.0) -> float:
    """At V_REF = 1 m/s, F = P / V; matches the adapter convention."""
    return power_w / v_ref_mps


# %% Print the shortfall table for each vehicle.
for vk in VEHICLE_KEYS:
    veh = VEHICLES[vk]
    supplied = supplied_thrust_n(veh.power_w)
    print(f"--- {veh.key} | mass={veh.mass_kg:.3e} kg, P={veh.power_w:.3e} W ---")
    print(
        f"{'R/r_s':>8} | {'R [m]':>12} | "
        f"{'F_Newt [N]':>14} | {'F_GR [N]':>14} | "
        f"{'F_supplied [N]':>14} | {'shortfall':>12}"
    )
    for ratio in RATIOS:
        R = ratio * r_s
        # F_Newtonian = G M m_probe / R²
        F_newt = sc.G * BH_MASS_KG * veh.mass_kg / (R * R)
        F_gr = F_newt * gr_hover_factor(R, r_s)
        shortfall = F_gr / supplied if supplied > 0 else float("inf")
        print(
            f"{ratio:>8.3f} | {R:>12.3e} | "
            f"{F_newt:>14.3e} | {F_gr:>14.3e} | "
            f"{supplied:>14.3e} | {shortfall:>12.3e}"
        )
    print()


# %% Run an integrated stationkeep mission at hover_radius_factor=1.5.
print("Running event_horizon_stationkeep(city_ship, 10 M_sun, hover @ 1.5 r_s) ...")
result = event_horizon_stationkeep(
    black_hole_mass_kg=BH_MASS_KG,
    duration_s=60.0,
    vehicle=VEHICLES["city_ship"],
    hover_radius_factor=1.5,
    use_gr_hover_correction=True,
    gain=1e-3,
    max_thrust_n=1e15,
    initial_offset_m=10.0,
    n_eval=120,
)
tm = result.target_metric
print(f"  target_radius_m            = {tm['target_radius_m']:.3e}")
print(f"  required_hover_n (Newton)  = {tm['required_hover_force_newtonian_n']:.3e}")
print(f"  required_hover_n (GR)      = {tm['required_hover_force_gr_n']:.3e}")
print(f"  supplied_thrust_cap_n       = {tm['supplied_thrust_cap_n']:.3e}")
print(f"  achieved peak thrust [N]    = "
      f"{float(np.linalg.norm(result.thrust_history_n, axis=1).max()):.3e}")
print(f"  peak g-load                 = {result.peak_g:.3e}")
print()

# %% Optional matplotlib plot — render only if [viz] extra is installed.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ASSETS_DIR, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    t = result.trajectory.t
    thrust_mag = np.linalg.norm(result.thrust_history_n, axis=1)
    ax_top.plot(t, thrust_mag, label="achieved |F| [N]")
    ax_top.axhline(
        tm["required_hover_force_newtonian_n"],
        color="orange",
        linestyle="--",
        label="required (Newtonian) [N]",
    )
    ax_top.axhline(
        tm["required_hover_force_gr_n"],
        color="red",
        linestyle=":",
        label="required (GR hover) [N]",
    )
    ax_top.set_yscale("log")
    ax_top.set_ylabel("thrust [N]")
    ax_top.set_title(
        "Blackhole explorer — hover @ "
        f"{tm['hover_radius_factor']:.2f}·r_s, 10 M_sun BH"
    )
    ax_top.legend(loc="best", fontsize=8)

    r_mag = np.linalg.norm(result.trajectory.r, axis=1)
    ax_bot.plot(t, r_mag / tm["schwarzschild_radius_m"])
    ax_bot.axhline(1.0, color="black", linestyle="--", label="event horizon")
    ax_bot.set_ylabel("|r| / r_s")
    ax_bot.set_xlabel("time [s]")
    ax_bot.legend(loc="best", fontsize=8)

    out_path = os.path.join(ASSETS_DIR, "blackhole_explorer.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"matplotlib: wrote {out_path}")
except ImportError as exc:
    print(f"matplotlib skipped: {exc}")


# %% Extension: negative-mass buffer between craft and horizon.
# EXCEPTIONALLY SPECULATIVE: the buffer's repulsive gravity (sign-flipped
# Newtonian; requires the Bondi 1957 negative-inertial-mass premise) pushes
# the craft outward, partially offsetting Schwarzschild attraction. The two
# headline numbers are ``buffer_offset_ratio`` (how much of the BH pull the
# buffer cancels — 1.0 = perfect cancellation) and ``augmented_shortfall_ratio``
# (the *remaining* gap between supplied and required thrust after the buffer
# is accounted for).
print()
print("=== Buffer extension: negative-mass element between craft and horizon ===")
city = VEHICLES["city_ship"]
print(f"vehicle: {city.key} | mass={city.mass_kg:.3e} kg, P={city.power_w:.3e} W")
# Run a few buffer masses (held at the same buffer radius) and tabulate.
HOVER_FACTOR = 50.0  # 50 r_s — safe integration window
BUFFER_FACTOR = 49.0  # 1 r_s gap between buffer and craft
BUFFER_MASSES_SOLAR = (1.0e-4, 1.0e-2, 1.0, 10.0, 100.0)
print(
    f"hover_radius_factor = {HOVER_FACTOR}, buffer_radius_factor = {BUFFER_FACTOR}, "
    f"gap = {(HOVER_FACTOR - BUFFER_FACTOR):.2f} r_s"
)
print(
    f"{'m_buf [M☉]':>12} | {'offset_ratio':>14} | "
    f"{'net_req_GR [N]':>14} | {'shortfall':>14}"
)
for m_solar in BUFFER_MASSES_SOLAR:
    buf_result = event_horizon_stationkeep_with_buffer(
        black_hole_mass_kg=BH_MASS_KG,
        duration_s=0.05,
        vehicle=city,
        buffer_mass_neg_kg=m_solar * M_SUN,
        buffer_radius_factor=BUFFER_FACTOR,
        hover_radius_factor=HOVER_FACTOR,
        use_gr_hover_correction=True,
        gain=1e-6,
        max_thrust_n=1.0e15,
        initial_offset_m=10.0,
        n_eval=5,
    )
    tm_b = buf_result.target_metric
    print(
        f"{m_solar:>12.4g} | {tm_b['buffer_offset_ratio']:>14.4g} | "
        f"{tm_b['net_required_hover_force_gr_n']:>14.4e} | "
        f"{tm_b['augmented_shortfall_ratio']:>14.4g}"
    )
print()
print(
    "Read the table together: the buffer mass needed for full cancellation at "
    f"these factors is roughly M_BH·(Δ/R_craft)² = 10·(1/50)² = {10 * (1 / 50) ** 2:g} M☉. "
    "Below that, the buffer barely dents the shortfall; above it, the buffer is "
    "doing all the work and the speculative leap has migrated from 'we have a "
    "counter-drive' to 'we have a stellar-mass negative-mass appendage'."
)
