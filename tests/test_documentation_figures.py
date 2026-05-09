"""Smoke tests for the README-figure render helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from usetheforce.control.controllers import ProportionalGuidance
from usetheforce.control.field import ControlledThrustField
from usetheforce.control.power import VehiclePowerState
from usetheforce.missions import ALL_ADAPTERS, MISSIONS, VEHICLES, evaluate_snapshot, run_mission
from usetheforce.missions.long_range import LongRangeMissionResult
from usetheforce.missions.render import (
    plot_falloff_vs_distance,
    plot_mission_delta_v_bar,
    plot_twr_heatmap,
    plot_vehicle_scale_strip,
)
from usetheforce.trajectories import integrate
from usetheforce.viz.control_animations import (
    animate_long_range_mission,
    animate_model_comparison,
)


def test_vehicle_scale_strip_returns_figure_with_six_points() -> None:
    fig = plot_vehicle_scale_strip(VEHICLES)
    ax = fig.axes[0]
    # One line + one scatter — the scatter has six offsets.
    scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
    assert scatter_collections, "expected at least one scatter collection"
    n_points = len(scatter_collections[0].get_offsets())
    assert n_points == len(VEHICLES) == 6


def test_twr_heatmap_marks_casimir_inapplicable() -> None:
    matrix = evaluate_snapshot(VEHICLES, ALL_ADAPTERS)
    fig = plot_twr_heatmap(matrix, VEHICLES)
    texts = [t.get_text() for t in fig.axes[0].texts]
    # Each Casimir cell × 6 vehicles = 12 "n/a" labels.
    assert texts.count("n/a") == 12


def test_falloff_vs_distance_one_line_per_model() -> None:
    sma = VEHICLES["smallsat"]
    fields = {
        mk: ALL_ADAPTERS[mk](sma, sma.power_w).field
        for mk in [
            "shaped_field_ansatz",
            "heavy_element_lattice",
            "stimulated_emission_array",
            "antimatter_graviton",
        ]
    }
    fig = plot_falloff_vs_distance(fields, r_min_m=0.1, r_max_m=1e3, n_points=50)
    assert len(fig.axes[0].lines) == 4


def test_mission_delta_v_bar_one_bar_per_result() -> None:
    pairs = [
        ("smallsat", "shaped_field_ansatz", "free_burn_100s"),
        ("crewed", "stimulated_emission_array", "leo_raise_100s"),
    ]
    results = [
        run_mission(VEHICLES[vk], mk, ALL_ADAPTERS[mk], MISSIONS[mission])
        for vk, mk, mission in pairs
    ]
    fig = plot_mission_delta_v_bar(results)
    # Number of bars in the BarContainer should match.
    bar_containers = [c for c in fig.axes[0].containers if hasattr(c, "patches")]
    assert bar_containers
    assert len(bar_containers[0].patches) == len(results)


# ---------------------------------------------------------------------------
# Animation smoke tests (notebooks/05_animations.py).
# ---------------------------------------------------------------------------


def _short_cruise(max_thrust_n: float = 1e4) -> LongRangeMissionResult:
    """Build a fast, synthetic LongRangeMissionResult for animation smoke tests.

    The production ``heliocentric_cruise`` factory hard-codes ``rtol=1e-10``
    which interacts badly with the controller framework's FD-velocity estimator
    on long timespans. This fixture composes the same proportional-guidance +
    Sun-gravity pipeline directly via ``integrate()`` with relaxed tolerances
    and a short timespan so tests stay quick.
    """
    GM_SUN = 1.32712440018e20
    AU = 1.495978707e11
    burn_s = 5.0 * 86400.0
    mass = 1.0e5

    def sun_bg(r: np.ndarray) -> np.ndarray:
        rn = float(np.linalg.norm(r))
        return -GM_SUN * r / rn**3 if rn > 0 else np.zeros(3)

    target = np.array([1.5 * AU, 0.0, 0.0])
    ctrl = ProportionalGuidance(target_position=target, gain=1e-6, max_thrust_n=max_thrust_n)
    field = ControlledThrustField(
        controller=ctrl,
        mass_kg=mass,
        background=sun_bg,
        power=VehiclePowerState(initial_energy_j=1e16, instantaneous_power_w=1e10),
    )

    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, float(np.sqrt(GM_SUN / AU)), 0.0])
    traj = integrate(
        field,
        mass=mass,
        r0=r0,
        v0=v0,
        t_span=(0.0, burn_s),
        n_eval=24,
        rtol=1e-6,
        atol=1e-3,
    )

    log_t = np.array([e[0] for e in field.thrust_log], dtype=float)
    log_F = np.array([e[1] for e in field.thrust_log], dtype=float)
    thrust_arr = np.empty((traj.t.size, 3))
    for i in range(3):
        thrust_arr[:, i] = np.interp(traj.t, log_t, log_F[:, i]) if log_t.size > 0 else 0.0
    mech = np.einsum("ij,ij->i", thrust_arr, traj.v)
    cum = np.concatenate(([0.0], np.cumsum(np.abs(mech[:-1]) * np.diff(traj.t))))
    power_hist = np.maximum(0.0, 1e16 - cum)

    return LongRangeMissionResult(
        name=f"cruise_max{max_thrust_n:.0e}",
        trajectory=traj,
        delta_v_mps=float(np.linalg.norm(traj.v[-1] - v0)),
        burn_time_s=burn_s,
        energy_used_j=float(cum[-1]),
        peak_g=max_thrust_n / (mass * 9.80665),
        thrust_history_n=thrust_arr,
        power_history_j=power_hist,
        controller_metadata={"controller": "ProportionalGuidance"},
        background="sun_central",
        target_metric={},
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_animate_long_range_mission_returns_funcanimation() -> None:
    result = _short_cruise()
    anim = animate_long_range_mission(result, output=None, fps=10, every=2)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


def test_animate_long_range_mission_writes_nonempty_gif(tmp_path: Path) -> None:
    result = _short_cruise()
    out = tmp_path / "dashboard.gif"
    ret = animate_long_range_mission(result, output=out, fps=10, every=4)
    assert ret is None
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_animate_model_comparison_returns_funcanimation() -> None:
    runs = [_short_cruise(mt) for mt in (1e3, 1e4)]
    anim = animate_model_comparison(runs, output=None, fps=10, every=2)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


def test_animate_model_comparison_writes_nonempty_gif(tmp_path: Path) -> None:
    runs = [_short_cruise(mt) for mt in (1e3, 1e4)]
    out = tmp_path / "compare.gif"
    ret = animate_model_comparison(runs, output=out, fps=10, every=4, labels=["lo", "hi"])
    assert ret is None
    assert out.exists()
    assert out.stat().st_size > 0
