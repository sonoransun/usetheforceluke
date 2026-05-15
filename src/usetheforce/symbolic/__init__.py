"""SymPy derivations — source of truth for the numerical kernels."""

from usetheforce.symbolic.casimir import casimir_pressure_expr, casimir_pressure_lambdified
from usetheforce.symbolic.conservation import kinetic_energy_expr, total_energy_expr
from usetheforce.symbolic.graviton import (
    graviton_force_radial_expr,
    graviton_force_radial_lambdified,
    graviton_potential_expr,
    graviton_potential_lambdified,
)
from usetheforce.symbolic.heavy_elements import (
    heavy_element_force_radial_expr,
    heavy_element_force_radial_lambdified,
    heavy_element_potential_expr,
    heavy_element_potential_lambdified,
)
from usetheforce.symbolic.negative_mass import (
    anti_chirp_dfdt_expr,
    anti_chirp_dfdt_lambdified,
    bondi_acceleration_expr,
    bondi_acceleration_lambdified,
    gw_dipole_power_expr,
    gw_dipole_power_lambdified,
    gw_quadrupole_power_expr,
    gw_quadrupole_power_lambdified,
)
from usetheforce.symbolic.qgp import (
    g_effective_expr,
    g_effective_lambdified,
    sb_energy_density_expr,
    sb_energy_density_lambdified,
)
from usetheforce.symbolic.schwarzschild import (
    gr_hover_factor_expr,
    gr_hover_factor_lambdified,
    newtonian_force_radial_expr,
    newtonian_force_radial_lambdified,
    newtonian_potential_expr,
    newtonian_potential_lambdified,
    schwarzschild_radius_expr,
    schwarzschild_radius_lambdified,
)
from usetheforce.symbolic.stimulated_emission import (
    single_emitter_force_radial_expr,
    single_emitter_force_radial_lambdified,
    single_emitter_intensity_expr,
    single_emitter_intensity_lambdified,
    single_emitter_potential_expr,
)

__all__ = [
    "anti_chirp_dfdt_expr",
    "anti_chirp_dfdt_lambdified",
    "bondi_acceleration_expr",
    "bondi_acceleration_lambdified",
    "casimir_pressure_expr",
    "casimir_pressure_lambdified",
    "g_effective_expr",
    "g_effective_lambdified",
    "gr_hover_factor_expr",
    "gr_hover_factor_lambdified",
    "graviton_force_radial_expr",
    "graviton_force_radial_lambdified",
    "graviton_potential_expr",
    "graviton_potential_lambdified",
    "gw_dipole_power_expr",
    "gw_dipole_power_lambdified",
    "gw_quadrupole_power_expr",
    "gw_quadrupole_power_lambdified",
    "heavy_element_force_radial_expr",
    "heavy_element_force_radial_lambdified",
    "heavy_element_potential_expr",
    "heavy_element_potential_lambdified",
    "kinetic_energy_expr",
    "newtonian_force_radial_expr",
    "newtonian_force_radial_lambdified",
    "newtonian_potential_expr",
    "newtonian_potential_lambdified",
    "sb_energy_density_expr",
    "sb_energy_density_lambdified",
    "schwarzschild_radius_expr",
    "schwarzschild_radius_lambdified",
    "single_emitter_force_radial_expr",
    "single_emitter_force_radial_lambdified",
    "single_emitter_intensity_expr",
    "single_emitter_intensity_lambdified",
    "single_emitter_potential_expr",
    "total_energy_expr",
]
