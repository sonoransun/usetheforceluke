# Propulsion evaluation — capability snapshot

Each cell reports the steady-state acceleration (m/s²) and the corresponding thrust-to-weight ratio at 1 g. Cells flagged `n/a` are not net-propulsive in the constant-thrust regime (e.g. Casimir cavity force is internal). All speculative parameter choices are listed at the end of this report.

## Vehicle catalogue

| Vehicle | Description | Mass (kg) | Power (W) |
| --- | --- | ---: | ---: |
| `cubesat_6u` | 6U CubeSat | 12 | 30 |
| `smallsat` | Small satellite (Starlink-class) | 500 | 714 |
| `crewed` | Crewed spacecraft (Crew Dragon-class) | 1.2e+04 | 1.06e+04 |
| `interplanetary` | Heavy interplanetary cruiser | 1e+05 | 6.45e+04 |
| `generation_ship` | Generation ship / large station | 1.00e+07 | 3.23e+06 |
| `city_ship` | Metropolitan city ship | 1.00e+09 | 1.62e+08 |

## Acceleration (m/s²)

| Vehicle | `parallel_plate_casimir` | `scaled_casimir` | `shaped_field_ansatz` | `heavy_element_lattice` | `stimulated_emission_array` | `antimatter_graviton` | `qgp_graviton` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cubesat_6u` | n/a | n/a | 2.5 | 2.5 | 2.5 | 2.5 | 2.5 |
| `smallsat` | n/a | n/a | 1.43 | 1.43 | 1.43 | 1.43 | 1.43 |
| `crewed` | n/a | n/a | 0.887 | 0.887 | 0.887 | 0.887 | 0.887 |
| `interplanetary` | n/a | n/a | 0.645 | 0.645 | 0.645 | 0.645 | 0.645 |
| `generation_ship` | n/a | n/a | 0.323 | 0.323 | 0.323 | 0.323 | 0.323 |
| `city_ship` | n/a | n/a | 0.162 | 0.162 | 0.162 | 0.162 | 0.162 |

## Thrust-to-weight (×1 g)

| Vehicle | `parallel_plate_casimir` | `scaled_casimir` | `shaped_field_ansatz` | `heavy_element_lattice` | `stimulated_emission_array` | `antimatter_graviton` | `qgp_graviton` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cubesat_6u` | n/a | n/a | 0.255 | 0.255 | 0.255 | 0.255 | 0.255 |
| `smallsat` | n/a | n/a | 0.146 | 0.146 | 0.146 | 0.146 | 0.146 |
| `crewed` | n/a | n/a | 0.0905 | 0.0905 | 0.0905 | 0.0905 | 0.0905 |
| `interplanetary` | n/a | n/a | 0.0658 | 0.0658 | 0.0658 | 0.0658 | 0.0658 |
| `generation_ship` | n/a | n/a | 0.033 | 0.033 | 0.033 | 0.033 | 0.033 |
| `city_ship` | n/a | n/a | 0.0165 | 0.0165 | 0.0165 | 0.0165 | 0.0165 |

⬆ = thrust-to-weight ≥ 1 (lift-off feasible against Earth gravity)

## Range scale (m)

| Vehicle | `parallel_plate_casimir` | `scaled_casimir` | `shaped_field_ansatz` | `heavy_element_lattice` | `stimulated_emission_array` | `antimatter_graviton` | `qgp_graviton` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cubesat_6u` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |
| `smallsat` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |
| `crewed` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |
| `interplanetary` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |
| `generation_ship` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |
| `city_ship` | 1.00e-07 | 1.00e-07 | 1 | 10 | 1 | 1 | 1 |

## Falloff: |F(r=1 km)| / |F(r=r_ref)|

Lower ratios mean faster decay with distance. This is where the four applicable models actually differ — the snapshot acceleration is matched by construction (same power, same V_REF), but the *shape* of the field around the vehicle is model-specific.

| Vehicle | `parallel_plate_casimir` | `scaled_casimir` | `shaped_field_ansatz` | `heavy_element_lattice` | `stimulated_emission_array` | `antimatter_graviton` | `qgp_graviton` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cubesat_6u` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |
| `smallsat` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |
| `crewed` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |
| `interplanetary` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |
| `generation_ship` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |
| `city_ship` | n/a | n/a | 0.00e+00 | 1.02e-04 | 1.00e-09 | 1.00e-06 | 1.00e-06 |

## Speculative parameter assumptions

These are the speculative coupling choices that drive the numbers above. They are *stated*, not derived from any physical theory.

### `parallel_plate_casimir`

- **geometry**: parallel plates, 1 cm², 100 nm separation
- **power_used**: 0 (passive geometry)
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0
- **note**: Real physics; reported for completeness but not propulsive
- **applicability**: not propulsive — Cavity force is internal; no centre-of-mass thrust in steady state

### `scaled_casimir`

- **geometry_factor**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0
- **note**: Speculative scaling; still not net-propulsive
- **applicability**: not propulsive — Same internal-force objection as parallel-plate Casimir

### `shaped_field_ansatz`

- **scaling**: A chosen so |F(r=σ)| = power / V_REF
- **sigma_m**: 1.0
- **amplitude**: 49.461638121003844
- **v_ref_mps**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0

### `heavy_element_lattice`

- **scaling**: κ chosen so |F(r=r_ref)| = power / V_REF
- **r_ref_m**: 10.0
- **softening_m**: 1.0
- **coupling_kappa**: 3045.1123131996296
- **v_ref_mps**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0

### `stimulated_emission_array`

- **scaling**: amplitude chosen so |F(r=r_ref)| = power / V_REF for a single emitter
- **r_ref_m**: 1.0
- **wavenumber**: 1.0
- **amplitude**: 3.872983346207417
- **v_ref_mps**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0

### `antimatter_graviton`

- **scaling**: Γ chosen so |F(r=r_ref)| = (power · η) / V_REF
- **r_ref_m**: 1.0
- **screening_lambda_m**: 1000000.0
- **coupling_g**: 1.0
- **efficiency_eta**: 1.0
- **annihilation_rate_gamma**: 2.50000000000125
- **probe_mass_kg**: 12.0
- **v_ref_mps**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0

### `qgp_graviton`

- **scaling**: graviton_yield chosen so |F(r=r_ref)| = power / V_REF
- **r_ref_m**: 1.0
- **screening_lambda_m**: 1000000.0
- **coupling_g**: 1.0
- **temperature_MeV**: 200.0
- **volume_m3**: 2.2894284851066637
- **qgp_energy_density_J_per_m3**: 5.1599999149407994e+35
- **qgp_total_energy_J**: 1.1813450788413428e+36
- **graviton_yield_eta**: 4.2324635617109224e-27
- **annihilation_rate_gamma**: 2.50000000000125
- **v_ref_mps**: 1.0
- **vehicle_power_W**: 30.0
- **vehicle_mass_kg**: 12.0
- **anchored_components**: Stefan–Boltzmann ε(T) with lattice-QCD g_eff(T)
- **speculative_components**: graviton emission coupling, Yukawa potential form, conversion to thrust

