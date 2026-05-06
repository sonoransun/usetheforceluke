"""Representative vehicle scales spanning ~10⁸× in mass.

Power budgets are engineering placeholders: P ≈ P0 · (m/m0)^0.85, reflecting
sub-linear scaling of usable onboard power with dry mass (more reactor mass
helps, but radiator/structure overhead grows faster than payload).
"""

from __future__ import annotations

from dataclasses import dataclass

# Anchor: a 12 kg CubeSat with 30 W onboard power.
_M0 = 12.0  # kg
_P0 = 30.0  # W
_BETA = 0.85


@dataclass(frozen=True, slots=True)
class Vehicle:
    """A representative vehicle scale."""

    key: str
    description: str
    mass_kg: float
    power_w: float

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError(f"{self.key}: mass_kg must be positive")
        if self.power_w <= 0:
            raise ValueError(f"{self.key}: power_w must be positive")


def power_budget(mass_kg: float) -> float:
    """Engineering-placeholder power budget (W) for a vehicle of given dry mass (kg)."""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    return _P0 * (mass_kg / _M0) ** _BETA


def _v(key: str, description: str, mass_kg: float) -> Vehicle:
    return Vehicle(key=key, description=description, mass_kg=mass_kg, power_w=power_budget(mass_kg))


VEHICLES: dict[str, Vehicle] = {
    "cubesat_6u": _v("cubesat_6u", "6U CubeSat", 12.0),
    "smallsat": _v("smallsat", "Small satellite (Starlink-class)", 500.0),
    "crewed": _v("crewed", "Crewed spacecraft (Crew Dragon-class)", 12_000.0),
    "interplanetary": _v("interplanetary", "Heavy interplanetary cruiser", 100_000.0),
    "generation_ship": _v("generation_ship", "Generation ship / large station", 1.0e7),
    "city_ship": _v("city_ship", "Metropolitan city ship", 1.0e9),
}
