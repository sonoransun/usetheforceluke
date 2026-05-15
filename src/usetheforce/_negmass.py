"""Internal helpers shared by negative-mass-binary ``ForceField`` implementations.

The four ``negmass`` models (``BondiRunawayPair``, ``AntiChirpBinary``,
``NegativeTotalMassBinary``, ``DipoleGravitonRadiator``) all share a small
arithmetic core derived from Bondi's 1957 solutions and the Peters–Mathews
gravitational-radiation formula. Centralising the closed-form expressions here
mirrors how ``_yukawa.py`` and ``_schwarzschild.py`` serve their respective
avenues.

Bondi self-acceleration of a zero-net-mass pair (one ``+m``, one ``-m`` rigidly
separated by ``d``)::

    a = G · |m| / d²    (Bondi 1957)

The pair self-accelerates indefinitely in the direction from ``-m`` toward
``+m``. Energy and momentum are *not* conserved for the centre-of-mass — that
is the load-bearing pathology of the negative-mass premise and the entire
reason the model is propulsion-interesting (Forward 2015) and physically
suspect.

References
----------
- Bondi, H. (1957). "Negative mass in general relativity". Rev. Mod. Phys. 29:423.
- Forward, R. (2015). "Negative matter propulsion". J. Propulsion & Power 6:1.
- Peters & Mathews (1963). Phys. Rev. 131:435 (quadrupole GW power).
- Loeb, A. (2024). "Negative mass binaries generate never-seen-before
  gravitational radiation". Medium.
"""

from __future__ import annotations

import math

import scipy.constants as sc

# Smallest magnitudes we treat as physically meaningful before raising.
M_NEG_FLOOR_KG: float = 1e-30
SEP_FLOOR_M: float = 1e-30

# Convenience re-exports.
G_NEWTON: float = sc.G  # m³ kg⁻¹ s⁻²
C_LIGHT: float = sc.c  # m / s


def validate_separation(separation_m: float) -> None:
    """Raise if separation is non-positive or below the singularity floor."""
    if separation_m <= 0:
        raise ValueError("separation_m must be positive")
    if separation_m < SEP_FLOOR_M:
        raise ValueError(
            f"separation_m={separation_m} m below floor {SEP_FLOOR_M} m; field is singular"
        )


def validate_neg_mass(m_neg_kg: float) -> None:
    """Raise if the magnitude is non-positive or below the singularity floor.

    Note: the *value* is a positive float; the negativity is part of the model
    premise (``m_negative_kg`` names the magnitude of a hypothetical negative
    inertial mass), not the sign of this argument.
    """
    if m_neg_kg <= 0:
        raise ValueError("m_negative_kg must be positive (it names the magnitude)")
    if m_neg_kg < M_NEG_FLOOR_KG:
        raise ValueError(
            f"m_negative_kg={m_neg_kg} kg below floor {M_NEG_FLOOR_KG} kg"
        )


def bondi_self_acceleration(m_neg_kg: float, separation_m: float) -> float:
    """Bondi 1957 self-acceleration of a zero-net-mass pair.

    ``a = G · |m_neg| / d²``. Direction is from the negative-mass element toward
    the positive-mass element; both move together because the negative-mass
    body is "pushed" toward the puller (its inertial response to attractive
    gravity is reversed). The result is a constant body acceleration of the
    composite craft, independent of the craft's own mass.
    """
    validate_neg_mass(m_neg_kg)
    validate_separation(separation_m)
    return G_NEWTON * m_neg_kg / (separation_m * separation_m)


def chirp_mass(m1_kg: float, m2_kg: float) -> float:
    """Signed Peters–Mathews chirp mass.

    For a standard (positive-mass) binary,
    ``M_c = (m1·m2)^(3/5) / (m1+m2)^(1/5)``. For Loeb's anti-chirp case the
    binary has positive total mass but one *negative* component, so
    ``m1·m2 < 0`` while ``m1+m2 > 0``. We return a *signed* value: positive
    for a normal chirping binary (orbit contracts, frequency rises) and
    negative for an anti-chirping binary (orbit expands, frequency falls).
    """
    if m1_kg + m2_kg <= 0:
        # Caller should branch via is_unstable_total_mass first.
        raise ValueError("chirp_mass undefined for non-positive total mass")
    product = m1_kg * m2_kg
    sign = 1.0 if product > 0 else -1.0
    return sign * (abs(product) ** 0.6) / ((m1_kg + m2_kg) ** 0.2)


def gw_quadrupole_power(
    m1_kg: float, m2_kg: float, separation_m: float, omega_rad_s: float
) -> float:
    """Peters–Mathews quadrupole GW power for a Keplerian circular binary.

    ``P = (32/5) · G⁴/c⁵ · (m1·m2)² · (m1+m2) / d⁵``. The Peters–Mathews
    derivation assumes positive masses; we evaluate it on the unsigned
    magnitudes here so the function returns a positive power for any
    parameter combination. The orbital-frequency dependence is captured by
    Kepler's third law (``ω² = G·(m1+m2)/d³``) — callers supplying ``omega``
    inconsistent with ``d`` will not be caught here.
    """
    validate_separation(separation_m)
    if omega_rad_s < 0:
        raise ValueError("omega_rad_s must be non-negative")
    return (
        (32.0 / 5.0)
        * G_NEWTON**4
        / C_LIGHT**5
        * (m1_kg * m2_kg) ** 2
        * abs(m1_kg + m2_kg)
        / separation_m**5
    )


def gw_dipole_power(m_neg_kg: float, separation_m: float, omega_rad_s: float) -> float:
    """Dipole GW power emitted when the equivalence principle fails for ``m_neg``.

    If the negative-mass body responds to gravity with anomalous inertial
    coupling, the mass-dipole moment ``D = m_neg · d`` ceases to be
    conserved and dipole gravitational radiation appears at leading order
    instead of quadrupole. Power scales as

        P_dipole = (2/3) · G/c³ · m_neg² · d² · ω⁴

    (Loeb 2024). Magnitudes only; the prefactor is the standard dipole
    radiation coefficient with G/c³ in place of the electromagnetic ε₀.
    """
    validate_neg_mass(m_neg_kg)
    validate_separation(separation_m)
    if omega_rad_s < 0:
        raise ValueError("omega_rad_s must be non-negative")
    return (
        (2.0 / 3.0)
        * G_NEWTON
        / C_LIGHT**3
        * m_neg_kg**2
        * separation_m**2
        * omega_rad_s**4
    )


def is_unstable_total_mass(m1_kg: float, m2_kg: float) -> bool:
    """``True`` iff the binary has *negative* total mass (Loeb's unstable case).

    A negative-total-mass binary is gravitationally repulsive and unbound; no
    stationary orbit exists and the two bodies fly apart. Loeb's article
    flags this regime as "short-lived"; the ``NegativeTotalMassBinary``
    model uses this predicate to decide whether to raise.
    """
    return m1_kg + m2_kg < 0


def keplerian_omega(m_total_kg: float, separation_m: float) -> float:
    """Kepler's third law for a circular orbit: ``ω = sqrt(G · M / d³)``.

    Convenience helper so the symbolic/numeric anti-chirp tests can agree on
    a single frequency for a given ``(M, d)``.
    """
    if m_total_kg <= 0:
        raise ValueError("m_total_kg must be positive for a bound circular orbit")
    validate_separation(separation_m)
    return math.sqrt(G_NEWTON * m_total_kg / separation_m**3)
