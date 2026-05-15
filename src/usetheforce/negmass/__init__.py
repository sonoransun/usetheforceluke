"""Negative-mass binaries — SPECULATIVE.

Four sub-models based on Bondi's 1957 negative-mass solutions:

- ``BondiRunawayPair`` — zero-net-mass pair, constant body acceleration
  ``a = G·|m_neg|/d²`` on the composite craft (Forward 2015 propulsion claim).
- ``AntiChirpBinary`` — positive-total-mass binary whose orbit *expands* due
  to the negative component, emitting anti-chirp GW (frequency decreases).
- ``NegativeTotalMassBinary`` — repulsive, unbound; short-lived (Loeb 2024).
- ``DipoleGravitonRadiator`` — Loeb's "what if WEP fails" case: dipole GW
  pattern instead of quadrupole.

All four are speculative; only ``BondiRunawayPair`` has a power adapter in
``missions.adapters``. The other three are research artifacts exposed as
``ForceField``s for analysts to combine via ``CompositeField`` if desired.

References
----------
- Bondi 1957, Rev. Mod. Phys. 29:423.
- Forward 2015, J. Propulsion & Power 6:1 ("Negative matter propulsion").
- Peters & Mathews 1963, Phys. Rev. 131:435.
- Loeb 2024, Medium ("Negative mass binaries…").
"""

from usetheforce.negmass.anti_chirp import AntiChirpBinary
from usetheforce.negmass.bondi_pair import BondiRunawayPair
from usetheforce.negmass.buffer import NegativeMassPointSource
from usetheforce.negmass.dipole_gw import DipoleGravitonRadiator
from usetheforce.negmass.unstable import NegativeTotalMassBinary

__all__ = [
    "AntiChirpBinary",
    "BondiRunawayPair",
    "DipoleGravitonRadiator",
    "NegativeMassPointSource",
    "NegativeTotalMassBinary",
]
