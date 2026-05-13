"""Black-hole "explorer" avenue — *exceptionally* speculative.

Two submodels:

- ``SchwarzschildGravity`` — anchored Newtonian / optional GR-hover gravitational
  field outside a Schwarzschild horizon. Anchored physics; ``speculative=False``.
- ``BlackHoleCounterDrive`` — speculative counter-thrust that cancels a supplied
  gravitational background, intended to quantify the propulsion shortfall as a
  vehicle approaches ``r_s``. There is no known mechanism for a drive that
  couples directly to local ``g``; treat the output as "what would be required
  if such a thing existed", never as engineering reality.
"""

from usetheforce.blackhole.counter_drive import BlackHoleCounterDrive
from usetheforce.blackhole.metric import SchwarzschildGravity

__all__ = ["BlackHoleCounterDrive", "SchwarzschildGravity"]
