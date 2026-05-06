"""usetheforce — speculative propulsion via artificial gravity fields.

Public API surface kept intentionally small. Subpackages:
- ``casimir``, ``qfield``, ``antimatter``: peer propulsion avenues, each producing a ``ForceField``.
- ``fields``: generic grid + operator kernels shared by avenues.
- ``trajectories``: ODE integration of motion through a ``ForceField``.
- ``symbolic``: SymPy derivations; lambdified callables for numerical use.
- ``viz``: matplotlib / plotly / PyVista rendering (lazy-imported).
"""

from usetheforce.protocol import ForceField
from usetheforce.units import Q_, ureg

__version__ = "0.0.1"

__all__ = ["Q_", "ForceField", "__version__", "ureg"]
