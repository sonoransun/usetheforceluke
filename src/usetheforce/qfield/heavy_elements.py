"""Counter-gravity force from a configurable lattice of heavy nuclei. SPECULATIVE.

Plummer-softened multi-site dipole ansatz: each lattice site contributes a
Newtonian-like 1/r² attraction softened by ε so finite-distance numerics stay
well-defined. The aggregate force is a vectorized sum over sites:

    F(r) = -κ · Σᵢ μᵢ · (r − rᵢ) / (|r − rᵢ|² + ε²)^(3/2)
    U(r) = -κ · Σᵢ μᵢ / sqrt(|r − rᵢ|² + ε²)

This is *not* derived from any nuclear physics — it's a parametric stand-in
that lets the rest of the framework (trajectories, conservation, viz) be
exercised end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class HeavyElementLattice:
    """Force from a finite array of softened-Coulomb sources.

    Per-call dataflow::

        .. mermaid::

            flowchart LR
                S[Sites r_i + strengths μ_i] --> D[d_i = r − r_i]
                D --> N["n_i = √(|d_i|² + ε²)  (Plummer softening)"]
                N --> W["w_i = -κ μ_i / n_i³"]
                W --> F["F = Σ_i w_i · d_i  (vectorised einsum)"]
    """

    metadata: dict[str, Any]

    def __init__(
        self,
        sites: np.ndarray | Sequence[Sequence[float]],
        strengths: np.ndarray | Sequence[float],
        coupling: float = 1.0,
        softening: float = 0.0,
    ) -> None:
        self._sites = np.asarray(sites, dtype=float)
        if self._sites.ndim != 2 or self._sites.shape[1] != 3:
            raise ValueError(f"sites must have shape (N, 3), got {self._sites.shape}")
        self._mu = np.asarray(strengths, dtype=float)
        if self._mu.shape != (self._sites.shape[0],):
            raise ValueError(
                f"strengths shape {self._mu.shape} must match number of sites {self._sites.shape[0]}"
            )
        self._kappa = float(coupling)
        self._eps = float(softening)
        if self._eps < 0:
            raise ValueError("softening must be non-negative")
        self.metadata = {
            "avenue": "qfield",
            "model": f"heavy-element lattice (N={len(self._mu)} sites)",
            "speculative": True,
            "speculative_components": ["coupling", "strengths", "site geometry"],
            "citation": "Plummer-softened multi-site dipole ansatz; not derived from nuclear physics",
        }

    def _displacements_and_norms(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = r[None, :] - self._sites  # (N, 3)
        sq = np.einsum("ij,ij->i", d, d)
        if self._eps == 0.0 and np.any(sq == 0.0):
            raise ValueError(
                "probe coincides with a lattice site and softening=0; "
                "supply softening > 0 to avoid the singularity"
            )
        n = np.sqrt(sq + self._eps**2)  # (N,)
        return d, n

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        d, n = self._displacements_and_norms(np.asarray(r, dtype=float))
        weights = (-self._kappa * self._mu) / n**3  # (N,)
        return np.einsum("i,ij->j", weights, d)

    def potential(self, r: np.ndarray) -> float:
        _, n = self._displacements_and_norms(np.asarray(r, dtype=float))
        return float(-self._kappa * np.sum(self._mu / n))
