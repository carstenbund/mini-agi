"""Lightweight metric helpers used by the novelty loop.

The real project computes a variety of graph and embedding statistics.
For the purposes of unit tests and offline evaluation we provide very
simple stand-ins that satisfy the interface required by
``run_loop_novelty``.
"""

from typing import Dict, Any

from symbolic_recursion.core.motif import SymbolicMemoryCore


def graph_stats(smc: SymbolicMemoryCore) -> Dict[str, float]:
    """Return trivial graph statistics for ``smc``.

    The values are intentionally simplistic; they merely allow the
    experiment script to run without depending on heavy graph libraries.
    """
    V = len(smc.motifs)
    E = sum(len(m.references) for m in smc.motifs.values())
    edge_density = (E / (V * (V - 1))) if V > 1 else 0.0
    return {
        "V": float(V),
        "E": float(E),
        "edge_density": edge_density,
        "recurrence_rate": 0.0,
        "lcc_fraction": 1.0 if V else 0.0,
        "clustering_coeff": 0.0,
        "avg_shortest_path": 0.0,
    }


def semantic_stats(smc: SymbolicMemoryCore, prev_centroid: Dict[str, float]) -> Dict[str, Any]:
    """Return placeholder semantic statistics.

    ``prev_centroid`` is ignored except for being echoed back so that the
    calling code can persist state across runs.
    """
    return {"centroid_shift": 0.0, "cohesion": 0.0, "centroid": {}}
