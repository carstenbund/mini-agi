# src/symbolic_recursion/experiments/analyze_existing.py

import json
from pathlib import Path

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.core.storage import load_motifs
from symbolic_recursion.utils.metrics import graph_stats, semantic_stats
from symbolic_recursion.utils.novelty import novelty_index

STATE_PATH = Path("experiments/state.json")

def run():
    # Load motifs from disk
    smc = SymbolicMemoryCore()
    smc.motifs = load_motifs()

    if not smc.motifs:
        print("No motifs found.")
        return

    # Load previous state if exists (for centroid continuity)
    if STATE_PATH.exists():
        state = json.load(open(STATE_PATH, "r"))
        prev_centroid = state.get("prev_centroid", {})
    else:
        prev_centroid = {}

    # Compute graph + semantic metrics
    g = graph_stats(smc)
    s = semantic_stats(smc, prev_centroid=prev_centroid)

    print("=== Graph Metrics ===")
    for k, v in g.items():
        print(f"{k:20s}: {v}")

    print("\n=== Semantic Metrics ===")
    for k, v in s.items():
        print(f"{k:20s}: {v}")

    # Show novelty of each motif
    print("\n=== Motif Novelties ===")
    for mid, m in smc.motifs.items():
        n = novelty_index(smc, m)
        print(f"{mid[:8]} {m.symbols} novelty={n['novelty_index']:.3f}")

if __name__ == "__main__":
    run()

