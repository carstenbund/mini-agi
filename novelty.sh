#!/bin/bash
set -e

ROOT="symbolic-memory-core"
[ -d "$ROOT" ] || { echo "❌ $ROOT not found. Initialize the repo first."; exit 1; }

mkdir -p "$ROOT/utils" "$ROOT/experiments/logs" "$ROOT/docs"
touch "$ROOT/utils/__init__.py" "$ROOT/experiments/__init__.py"

# ---------------------------------------
# utils/novelty.py
# ---------------------------------------
cat > "$ROOT/utils/novelty.py" << 'EOF'
from typing import Dict, List, Set, Tuple
from embeddings.embedder import embed_text, cosine_sparse
from core.motif import SymbolicMemoryCore, MotifNode

def _all_vectors(smc: SymbolicMemoryCore) -> List[Dict[str,float]]:
    return [embed_text(m.content) for m in smc.list_motifs()]

def _max_cosine_to_corpus(vec: Dict[str,float], corpus: List[Dict[str,float]]) -> float:
    if not corpus: return 0.0
    best = 0.0
    for v in corpus:
        c = cosine_sparse(vec, v)
        if c > best:
            best = c
    return best

def semantic_novelty(smc: SymbolicMemoryCore, m: MotifNode) -> float:
    """1 - max cosine(new, any prior). Higher = more novel."""
    new_vec = embed_text(m.content)
    prior_vecs = _all_vectors(smc)
    # Exclude self if already inserted
    if prior_vecs and embed_text(m.content) in prior_vecs and len(prior_vecs) > 1:
        pass
    max_sim = _max_cosine_to_corpus(new_vec, prior_vecs)
    return max(0.0, 1.0 - max_sim)

def symbolic_novelty(smc: SymbolicMemoryCore, m: MotifNode) -> float:
    """Fraction of m.symbols unseen in graph."""
    seen: Set[str] = set()
    for x in smc.list_motifs():
        for s in x.symbols:
            seen.add(s.lower())
    if not m.symbols:
        return 0.0
    new_syms = [s for s in m.symbols if s.lower() not in seen]
    return len(new_syms) / float(len(m.symbols))

def structural_novelty(smc: SymbolicMemoryCore, m: MotifNode, sim_threshold: float = 0.35) -> float:
    """
    Cheap proxy for bridge potential: count how many existing motifs are
    semantically similar above threshold but not yet linked.
    Normalize by total motifs to keep [0,1].
    """
    new_v = embed_text(m.content)
    V = len(smc.motifs)
    if V == 0:
        return 0.0
    hits = 0
    for other in smc.list_motifs():
        if other.id == m.id: 
            continue
        sim = cosine_sparse(new_v, embed_text(other.content))
        if sim >= sim_threshold:
            hits += 1
    return hits / float(V)

def novelty_index(smc: SymbolicMemoryCore, m: MotifNode,
                  alpha: float = 0.5, beta: float = 0.2, gamma: float = 0.3,
                  sim_threshold: float = 0.35) -> Dict[str, float]:
    sem = semantic_novelty(smc, m)
    sym = symbolic_novelty(smc, m)
    stc = structural_novelty(smc, m, sim_threshold=sim_threshold)
    score = alpha*sem + beta*sym + gamma*stc
    return {"semantic": sem, "symbolic": sym, "structural": stc, "novelty_index": score}
EOF

# ---------------------------------------
# experiments/run_loop_novelty.py
# ---------------------------------------
cat > "$ROOT/experiments/run_loop_novelty.py" << 'EOF'
import json, sys
from pathlib import Path
from typing import List, Dict, Any

from core.motif import SymbolicMemoryCore
from core.storage import load_motifs, save_motifs
from core.router import suggest_links
from threads.manager import ThreadManager
from utils.metrics import graph_stats, semantic_stats
from utils.novelty import novelty_index

STATE_PATH = Path("experiments/state.json")
LOGS_DIR   = Path("experiments/logs")

def _load_state():
    if STATE_PATH.exists():
        return json.load(open(STATE_PATH, "r"))
    return {"prev_centroid": {}}

def _save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(state, open(STATE_PATH, "w"), indent=2)

def run(cfg_path: str):
    cfg = json.load(open(cfg_path, "r"))
    model = cfg.get("model", "llama2")
    use_stub = bool(cfg.get("use_stub", False))
    link_threshold = float(cfg.get("capture_threshold", 0.35))
    novelty_threshold = float(cfg.get("novelty_threshold", 0.55))   # pursue threshold
    cycles = int(cfg.get("cycles", 3))
    prompts = cfg.get("prompts", [])

    smc = SymbolicMemoryCore(); smc.motifs = load_motifs()
    tm = ThreadManager(smc)

    if use_stub:
        import core.ollama_interface as oi
        from core.model_stub import stub_response
        oi.query_ollama = lambda prompt, model=model, timeout=None: stub_response(prompt, model, timeout)

    state = _load_state()

    for t in range(cycles):
        cycle_log: Dict[str, Any] = {"cycle": t, "new_motifs": []}
        pursue_queue: List[str] = []

        # 1) Generate + capture
        new_ids: List[str] = []
        for p in prompts:
            thr = tm.new_thread(p.get("name","session"), model=model)
            resp = thr.ask(p["prompt"])
            m = tm.capture_as_motif(thr, p.get("symbols", []), resp)
            new_ids.append(m.id)

        # 2) Auto-link and score novelty
        for mid in new_ids:
            m = smc.get_motif(mid)
            # auto-link by similarity
            linked_any = False
            for cand, score in suggest_links(smc, mid, top_k=3):
                if score >= link_threshold:
                    smc.link_motifs(mid, cand.id)
                    linked_any = True
            # novelty scoring
            n = novelty_index(smc, m)
            entry = {"id": mid, **n, "linked": linked_any}
            cycle_log["new_motifs"].append(entry)

            # pursuit decision
            if n["novelty_index"] >= novelty_threshold:
                pursue_queue.append(mid)

        # 3) Metrics + state
        g = graph_stats(smc)
        s = semantic_stats(smc, prev_centroid=state.get("prev_centroid", {}))
        cycle_log.update({
            "V": g["V"], "E": g["E"],
            "edge_density": g["edge_density"],
            "recurrence_rate": g["recurrence_rate"],
            "lcc_fraction": g["lcc_fraction"],
            "clustering_coeff": g["clustering_coeff"],
            "avg_shortest_path": g["avg_shortest_path"],
            "centroid_shift": s["centroid_shift"],
            "cohesion": s["cohesion"],
            "pursue_queue": pursue_queue
        })

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(cycle_log, open(LOGS_DIR / f"novelty_cycle_{t}.json", "w"), indent=2)

        # 4) Persist
        state["prev_centroid"] = s["centroid"]
        _save_state(state)
        save_motifs(smc.motifs)

        print(f"[cycle {t}] V={int(g['V'])} nov_candidates={len(pursue_queue)} "
              f"rec={g['recurrence_rate']:.2f} lcc={g['lcc_fraction']:.2f} "
              f"shift={s['centroid_shift']:.2f} coh={s['cohesion']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/run_loop_novelty.py experiments/scenario.json")
        sys.exit(1)
    run(sys.argv[1])
EOF

# ---------------------------------------
# experiments/report_novelty.py
# ---------------------------------------
cat > "$ROOT/experiments/report_novelty.py" << 'EOF'
import json, glob

def load_cycles():
    paths = sorted(glob.glob("experiments/logs/novelty_cycle_*.json"))
    return [json.load(open(p)) for p in paths]

def main():
    rows = load_cycles()
    if not rows:
        print("No novelty logs found.")
        return

    print("cycle | V | pursue | mean_novelty | mean_sem | mean_sym | mean_stc | lcc | rec | coh | shift")
    print("------|---|--------|--------------|----------|----------|----------|-----|-----|-----|------")
    for r in rows:
        new = r.get("new_motifs", [])
        nvals = [x["novelty_index"] for x in new] or [0.0]
        sems  = [x["semantic"] for x in new] or [0.0]
        syms  = [x["symbolic"] for x in new] or [0.0]
        stcs  = [x["structural"] for x in new] or [0.0]
        print(f"{r['cycle']:>5} |"
              f"{int(r.get('V',0)):>2} |"
              f"{len(r.get('pursue_queue',[])):>6} |"
              f"{sum(nvals)/len(nvals):>12.3f} |"
              f"{sum(sems)/len(sems):>8.3f} |"
              f"{sum(syms)/len(syms):>8.3f} |"
              f"{sum(stcs)/len(stcs):>8.3f} |"
              f"{r.get('lcc_fraction',0):>3.2f} |"
              f"{r.get('recurrence_rate',0):>3.2f} |"
              f"{r.get('cohesion',0):>3.2f} |"
              f"{r.get('centroid_shift',0):>4.2f}")

    # Top candidates per last cycle
    last = rows[-1]
    cand = last.get("new_motifs", [])
    cand.sort(key=lambda x: x["novelty_index"], reverse=True)
    top = cand[:5]
    print("\nTop novelty candidates (last cycle):")
    for x in top:
        print(f"- {x['id']} | novelty={x['novelty_index']:.3f} (sem={x['semantic']:.3f}, sym={x['symbolic']:.3f}, stc={x['structural']:.3f}) linked={x['linked']}")
EOF

# ---------------------------------------
# docs/Novelty.md
# ---------------------------------------
cat > "$ROOT/docs/Novelty.md" << 'EOF'
# Novelty & Pursuit in SMC

## Purpose
Identify **valuable novelty**: motifs that push into new semantic territory **and** can be integrated into the evolving symbolic reality.

## Components
- **Semantic**: 1 − max cosine(new, prior)
- **Symbolic**: fraction of motif symbols unseen in the graph
- **Structural**: proxy for bridge potential (neighbors above sim τ)

Combined:
