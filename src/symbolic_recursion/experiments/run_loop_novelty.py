import json, os, sys
from pathlib import Path
from typing import List, Dict, Any

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.core.storage import load_motifs, save_motifs
from symbolic_recursion.core.router import suggest_links
from symbolic_recursion.threads.manager import ThreadManager
from symbolic_recursion.utils.metrics import graph_stats, semantic_stats
from symbolic_recursion.utils.novelty import novelty_index

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
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, os.path.basename(cfg_path))  # always load from same dir
    cfg = json.load(open(cfg_path, "r"))
    model = cfg.get("model", "llama2")
    use_stub = bool(cfg.get("use_stub", False))
    link_threshold = float(cfg.get("capture_threshold", 0.35))
    novelty_threshold = float(cfg.get("novelty_threshold", 0.55))   # pursue threshold
    cycles = int(cfg.get("cycles", 3))
    prompts = cfg.get("prompts", [])

    smc = SymbolicMemoryCore(); smc.motifs = load_motifs()
    tm = ThreadManager(smc)

    router = None
    rt = os.getenv("SMC_ROUTER", "").lower()
    if rt:
        from symbolic_recursion.core.router_factory import make_router
        router = make_router(rt)
        router.rebuild_from_smc(smc)

    if use_stub:
        import symbolic_recursion.core.ollama_interface as oi
        from symbolic_recursion.core.model_stub import stub_response
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
            if router:
                router.add_motif(smc, m)

        # 2) Auto-link and score novelty
        for mid in new_ids:
            m = smc.get_motif(mid)
            # auto-link by similarity
            linked_any = False
            if router:
                suggestions = router.suggest_for_motif(smc, mid, top_k=3)
            else:
                suggestions = suggest_links(smc, mid, top_k=3)
            for cand, score in suggestions:
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

        # 3) Pursue (opt-in): consume intentions — bridge the top surprise,
        # deepen queued high-novelty motifs — before metrics, so metrics
        # reflect the post-pursuit field.
        pursue_cfg = cfg.get("pursue", {})
        if pursue_cfg.get("enabled", False):
            from symbolic_recursion.core.pursue import run_pursuits
            pursued = run_pursuits(smc, tm, pursue_queue, cfg=pursue_cfg, model=model)
            for r in pursued:
                if r.motif_id and router:
                    router.add_motif(smc, smc.get_motif(r.motif_id))
            cycle_log["pursuits"] = [
                {"kind": r.kind, "motif_id": r.motif_id,
                 "targets": r.targets, "skipped": r.skipped}
                for r in pursued
            ]

        # 4) Metrics + state
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

        # 5) Persist + journal
        state["prev_centroid"] = s["centroid"]
        _save_state(state)
        save_motifs(smc.motifs)
        from symbolic_recursion.graph.trajectory import record_event
        record_event(smc, {"type": "cycle", "cycle": t,
                           "captures": len(new_ids),
                           "pursuits": len(cycle_log.get("pursuits", []))})

        print(f"[cycle {t}] V={int(g['V'])} nov_candidates={len(pursue_queue)} "
              f"rec={g['recurrence_rate']:.2f} lcc={g['lcc_fraction']:.2f} "
              f"shift={s['centroid_shift']:.2f} coh={s['cohesion']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/run_loop_novelty.py experiments/scenario.json")
        sys.exit(1)
    run(sys.argv[1])
