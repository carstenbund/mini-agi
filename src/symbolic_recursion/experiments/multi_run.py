# src/symbolic_recursion/experiments/multi_run.py
from __future__ import annotations
import argparse, os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore, MotifNode
from symbolic_recursion.core.storage import load_motifs, save_motifs
from symbolic_recursion.threads.manager import ThreadManager
from symbolic_recursion.utils.novelty import novelty_index
from symbolic_recursion.core.runtime_policy import Policy
from symbolic_recursion.utils.context import render_context

# Routers
from symbolic_recursion.core.vector_router import VectorRouter
try:
    from symbolic_recursion.core.chroma_router import ChromaRouter  # persistent
except Exception:
    ChromaRouter = None  # type: ignore


# -------- Router factory --------

def _make_router(kind: str):
    if kind.lower() == "chroma":
        from symbolic_recursion.core.chroma_router import ChromaRouter
        from symbolic_recursion.embeddings.sbert import Embeddings
        return ChromaRouter(embeddings=Embeddings())
    from symbolic_recursion.core.vector_router import VectorRouter
    return VectorRouter(None)


def _make_routerx(kind: str):
    k = (kind or os.getenv("SMC_ROUTER", "vector")).lower()
    if k == "chroma":
        if ChromaRouter is None:
            raise RuntimeError("ChromaRouter not available. `pip install chromadb` and ensure import path is correct.")
        # Let ChromaRouter pick up your default Embeddings or legacy fallback internally
        return ChromaRouter()
    # In-memory fallback (SBERT+FAISS or numpy; legacy sparse if SBERT missing)
    return VectorRouter(None)


_router = None  # late-bound


def _ensure_router_built(smc: SymbolicMemoryCore, kind: str):
    global _router
    if _router is None:
        _router = _make_router(kind)
    _router.rebuild_from_smc(smc)


# -------- Helpers --------

def _top1_sim(text: str) -> Tuple[float, Optional[MotifNode]]:
    """Return (similarity, nearest_motif) for a text against current index."""
    res = _router.search_text(text, top_k=1)
    if not res:
        return 0.0, None
    return float(res[0][1]), res[0][0]


def _link_with_policy(smc: SymbolicMemoryCore, m: MotifNode, policy: Policy) -> int:
    """Propose links via router; accept per policy with symbol-aware Jaccard & fanout cap."""
    suggestions = _router.suggest_for_motif(smc, m.id, top_k=8)
    linked = 0
    for cand, score in suggestions:
        # symbol-aware Jaccard
        a, b = set(m.symbols), set(cand.symbols)
        jacc = len(a & b) / max(1, len(a | b))
        if policy.should_link(score, jacc, linked):
            smc.link_motifs(m.id, cand.id)
            linked += 1
    return linked


def _context_for_prompt(prompt: str, k: int = 5) -> str:
    """Retrieve top-k motifs and render a compact context block."""
    retrieved = _router.search_text(prompt, top_k=k)
    return render_context(retrieved)


def _ask_and_capture(
    tm: ThreadManager,
    smc: SymbolicMemoryCore,
    topic: Dict[str, Any],
    policy: Policy,
    model: str
) -> Dict[str, Any]:
    """
    One worker:
      - build context from the existing graph,
      - ask the model,
      - capture motif,
      - dedup -> novelty gate -> link (using index, but we batch-index after all threads).
    """
    thread = tm.new_thread(topic.get("name", "session"), model=model)

    # Teach the model what we already know
    ctx = _context_for_prompt(topic["prompt"])
    final_prompt = f"{ctx}\n\n## Task\n{topic['prompt']}"
    resp = thread.ask(final_prompt)

    m = tm.capture_as_motif(thread, topic.get("symbols", []), resp)

    # --- Dedup (nearest-neighbor via router; motif not yet indexed) ---
    t1_sim, nearest = _top1_sim(m.content)
    if policy.is_duplicate(t1_sim) and nearest is not None:
        # Merge: extend symbols on nearest, drop this motif
        nearest.symbols = sorted(set(nearest.symbols) | set(m.symbols))
        smc.motifs.pop(m.id, None)
        return {
            "id": nearest.id,
            "deduped_into": nearest.id,
            "linked": 0,
            "novelty": 0.0,
            "top1_sim": float(t1_sim),
            "skip": True,
            "reason": "dedup"
        }

    # --- Novelty gating ---
    n = novelty_index(smc, m)  # expects {"novelty_index": float, ...}
    if not policy.should_persist(n.get("novelty_index", 0.0)):
        smc.motifs.pop(m.id, None)
        return {
            "id": m.id,
            "skip": True,
            "reason": "low_novelty",
            "top1_sim": float(t1_sim),
            **n
        }

    # --- Link (using router on content; we can link before adding to index) ---
    linked = _link_with_policy(smc, m, policy)

    # We DO NOT add motif to index here to avoid per-thread contention.
    # The main thread will batch add all kept motifs once futures complete.
    return {
        "id": m.id,
        "skip": False,
        "linked": linked,
        "top1_sim": float(t1_sim),
        **n
    }


# -------- Entry point --------

def run():
    ap = argparse.ArgumentParser(description="Multi-thread topic runner with persistent routing & policies.")
    ap.add_argument("scenario", help="Path to JSON scenario with prompts.")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", type=str, default=os.getenv("SMC_MODEL", "llama3:instruct"))
    ap.add_argument("--router", type=str, default=os.getenv("SMC_ROUTER", "chroma"),
                    help="vector|chroma (default from $SMC_ROUTER or 'chroma')")
    ap.add_argument("--link_threshold", type=float, default=0.60)
    ap.add_argument("--dedup_threshold", type=float, default=0.93)
    ap.add_argument("--novelty_min", type=float, default=0.10)
    ap.add_argument("--max_links", type=int, default=3)
    args = ap.parse_args()

    # Load memory + thread manager
    smc = SymbolicMemoryCore(); smc.motifs = load_motifs()
    tm = ThreadManager(smc)

    policy = Policy(
        link_threshold=args.link_threshold,
        dedup_threshold=args.dedup_threshold,
        novelty_min=args.novelty_min,
        max_links_per_motif=args.max_links,
    )

    # Scenario: { "topics": [ {name, prompt, symbols[]}, ... ] }
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, os.path.basename(args.scenario))  # always load from same dir
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    topics = cfg["topics"]

    # Build router from current motifs
    _ensure_router_built(smc, args.router)

    # Concurrent execution
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_ask_and_capture, tm, smc, t, policy, args.model) for t in topics]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Persist motifs to disk
    save_motifs(smc.motifs)

    # Batch add newly accepted motifs to the index (if router supports add_many)
    kept_ids = [r["id"] for r in results if not r.get("skip")]
    kept_motifs = [smc.get_motif(mid) for mid in kept_ids if smc.get_motif(mid) is not None]
    if hasattr(_router, "add_many"):
        _router.add_many(smc, kept_motifs)  # ChromaRouter fast path
    else:
        for m in kept_motifs:
            _router.add_motif(smc, m)       # VectorRouter / generic path

    # Final log
    log = {
        "policy": vars(policy),
        "counts": {"V": len(smc.motifs)},
        "results": results
    }
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    run()

