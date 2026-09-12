"""The pursue step — motifs initiating their own follow-up exploration.

Closes the loop that run_loop_novelty left open: the pursue_queue was
declared but never consumed, and surprising connections were detected
but never explored.

Division of labor:

- The human supplies, once, a TEMPLATE per pursuit kind (in the scenario
  config) — the epistemic move the system should make.
- The field supplies, per firing, the specifics — which motifs, which
  symbols, which threads, the context block — mechanically, from
  ``graph.analyze_field`` and ``utils.context.render_context``.

Two pursuit kinds:

- ``bridge``  fires on the top surprising cross-community connection:
              "what higher abstraction binds A and B?"
- ``deepen``  fires on a high-novelty motif from the pursue_queue:
              "develop this one level of abstraction higher."

Everything before the model call is deterministic. The model call goes
through the existing ThreadManager path, so the stub backend and any
router indexing keep working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.graph import analyze_field
from symbolic_recursion.utils.context import render_context

DEFAULT_RECENCY_HALF_LIFE_HOURS = 24.0

DEFAULT_TEMPLATES: Dict[str, str] = {
    "bridge": (
        "Explore the connection between [{a_symbols}] and [{b_symbols}]. "
        "The field links them across threads ({a_thread} x {b_thread}) "
        "although their vocabularies barely overlap. "
        "What higher abstraction binds them? State it, develop it, and end "
        "with one testable prediction."
    ),
    "deepen": (
        "This motif scored novel against the whole field:\n\n{content}\n\n"
        "Develop it one level of abstraction higher, or state precisely why "
        "it resists integration with the field."
    ),
}


@dataclass
class Pursuit:
    """A planned pursuit: everything but the model response."""
    kind: str
    prompt: str
    symbols: List[str]
    targets: List[str]          # motif ids the new capture will link to
    thread_name: str


@dataclass
class PursuitResult:
    kind: str
    motif_id: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    thread_name: str = ""
    skipped: Optional[str] = None  # reason, when nothing fired


def _merge_symbols(a: List[str], b: List[str], per_side: int = 3) -> List[str]:
    """Mechanical symbol inheritance for a bridge capture: up to
    ``per_side`` symbols from each endpoint, order preserved, deduped."""
    out: List[str] = []
    for side in (a[:per_side], b[:per_side]):
        for s in side:
            if s not in out:
                out.append(s)
    return out


def _context_block(
    smc: SymbolicMemoryCore, graph, seed_ids: List[str], k: int = 5
) -> str:
    """Structure-driven context: the seeds plus their strongest graph
    neighbors, rendered through the existing render_context()."""
    picked, seen = [], set()
    for mid in seed_ids:
        ranked = [(mid, 1.0)] + sorted(
            graph.adj.get(mid, {}).items(), key=lambda kv: (-kv[1], kv[0])
        )
        for cand, w in ranked:
            if cand not in seen:
                seen.add(cand)
                picked.append((smc.get_motif(cand), min(w, 1.0)))
    return render_context(picked[:k])


def _age_hours(smc: SymbolicMemoryCore, motif_id: str, now: datetime) -> float:
    """Age of a motif in hours; missing/unparseable created_at counts as old."""
    m = smc.get_motif(motif_id)
    try:
        created = datetime.fromisoformat(m.created_at)
    except (TypeError, ValueError, AttributeError):
        return float("inf")
    return max(0.0, (now - created).total_seconds() / 3600.0)


def _recency_factor(
    smc: SymbolicMemoryCore, a: str, b: str, now: datetime, half_life_hours: float
) -> float:
    """Damping in [0, 1) driven by the YOUNGER endpoint: 0 at age zero,
    0.5 at the half-life, saturating toward 1. A pursuit capture's fresh
    edges are cross-community and cross-thread by construction; without
    this the loop chain-pursues its own tail instead of the frontier."""
    age = min(_age_hours(smc, a, now), _age_hours(smc, b, now))
    if age == float("inf"):
        return 1.0
    return age / (age + half_life_hours)


def _is_resolved(smc: SymbolicMemoryCore, a: str, b: str) -> bool:
    """A surprise is resolved once some motif references both endpoints —
    a bridge capture exists, so the pair is no longer an open question.

    An edge a pursuit capture made to its own target is CONSTRUCTED, not
    surprising — the capture is the answer, so the pair is resolved by
    construction (otherwise the loop pursues its own products' edges)."""
    for mid, other in ((a, b), (b, a)):
        m = smc.get_motif(mid)
        if m and other in m.references and m.thread_id.startswith("pursue"):
            return True
    for m in smc.list_motifs():
        if a in m.references and b in m.references:
            return True
    return False


def plan_bridge(
    smc: SymbolicMemoryCore,
    analysis: Optional[dict] = None,
    template: str = DEFAULT_TEMPLATES["bridge"],
    half_life_hours: float = DEFAULT_RECENCY_HALF_LIFE_HOURS,
    now: Optional[datetime] = None,
) -> Optional[Pursuit]:
    """Plan a pursuit of the best OPEN surprising connection.

    Selection = raw surprise score x recency factor, skipping resolved
    pairs. None if the field has no open surprise to offer."""
    if analysis is None:
        analysis = analyze_field(smc)
    if now is None:
        now = datetime.utcnow()
    best, best_key = None, None
    for s in analysis["surprises"]:
        if _is_resolved(smc, s["a"], s["b"]):
            continue
        damped = s["score"] * _recency_factor(smc, s["a"], s["b"], now, half_life_hours)
        key = (-damped, s["a"], s["b"])
        if best_key is None or key < best_key:
            best, best_key = s, key
    if best is None:
        return None
    top = best
    a, b = smc.get_motif(top["a"]), smc.get_motif(top["b"])
    task = template.format(
        a_symbols=", ".join(a.symbols),
        b_symbols=", ".join(b.symbols),
        a_thread=a.thread_id,
        b_thread=b.thread_id,
        reasons="; ".join(top["reasons"]),
    )
    ctx = _context_block(smc, analysis["graph"], [a.id, b.id])
    return Pursuit(
        kind="bridge",
        prompt=f"{ctx}\n\n## Task\n{task}",
        symbols=_merge_symbols(a.symbols, b.symbols),
        targets=[a.id, b.id],
        thread_name=f"pursue-bridge-{a.id[:8]}-{b.id[:8]}",
    )


def plan_deepen(
    smc: SymbolicMemoryCore,
    motif_id: str,
    analysis: Optional[dict] = None,
    template: str = DEFAULT_TEMPLATES["deepen"],
) -> Optional[Pursuit]:
    """Plan a deepening pursuit of one (typically high-novelty) motif."""
    m = smc.get_motif(motif_id)
    if m is None:
        return None
    if analysis is None:
        analysis = analyze_field(smc)
    task = template.format(
        content=m.content.strip(),
        symbols=", ".join(m.symbols),
        thread=m.thread_id,
    )
    ctx = _context_block(smc, analysis["graph"], [m.id])
    return Pursuit(
        kind="deepen",
        prompt=f"{ctx}\n\n## Task\n{task}",
        symbols=list(m.symbols),
        targets=[m.id],
        thread_name=f"pursue-deepen-{m.id[:8]}",
    )


def execute(smc: SymbolicMemoryCore, tm, pursuit: Pursuit, model: str) -> PursuitResult:
    """Fire a planned pursuit through the existing ask -> capture -> link
    path. ``tm`` is a threads.manager.ThreadManager."""
    thread = tm.new_thread(pursuit.thread_name, model=model)
    resp = thread.ask(pursuit.prompt)
    m = tm.capture_as_motif(thread, pursuit.symbols, resp)
    for target in pursuit.targets:
        smc.link_motifs(m.id, target)
    from symbolic_recursion.core.flow import record_flow
    record_flow({"kind": pursuit.kind, "thread": pursuit.thread_name,
                 "model": model, "motif_id": m.id, "targets": pursuit.targets,
                 "prompt": pursuit.prompt, "response": resp})
    return PursuitResult(
        kind=pursuit.kind,
        motif_id=m.id,
        targets=list(pursuit.targets),
        thread_name=pursuit.thread_name,
    )


def run_pursuits(
    smc: SymbolicMemoryCore,
    tm,
    pursue_queue: List[str],
    cfg: Optional[dict] = None,
    model: str = "llama3:instruct",
    analysis: Optional[dict] = None,
) -> List[PursuitResult]:
    """Consume pursuit intentions for one cycle.

    ``cfg`` (the scenario's ``pursue`` section):
      enabled        bool, default False — the step is opt-in
      max_per_cycle  int, default 1
      templates      {kind: template} overrides, merged over defaults
      recency_half_life_hours  float, default 24 — fresh edges must season
                     before they can be pursued (see _recency_factor)

    Fires at most one bridge (top surprise), then deepens motifs from the
    queue until the per-cycle budget is spent. The queue is consumed
    front-first; unfired entries are left in place for the next cycle.
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return []
    budget = int(cfg.get("max_per_cycle", 1))
    templates = {**DEFAULT_TEMPLATES, **cfg.get("templates", {})}
    if analysis is None:
        analysis = analyze_field(smc)

    half_life = float(cfg.get("recency_half_life_hours", DEFAULT_RECENCY_HALF_LIFE_HOURS))

    results: List[PursuitResult] = []
    bridge = plan_bridge(smc, analysis, templates["bridge"], half_life_hours=half_life)
    if bridge is not None and budget > 0:
        results.append(execute(smc, tm, bridge, model))
        budget -= 1
    else:
        results.append(PursuitResult(kind="bridge", skipped="no surprising connection"))

    while budget > 0 and pursue_queue:
        mid = pursue_queue.pop(0)
        plan = plan_deepen(smc, mid, analysis, templates["deepen"])
        if plan is None:
            results.append(PursuitResult(kind="deepen", skipped=f"unknown motif {mid}"))
            continue
        results.append(execute(smc, tm, plan, model))
        budget -= 1
    return results
