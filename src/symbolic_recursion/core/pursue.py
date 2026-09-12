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

from symbolic_recursion.core.agent import agent_id
from symbolic_recursion.core.claims import (
    DEFAULT_TTL_HOURS, active_claims, claim, claimed_by_other, release, target_key,
)
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
    "self": (
        "You are reasoning about the software that maintains this field — "
        "the pipeline whose specification the target motif [{spec_symbols}] "
        "states (its text is in the context above). The pipeline's recent "
        "observed behavior is in the trajectory section above; the current "
        "regime reading is: {regime}. "
        "What revision to the pipeline does the specification imply, given "
        "the observed behavior? Name the revision, justify it from the spec "
        "and the behavior both, and end with a concrete implementable change "
        "(which component, what new behavior) plus one observable that would "
        "show, after implementation, that it worked."
    ),
}

DEFAULT_SPEC_THREADS = ("inherited-judgment",)


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
    review: Optional[str] = None   # "accept" | "revise" | "reject" when reviewed
    review_evidence: str = ""
    owner: str = ""                # agent id that fired it


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


def _failure_block(targets: List[str]) -> str:
    """Retry-with-verdict: the latest rejected/revise attempt at the same
    targets, returned as a prompt section so failure is inherited by the
    next attempt (graded-inheritance Condition 4/6 — failures and
    consequences as feedback). Empty string when there is none."""
    from symbolic_recursion.core.flow import load_flow
    prior = None
    tset = set(targets)
    for e in load_flow():
        verdict = (e.get("review") or {}).get("verdict")
        if verdict in ("reject", "revise") and set(e.get("targets") or []) == tset:
            prior = e
    if prior is None:
        return ""
    resp = (prior.get("response") or "").strip()
    if len(resp) > 900:
        resp = resp[:900] + " […]"
    evidence = (prior.get("review") or {}).get("evidence", "")
    return (
        f"\n\n## Previous attempt (review: {prior['review']['verdict']})\n"
        f"{resp}\n\n"
        f"Reviewer evidence: {evidence}\n"
        "Address the objection directly; do not resubmit the same synthesis."
    )


def _goal_symbols(smc: SymbolicMemoryCore, goal_threads: tuple) -> frozenset:
    syms = set()
    for m in smc.list_motifs():
        if m.thread_id in goal_threads:
            syms.update(s.strip().lower() for s in m.symbols if s.strip())
    return frozenset(syms)


def _goal_boost(smc: SymbolicMemoryCore, a: str, b: str,
                goal_syms: frozenset, weight: float) -> float:
    """Multiplier >= 1 pulling selection toward the goal program.

    Affinity of an endpoint = fraction of ITS symbols that are goal
    vocabulary (how much of its identity is goal-relevant); the boost
    uses the stronger endpoint. weight 0 or empty goals = neutral."""
    if not goal_syms or weight <= 0.0:
        return 1.0
    def _aff(mid: str) -> float:
        ms = {s.strip().lower() for s in smc.get_motif(mid).symbols if s.strip()}
        return len(ms & goal_syms) / len(ms) if ms else 0.0
    return 1.0 + weight * max(_aff(a), _aff(b))


def plan_bridge(
    smc: SymbolicMemoryCore,
    analysis: Optional[dict] = None,
    template: str = DEFAULT_TEMPLATES["bridge"],
    half_life_hours: float = DEFAULT_RECENCY_HALF_LIFE_HOURS,
    now: Optional[datetime] = None,
    owner: Optional[str] = None,
    respect_claims: bool = True,
    goal_threads: Optional[tuple] = None,
    goal_weight: float = 1.0,
) -> Optional[Pursuit]:
    """Plan a pursuit of the best OPEN, UNCLAIMED surprising connection.

    Selection = raw surprise score x recency factor x goal boost,
    skipping resolved pairs and pairs another owner currently holds a
    claim on (see ``core.claims``). ``goal_threads`` names threads whose
    motifs act as the goal program: seams sharing their vocabulary are
    preferred (a pull, never a fence — off-goal seams still compete).
    None if the field has no open surprise to offer."""
    if analysis is None:
        analysis = analyze_field(smc)
    if now is None:
        now = datetime.utcnow()
    me = owner or agent_id()
    active = active_claims(now) if respect_claims else {}
    goal_syms = _goal_symbols(smc, goal_threads) if goal_threads else frozenset()
    best, best_key = None, None
    for s in analysis["surprises"]:
        if _is_resolved(smc, s["a"], s["b"]):
            continue
        if claimed_by_other("bridge", target_key([s["a"], s["b"]]), me, active=active):
            continue
        damped = s["score"] * _recency_factor(smc, s["a"], s["b"], now, half_life_hours) \
            * _goal_boost(smc, s["a"], s["b"], goal_syms, goal_weight)
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
        prompt=f"{ctx}\n\n## Task\n{task}{_failure_block([a.id, b.id])}",
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
        prompt=f"{ctx}\n\n## Task\n{task}{_failure_block([m.id])}",
        symbols=list(m.symbols),
        targets=[m.id],
        thread_name=f"pursue-deepen-{m.id[:8]}",
    )


def plan_self(
    smc: SymbolicMemoryCore,
    analysis: Optional[dict] = None,
    template: str = DEFAULT_TEMPLATES["self"],
    spec_threads: tuple = DEFAULT_SPEC_THREADS,
    window: int = 8,
    owner: Optional[str] = None,
    respect_claims: bool = True,
) -> Optional[Pursuit]:
    """Plan a self-pursuit: the field reasoning about its own pipeline.

    Pairs a specification motif (from ``spec_threads`` — documents that
    govern this software) with the pipeline's own observed behavior (the
    trajectory tail and regime verdict), and asks what revision the spec
    implies. Captures are improvement PROPOSALS — motifs, reviewed like
    anything else; they cross into code only through a carrier's
    judgment. Spec motifs are taken most-connected-first, skipping ones
    already self-pursued (flow ledger) or claimed by another owner.
    """
    if analysis is None:
        analysis = analyze_field(smc)
    me = owner or agent_id()
    from symbolic_recursion.core.flow import load_flow
    already = {tuple(e.get("targets") or []) for e in load_flow()
               if e.get("kind") == "self"}
    active = active_claims() if respect_claims else {}
    g = analysis["graph"]
    candidates = sorted(
        (m for m in smc.list_motifs() if m.thread_id in spec_threads),
        key=lambda m: (-g.weighted_degree(m.id), m.id),
    )
    spec = None
    for c in candidates:
        if (c.id,) in already:
            continue
        if claimed_by_other("self", target_key([c.id]), me, active=active):
            continue
        spec = c
        break
    if spec is None:
        return None

    from symbolic_recursion.graph.trajectory import classify_regime, load_events
    events = load_events()
    verdict = classify_regime(events, window=window)
    tail = []
    for e in events[-window:]:
        met = e["metrics"]
        tail.append(f"- {e['ts'][:16]} {e['event'].get('type','?')}: "
                    f"motifs={met['motif_count']} communities={met['community_count']} "
                    f"binding={met['narrative_binding']} open_surprises={e.get('open_surprises','?')}")
    telemetry = (f"## Observed pipeline behavior (trajectory tail)\n"
                 f"regime: {verdict['regime']} ({'; '.join(verdict['evidence'][-2:])})\n"
                 + "\n".join(tail))

    task = template.format(
        spec_symbols=", ".join(spec.symbols),
        regime=verdict["regime"],
    )
    ctx = _context_block(smc, g, [spec.id])
    return Pursuit(
        kind="self",
        prompt=f"{ctx}\n\n{telemetry}\n\n## Task\n{task}{_failure_block([spec.id])}",
        symbols=list(spec.symbols) + ["proposal"],
        targets=[spec.id],
        thread_name=f"pursue-self-{spec.id[:12]}",
    )


def execute(
    smc: SymbolicMemoryCore, tm, pursuit: Pursuit, model: str,
    review_cfg: Optional[dict] = None,
    owner: Optional[str] = None,
    claim_ttl_hours: float = DEFAULT_TTL_HOURS,
) -> PursuitResult:
    """Fire a planned pursuit through the existing ask -> capture -> link
    path. ``tm`` is a threads.manager.ThreadManager.

    The owner claims the target for the duration of the model call
    (``core.claims``) so a concurrent session's planner skips it, and
    releases the claim when done — a rejected pair is immediately open
    for another attempt.

    With ``review_cfg = {"enabled": True, "model": ...}`` the capture is
    read by the reviewer BEFORE the strings are tied: links to the
    targets are made only on ``accept``. The card stays in the field and
    the flow ledger either way; a rejected bridge leaves the surprise
    unresolved — an open question for a better attempt.

    Stale-plan guard: on a shared field another writer may bridge the
    planned pair between planning and execution (claims prevent this
    while a claim is held, not before). A bridge whose pair is already
    resolved is skipped rather than doubled, and nothing is claimed."""
    if pursuit.kind == "bridge" and len(pursuit.targets) == 2 \
            and _is_resolved(smc, pursuit.targets[0], pursuit.targets[1]):
        return PursuitResult(kind=pursuit.kind, targets=list(pursuit.targets),
                             thread_name=pursuit.thread_name,
                             skipped="pair resolved since planning")
    me = owner or agent_id()
    key = target_key(pursuit.targets)
    claim(pursuit.kind, key, me, ttl_hours=claim_ttl_hours)
    try:
        return _execute_claimed(smc, tm, pursuit, model, review_cfg, me)
    finally:
        release(pursuit.kind, key, me)


def _execute_claimed(smc, tm, pursuit, model, review_cfg, me) -> PursuitResult:
    thread = tm.new_thread(pursuit.thread_name, model=model)
    resp = thread.ask(pursuit.prompt)
    m = tm.capture_as_motif(thread, pursuit.symbols, resp)

    review = None
    if review_cfg and review_cfg.get("enabled"):
        from symbolic_recursion.core.review import review_capture
        review = review_capture(
            smc, resp, pursuit.targets,
            model=review_cfg.get("model", model),
            query_fn=review_cfg.get("query_fn"),
        )

    if review is None or review.verdict == "accept":
        for target in pursuit.targets:
            smc.link_motifs(m.id, target)

    from symbolic_recursion.core.flow import record_flow
    entry = {"kind": pursuit.kind, "thread": pursuit.thread_name, "agent": me,
             "model": model, "motif_id": m.id, "targets": pursuit.targets,
             "prompt": pursuit.prompt, "response": resp}
    if review is not None:
        entry["review"] = {"verdict": review.verdict, "evidence": review.evidence,
                           "prediction": review.prediction, "reviewer_agent": me}
    record_flow(entry)
    return PursuitResult(
        kind=pursuit.kind,
        motif_id=m.id,
        targets=list(pursuit.targets),
        thread_name=pursuit.thread_name,
        review=(review.verdict if review else None),
        review_evidence=(review.evidence if review else ""),
        owner=me,
    )


def run_pursuits(
    smc: SymbolicMemoryCore,
    tm,
    pursue_queue: List[str],
    cfg: Optional[dict] = None,
    model: str = "llama3:instruct",
    analysis: Optional[dict] = None,
    owner: Optional[str] = None,
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
    review_cfg = cfg.get("review")
    me = owner or agent_id()

    results: List[PursuitResult] = []
    bridge = plan_bridge(smc, analysis, templates["bridge"], half_life_hours=half_life, owner=me)
    if bridge is not None and budget > 0:
        results.append(execute(smc, tm, bridge, model, review_cfg=review_cfg, owner=me))
        budget -= 1
    else:
        results.append(PursuitResult(kind="bridge", skipped="no open, unclaimed surprising connection"))

    while budget > 0 and pursue_queue:
        mid = pursue_queue.pop(0)
        other = claimed_by_other("deepen", mid, me)
        if other:
            results.append(PursuitResult(kind="deepen", skipped=f"{mid} claimed by {other}"))
            continue
        plan = plan_deepen(smc, mid, analysis, templates["deepen"])
        if plan is None:
            results.append(PursuitResult(kind="deepen", skipped=f"unknown motif {mid}"))
            continue
        results.append(execute(smc, tm, plan, model, review_cfg=review_cfg, owner=me))
        budget -= 1
    return results
