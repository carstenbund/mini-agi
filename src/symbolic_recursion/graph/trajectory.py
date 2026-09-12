"""Trajectory instrumentation — evaluating the direction the field takes.

An instrument panel, not a fence: nothing here gates or steers. Every
field mutation appends one line to an append-only journal
(``data/trajectory.jsonl``); the regime classifier reads the recent
window and names the trajectory using the field's own mis-governance
vocabulary (captured in the stewardship motifs):

- ``fragmenting``     under-regulation: communities up, binding down —
                      accumulating without integrating
- ``stagnating``      over-regulation: activity without structural
                      consequence — grooming what is already known
- ``concentrating``   misdirected regulation: pursuits locked onto one
                      region — the obsession signature
- ``oscillating``     the healthy rhythm: communities split and merge,
                      binding dips on capture and recovers on pursuit
- ``consolidating``   binding rising, communities merging — fine in
                      moderation, watch for slide into stagnation
- ``idle`` / ``mixed`` / ``insufficient-history``

Rules are deterministic and every verdict ships with its evidence, so
the diagnosis can be argued with. The steward judges the destination;
this panel only measures the direction.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.graph.analytics import analyze_field

BINDING_EPS = 0.01


def trajectory_path() -> str:
    """``SMC_TRAJECTORY_PATH`` env var wins; else ``data/trajectory.jsonl``
    relative to the working directory (same convention as the motif store)."""
    return os.path.abspath(
        os.environ.get("SMC_TRAJECTORY_PATH", os.path.join("data", "trajectory.jsonl"))
    )


def _open_surprises(smc: SymbolicMemoryCore, analysis: dict) -> int:
    """Count unresolved cross-community edges — the open-question ledger."""
    from symbolic_recursion.core.pursue import _is_resolved
    from symbolic_recursion.graph.analytics import surprising_connections

    all_surprises = surprising_connections(
        analysis["graph"], analysis["communities"], smc,
        top_n=len(analysis["graph"].edges) or 1,
    )
    return sum(1 for s in all_surprises if not _is_resolved(smc, s["a"], s["b"]))


def record_event(
    smc: SymbolicMemoryCore,
    event: Dict,
    path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict:
    """Append one journal line: the event plus the field's state after it.

    ``event`` is a small dict with at least ``{"type": ...}`` — e.g.
    ``{"type": "pursuit", "kind": "bridge", "motif_id": ..., "targets": [...]}``.
    For pursuit events the targets' communities are recorded too (the
    concentration signal). Returns the written line as a dict.
    """
    path = path or trajectory_path()
    analysis = analyze_field(smc)
    event = dict(event)
    if event.get("type") == "pursuit" and event.get("targets"):
        event["target_communities"] = sorted(
            {analysis["communities"].get(t, -1) for t in event["targets"]}
        )
    line = {
        "ts": (now or datetime.utcnow()).isoformat(),
        "event": event,
        "metrics": analysis["metrics"],
        "open_surprises": _open_surprises(smc, analysis),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


def load_events(path: Optional[str] = None) -> List[Dict]:
    path = path or trajectory_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def classify_regime(events: List[Dict], window: int = 8) -> Dict:
    """Name the trajectory over the recent window, with evidence.

    Deterministic rules over metric deltas; returns
    ``{"regime": str, "evidence": [str, ...]}``.
    """
    recent = events[-window:]
    if len(recent) < 3:
        return {"regime": "insufficient-history",
                "evidence": [f"only {len(recent)} journaled events (need 3)"]}

    first_m, last_m = recent[0]["metrics"], recent[-1]["metrics"]
    d_motifs = last_m["motif_count"] - first_m["motif_count"]
    d_comm = last_m["community_count"] - first_m["community_count"]
    d_bind = last_m["narrative_binding"] - first_m["narrative_binding"]
    d_open = recent[-1].get("open_surprises", 0) - recent[0].get("open_surprises", 0)
    bindings = [e["metrics"]["narrative_binding"] for e in recent]
    comms = [e["metrics"]["community_count"] for e in recent]
    ev = [
        f"motifs {first_m['motif_count']} -> {last_m['motif_count']}",
        f"communities {first_m['community_count']} -> {last_m['community_count']}",
        f"binding {first_m['narrative_binding']} -> {last_m['narrative_binding']}",
        f"open surprises {recent[0].get('open_surprises', 0)} -> {recent[-1].get('open_surprises', 0)}",
    ]

    pursuits = [e for e in recent if e["event"].get("type") == "pursuit"]
    target_comms = {c for p in pursuits for c in p["event"].get("target_communities", [])}
    if len(pursuits) >= 3 and len(target_comms) == 1:
        return {"regime": "concentrating",
                "evidence": ev + [f"{len(pursuits)} pursuits, all targeting community {target_comms.pop()}"]}

    if d_comm > 0 and d_bind < -BINDING_EPS:
        return {"regime": "fragmenting", "evidence": ev}

    if d_motifs == 0 and not pursuits:
        return {"regime": "idle", "evidence": ev}

    if d_motifs > 0 and abs(d_bind) <= BINDING_EPS and d_comm == 0 and d_open == 0:
        return {"regime": "stagnating",
                "evidence": ev + ["field grew with no structural consequence"]}

    dip_recovered = min(bindings) < min(bindings[0], bindings[-1]) - BINDING_EPS
    comm_both_ways = max(comms) > comms[0] and min(comms) < max(comms) and comms[-1] != max(comms)
    if dip_recovered or (max(comms) != min(comms) and comm_both_ways):
        return {"regime": "oscillating (healthy)",
                "evidence": ev + ["binding dipped and recovered" if dip_recovered
                                  else "communities split and re-merged"]}

    if d_bind > BINDING_EPS and d_comm <= 0:
        return {"regime": "consolidating", "evidence": ev}

    return {"regime": "mixed", "evidence": ev}


def render_trajectory(events: List[Dict], window: int = 8) -> str:
    """Markdown trend report over the recent window."""
    lines = [f"# Field Trajectory — {datetime.utcnow().date().isoformat()}", ""]
    if not events:
        lines.append("No journaled events yet — the panel starts recording "
                     "with the next capture, link, or pursuit.")
        return "\n".join(lines) + "\n"

    recent = events[-window:]
    lines += [
        f"{len(events)} events journaled; showing last {len(recent)}.",
        "",
        "| ts | event | motifs | refs | communities | binding | open surprises |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in recent:
        m = e["metrics"]
        etype = e["event"].get("type", "?")
        detail = e["event"].get("kind") or e["event"].get("motif_id", "")
        label = f"{etype}{':' + str(detail)[:12] if detail else ''}"
        lines.append(
            f"| {e['ts'][:16]} | {label} | {m['motif_count']} | {m['reference_edges']} "
            f"| {m['community_count']} | {m['narrative_binding']} | {e.get('open_surprises', '?')} |"
        )
    verdict = classify_regime(events, window=window)
    lines += ["", f"## Regime: {verdict['regime']}", ""]
    lines += [f"- {x}" for x in verdict["evidence"]]
    return "\n".join(lines) + "\n"
