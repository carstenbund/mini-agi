"""The exhaust loop — autonomy in time, bounded by three stop conditions.

Runs pursuits until the field itself says stop:

- ``settled``              no open seam scores above ``min_score`` (input-
                           side exhaustion: nothing worth asking)
- ``diminishing-returns``  ``patience`` consecutive strikes, where a
                           strike is a capture the field judges weak — a
                           review reject/revise, or novelty below the
                           threshold (output-side exhaustion: pursuing
                           stops producing)
- ``budget``               the hard rail (autonomy without a rail is a
                           bill)

Each firing re-reads the field, so every capture changes the next
selection. Everything is journaled: one pursuit event per firing, one
``exhaust-stop`` event with the reason at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.core.pursue import (
    DEFAULT_TEMPLATES, PursuitResult, plan_bridge, execute,
)
from symbolic_recursion.graph import analyze_field
from symbolic_recursion.utils.novelty import novelty_index

DEFAULT_MAX_PURSUITS = 10
DEFAULT_MIN_SCORE = 0.02
DEFAULT_PATIENCE = 2
DEFAULT_NOVELTY_STRIKE = 0.2


@dataclass
class ExhaustReport:
    stop_reason: str
    fired: int = 0
    results: List[PursuitResult] = field(default_factory=list)


def run_exhaust(
    smc: SymbolicMemoryCore,
    tm,
    model: str,
    cfg: Optional[Dict] = None,
    review_cfg: Optional[Dict] = None,
    goal_threads: Optional[tuple] = None,
    on_fire=None,
) -> ExhaustReport:
    """Pursue until quiescence. ``cfg`` keys (all optional):
    ``max_pursuits``, ``min_score``, ``patience``, ``novelty_strike``,
    ``templates``. ``on_fire(result)`` is called after each firing (the
    CLI uses it to persist and journal incrementally)."""
    cfg = cfg or {}
    max_pursuits = int(cfg.get("max_pursuits", DEFAULT_MAX_PURSUITS))
    min_score = float(cfg.get("min_score", DEFAULT_MIN_SCORE))
    patience = int(cfg.get("patience", DEFAULT_PATIENCE))
    novelty_strike = float(cfg.get("novelty_strike", DEFAULT_NOVELTY_STRIKE))
    templates = {**DEFAULT_TEMPLATES, **cfg.get("templates", {})}

    report = ExhaustReport(stop_reason="budget")
    strikes = 0
    while report.fired < max_pursuits:
        analysis = analyze_field(smc)
        plan = plan_bridge(smc, analysis, templates["bridge"],
                           goal_threads=goal_threads)
        if plan is None or plan.score < min_score:
            report.stop_reason = "settled"
            break
        result = execute(smc, tm, plan, model, review_cfg=review_cfg)
        if result.skipped:
            report.results.append(result)
            continue  # stale plan: another writer got there; re-read and go on
        report.fired += 1
        report.results.append(result)
        if on_fire:
            on_fire(result)

        weak = result.review in ("reject", "revise")
        if not weak:
            captured = smc.get_motif(result.motif_id)
            if captured is not None:
                n = novelty_index(smc, captured)
                weak = n.get("novelty_index", 1.0) < novelty_strike
        strikes = strikes + 1 if weak else 0
        if strikes >= patience:
            report.stop_reason = "diminishing-returns"
            break
    return report
