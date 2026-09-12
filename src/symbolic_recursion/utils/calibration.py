"""Calibration — banded novelty, the nursery, and the cold-start floor.

The scalar novelty gate conflates three situations distance cannot
distinguish. The profile (max_sim, mean_sim) against the field separates
them:

- ``redundant``  high max similarity: a paraphrase of something held
- ``band``       binds strongly somewhere while distant from the rest —
                 the adjacent possible, the material worth pursuing
- ``unbound``    far from everything: alien insight or banal noise,
                 locally undecidable — so neither celebrated nor
                 discarded but placed in the NURSERY: marginal custody
                 at admission. If later material binds to it, it
                 graduates; if nothing ever does, it goes stale
                 (reported, never silently deleted).
- ``seed``       cold-start floor: below a minimum field size novelty
                 judgments have no sample to stand on, so nothing is
                 gated.

Deterministic, stdlib only. The nursery ledger lives at
``data/nursery.json`` (``SMC_NURSERY_PATH`` overrides).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.embeddings.embedder import embed_text, cosine_sparse

COLD_START_FLOOR = 15
REDUNDANT_MAX_SIM = 0.75
UNBOUND_MAX_SIM = 0.25
STALE_AFTER_HOURS = 168.0


def novelty_profile(
    smc: SymbolicMemoryCore,
    m: MotifNode,
    cold_start_floor: int = COLD_START_FLOOR,
    redundant_at: float = REDUNDANT_MAX_SIM,
    unbound_below: float = UNBOUND_MAX_SIM,
) -> Dict:
    """Profile a motif against the field: {"max_sim", "mean_sim", "verdict"}."""
    others = [x for x in smc.list_motifs() if x.id != m.id]
    if len(others) < cold_start_floor:
        return {"max_sim": 0.0, "mean_sim": 0.0, "verdict": "seed"}
    vec = embed_text(m.content)
    sims = [cosine_sparse(vec, embed_text(x.content)) for x in others]
    max_sim = max(sims)
    mean_sim = sum(sims) / len(sims)
    if max_sim >= redundant_at:
        verdict = "redundant"
    elif max_sim <= unbound_below:
        verdict = "unbound"
    else:
        verdict = "band"
    return {"max_sim": round(max_sim, 4), "mean_sim": round(mean_sim, 4),
            "verdict": verdict}


def nursery_path() -> str:
    return os.path.abspath(
        os.environ.get("SMC_NURSERY_PATH", os.path.join("data", "nursery.json"))
    )


def _load_nursery() -> Dict[str, Dict]:
    path = nursery_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_nursery(reg: Dict[str, Dict]) -> None:
    path = nursery_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def _degree(smc: SymbolicMemoryCore, motif_id: str) -> int:
    m = smc.get_motif(motif_id)
    if m is None:
        return 0
    incoming = sum(1 for x in smc.list_motifs() if motif_id in x.references)
    return len(m.references) + incoming


def nursery_enter(smc: SymbolicMemoryCore, motif_id: str,
                  now: Optional[datetime] = None) -> Dict:
    """Place an unbound motif in custody. Records its degree at entry so
    graduation can be judged by binding gained afterwards."""
    reg = _load_nursery()
    reg[motif_id] = {
        "entered": (now or datetime.utcnow()).isoformat(),
        "entry_degree": _degree(smc, motif_id),
        "graduated": None,
    }
    _save_nursery(reg)
    return reg[motif_id]


def nursery_pass(smc: SymbolicMemoryCore,
                 now: Optional[datetime] = None,
                 stale_after_hours: float = STALE_AFTER_HOURS) -> List[Dict]:
    """Review custody: a resident graduates when the field has bound to it
    (degree grew past its entry degree); one unbound past the stale window
    is flagged stale — reported, never deleted. Returns status rows."""
    now = now or datetime.utcnow()
    reg = _load_nursery()
    rows = []
    changed = False
    for mid, rec in sorted(reg.items()):
        if rec.get("graduated"):
            rows.append({"motif": mid, "status": "graduated",
                         "at": rec["graduated"]})
            continue
        if smc.get_motif(mid) is None:
            rows.append({"motif": mid, "status": "missing"})
            continue
        if _degree(smc, mid) > rec.get("entry_degree", 0):
            rec["graduated"] = now.isoformat()
            changed = True
            rows.append({"motif": mid, "status": "graduated", "at": rec["graduated"]})
            continue
        try:
            entered = datetime.fromisoformat(rec["entered"])
            age_h = (now - entered).total_seconds() / 3600.0
        except (KeyError, ValueError, TypeError):
            age_h = 0.0
        rows.append({"motif": mid,
                     "status": "stale" if age_h > stale_after_hours else "held",
                     "age_hours": round(age_h, 1)})
    if changed:
        _save_nursery(reg)
    return rows
