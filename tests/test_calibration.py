from __future__ import annotations

from datetime import datetime, timedelta

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.utils.calibration import (
    novelty_profile,
    nursery_enter,
    nursery_pass,
)


def _motif(mid: str, content: str, refs=None) -> MotifNode:
    return MotifNode(id=mid, symbols=[mid], content=content, thread_id="t",
                     references=list(refs or []))


def _field(n: int, text: str = "alpha beta gamma delta") -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    for i in range(n):
        smc.add_motif(_motif(f"m{i}", f"{text} variant {i}"))
    return smc


def test_cold_start_floor_returns_seed():
    smc = _field(3)
    m = _motif("new", "totally unrelated content about zebras")
    smc.add_motif(m)
    assert novelty_profile(smc, m)["verdict"] == "seed"


def test_redundant_band_unbound_verdicts():
    smc = _field(20)
    dup = _motif("dup", "alpha beta gamma delta variant 3")
    smc.add_motif(dup)
    assert novelty_profile(smc, dup)["verdict"] == "redundant"

    unbound = _motif("alien", "zebra quantum harpsichord nebula")
    smc.add_motif(unbound)
    assert novelty_profile(smc, unbound)["verdict"] == "unbound"

    banded = _motif("adjacent", "alpha beta epsilon zeta established connection")
    smc.add_motif(banded)
    prof = novelty_profile(smc, banded)
    assert prof["verdict"] == "band"
    assert 0.25 < prof["max_sim"] < 0.75


def test_nursery_graduates_on_binding():
    now = datetime.utcnow()
    smc = _field(2)
    alien = _motif("alien", "zebra quantum harpsichord")
    smc.add_motif(alien)
    nursery_enter(smc, "alien", now=now)
    rows = nursery_pass(smc, now=now)
    assert rows == [{"motif": "alien", "status": "held", "age_hours": 0.0}]
    # the field binds to it -> graduation
    smc.add_motif(_motif("later", "quantum follow-up", refs=["alien"]))
    rows = nursery_pass(smc, now=now)
    assert rows[0]["status"] == "graduated"
    # graduation is remembered
    assert nursery_pass(smc, now=now)[0]["status"] == "graduated"


def test_nursery_flags_stale_never_deletes():
    now = datetime.utcnow()
    smc = _field(2)
    alien = _motif("alien", "zebra quantum harpsichord")
    smc.add_motif(alien)
    nursery_enter(smc, "alien", now=now - timedelta(hours=200))
    rows = nursery_pass(smc, now=now)
    assert rows[0]["status"] == "stale"
    assert smc.get_motif("alien") is not None
