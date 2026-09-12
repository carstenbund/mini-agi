from __future__ import annotations

import pytest

import symbolic_recursion.threads.manager as tmgr
from symbolic_recursion.core.exhaust import run_exhaust
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.pursue import _parse_minted, plan_bridge, execute
from symbolic_recursion.threads.manager import ThreadManager


def _motif(mid, symbols, thread="t", refs=None, content=None):
    return MotifNode(id=mid, symbols=list(symbols),
                     content=content or f"content of {mid}",
                     thread_id=thread, references=list(refs or []))


def _rich_field() -> SymbolicMemoryCore:
    """Three clusters, two cross-cluster seams, all seasoned."""
    from datetime import datetime, timedelta
    old = (datetime.utcnow() - timedelta(hours=100)).isoformat()
    smc = SymbolicMemoryCore()
    for cluster, syms in (("a", ["justice"]), ("b", ["recursion"]), ("c", ["trust"])):
        for i in range(3):
            m = _motif(f"{cluster}{i}", syms, thread=cluster.upper(),
                       refs=[f"{cluster}{j}" for j in range(i)])
            m.created_at = old
            smc.add_motif(m)
    smc.link_motifs("a2", "b0")
    smc.link_motifs("b2", "c0")
    return smc


@pytest.fixture
def _distinct_stub(monkeypatch):
    """Each call returns clearly different text, so novelty stays high."""
    counter = {"n": 0}
    vocab = ["zebra quantum", "harpsichord nebula", "glacier trombone",
             "obsidian kite", "meridian fox"]
    def fake(prompt, model="stub", timeout=None):
        counter["n"] += 1
        return f"synthesis {vocab[counter['n'] % len(vocab)]} number {counter['n']}"
    monkeypatch.setattr(tmgr, "query_ollama", fake)
    return counter


@pytest.fixture
def _echo_stub(monkeypatch):
    """Every call returns the same text: novelty collapses -> strikes."""
    monkeypatch.setattr(tmgr, "query_ollama",
                        lambda prompt, model="stub", timeout=None: "the same synthesis every time")


def test_exhaust_stops_settled(_distinct_stub):
    smc = _rich_field()
    report = run_exhaust(smc, ThreadManager(smc), model="stub",
                         cfg={"max_pursuits": 10, "min_score": 0.02,
                              "novelty_strike": 0.0})
    # two seams to bridge; then nothing above the floor (fresh captures damped)
    assert report.stop_reason == "settled"
    assert 1 <= report.fired <= 3


def test_exhaust_stops_on_diminishing_returns(_echo_stub):
    smc = _rich_field()
    report = run_exhaust(smc, ThreadManager(smc), model="stub",
                         cfg={"max_pursuits": 10, "min_score": 0.0,
                              "patience": 2, "novelty_strike": 0.9})
    assert report.stop_reason in ("diminishing-returns", "settled")
    if report.stop_reason == "diminishing-returns":
        assert report.fired >= 2


def test_exhaust_respects_budget(_distinct_stub):
    smc = _rich_field()
    report = run_exhaust(smc, ThreadManager(smc), model="stub",
                         cfg={"max_pursuits": 1, "min_score": 0.0,
                              "novelty_strike": 0.0})
    assert report.stop_reason in ("budget", "settled")
    assert report.fired <= 1


def test_exhaust_journals_via_on_fire(_distinct_stub):
    smc = _rich_field()
    seen = []
    run_exhaust(smc, ThreadManager(smc), model="stub",
                cfg={"max_pursuits": 2, "min_score": 0.0, "novelty_strike": 0.0},
                on_fire=lambda r: seen.append(r.motif_id))
    assert seen and all(mid in smc.motifs for mid in seen)


# --- symbol minting ---

def test_parse_minted_validates_and_caps():
    got = _parse_minted(
        "text...\nSYMBOLS: reciprocity-deficit, Justice, ok-sym, BAD SYM, x, "
        "another-one, too-many",
        existing=["justice"])
    # 'justice' exists (case-folded), 'BAD SYM' malformed, 'x' too short,
    # cap at three valid new ones
    assert got == ["reciprocity-deficit", "ok-sym", "another-one"]
    assert _parse_minted("no line here", ["a"]) == []


def test_minting_applies_only_when_links_tied(monkeypatch):
    smc = _rich_field()
    monkeypatch.setattr(
        tmgr, "query_ollama",
        lambda prompt, model="s", timeout=None:
            "a genuine synthesis\n\nSYMBOLS: fresh-notion, second-notion")
    plan = plan_bridge(smc)
    accepted = execute(
        smc, ThreadManager(smc), plan, model="s",
        review_cfg={"enabled": True,
                    "query_fn": lambda p, m: "VERDICT: accept\nEVIDENCE: fine\nPREDICTION: none"})
    assert accepted.minted == ["fresh-notion", "second-notion"]
    assert "fresh-notion" in smc.get_motif(accepted.motif_id).symbols

    plan2 = plan_bridge(smc)
    rejected = execute(
        smc, ThreadManager(smc), plan2, model="s",
        review_cfg={"enabled": True,
                    "query_fn": lambda p, m: "VERDICT: reject\nEVIDENCE: no\nPREDICTION: none"})
    assert rejected.minted == []
    assert "fresh-notion" not in smc.get_motif(rejected.motif_id).symbols
