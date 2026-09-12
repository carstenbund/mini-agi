from __future__ import annotations

import pytest

import symbolic_recursion.threads.manager as tmgr
from symbolic_recursion.core.model_stub import stub_response
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.pursue import plan_bridge, execute
from symbolic_recursion.core.review import build_review_prompt, parse_review
from symbolic_recursion.graph.trajectory import classify_regime
from symbolic_recursion.threads.manager import ThreadManager


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    monkeypatch.setattr(
        tmgr, "query_ollama",
        lambda prompt, model="stub", timeout=None: stub_response(prompt, model, timeout),
    )


def _motif(mid: str, symbols, thread: str = "test", refs=None) -> MotifNode:
    return MotifNode(
        id=mid, symbols=list(symbols), content=f"content of {mid}",
        thread_id=thread, references=list(refs or []),
    )


def _bridged_field() -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a1", ["justice"], "A", refs=["a2"]))
    smc.add_motif(_motif("a2", ["justice"], "A"))
    smc.add_motif(_motif("b1", ["recursion"], "B", refs=["b2"]))
    smc.add_motif(_motif("b2", ["recursion"], "B"))
    smc.link_motifs("a2", "b1")
    return smc


def test_parse_verdicts():
    r = parse_review("VERDICT: accept\nEVIDENCE: covers both\nPREDICTION: X rises")
    assert (r.verdict, r.evidence, r.prediction) == ("accept", "covers both", "X rises")
    assert r.parse_ok
    assert parse_review("verdict: REJECT\nEVIDENCE: sticker words").verdict == "reject"
    # unparseable degrades to revise — conservative: links withheld
    r = parse_review("the model rambled instead")
    assert r.verdict == "revise" and not r.parse_ok


def test_review_prompt_contains_targets_and_candidate():
    smc = _bridged_field()
    prompt = build_review_prompt(smc, "CANDIDATE TEXT", ["a2", "b1"])
    assert "content of a2" in prompt and "content of b1" in prompt
    assert "CANDIDATE TEXT" in prompt
    assert "VERDICT:" in prompt


def _run_reviewed(verdict_line: str):
    smc = _bridged_field()
    plan = plan_bridge(smc)
    result = execute(
        smc, ThreadManager(smc), plan, model="gen-model",
        review_cfg={"enabled": True, "model": "review-model",
                    "query_fn": lambda p, m: verdict_line},
    )
    return smc, plan, result


def test_accept_ties_the_links():
    smc, plan, result = _run_reviewed("VERDICT: accept\nEVIDENCE: genuine abstraction\nPREDICTION: none")
    assert result.review == "accept"
    captured = smc.get_motif(result.motif_id)
    assert set(captured.references) >= set(plan.targets)


def test_reject_keeps_card_but_withholds_links_and_reopens_pair():
    smc, plan, result = _run_reviewed("VERDICT: reject\nEVIDENCE: restatement with sticker words\nPREDICTION: none")
    assert result.review == "reject"
    captured = smc.get_motif(result.motif_id)
    assert captured is not None                    # the card stays
    assert captured.references == []               # the strings are not tied
    # the pair remains an open question: plan_bridge can select it again
    replan = plan_bridge(smc)
    assert replan is not None and set(replan.targets) == set(plan.targets)


def test_review_verdict_lands_in_flow_ledger():
    from symbolic_recursion.core.flow import load_flow
    _smc, _plan, result = _run_reviewed("VERDICT: revise\nEVIDENCE: close but no consequence\nPREDICTION: none")
    e = load_flow()[-1]
    assert e["review"]["verdict"] == "revise"
    assert "consequence" in e["review"]["evidence"]


def test_thrashing_regime_from_rejected_pursuits():
    def _ev(review):
        return {"ts": "t", "event": {"type": "pursuit", "review": review},
                "metrics": {"motif_count": 10, "reference_edges": 10,
                            "community_count": 2, "narrative_binding": 0.4},
                "open_surprises": 5}
    events = [_ev("reject"), _ev("revise"), _ev("reject"), _ev("accept")]
    v = classify_regime(events)
    assert v["regime"] == "thrashing"
    assert any("1/4" in x for x in v["evidence"])
