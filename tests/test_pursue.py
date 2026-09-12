from __future__ import annotations

import pytest

import symbolic_recursion.threads.manager as tmgr
from symbolic_recursion.core.model_stub import stub_response
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.pursue import (
    plan_bridge,
    plan_deepen,
    execute,
    run_pursuits,
)
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
    """Two reference-triangles from different threads, one bridge edge."""
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a1", ["justice"], "A", refs=["a2", "a3"]))
    smc.add_motif(_motif("a2", ["justice", "fairness"], "A", refs=["a3"]))
    smc.add_motif(_motif("a3", ["fairness"], "A"))
    smc.add_motif(_motif("b1", ["recursion"], "B", refs=["b2", "b3"]))
    smc.add_motif(_motif("b2", ["recursion", "loops"], "B", refs=["b3"]))
    smc.add_motif(_motif("b3", ["loops"], "B"))
    smc.link_motifs("a3", "b1")
    return smc


def test_plan_bridge_fills_template_from_field():
    smc = _bridged_field()
    plan = plan_bridge(smc)
    assert plan is not None and plan.kind == "bridge"
    assert set(plan.targets) == {"a3", "b1"}
    # the field's specifics reached the prompt: symbols, threads, context
    assert "fairness" in plan.prompt and "recursion" in plan.prompt
    assert "A x B" in plan.prompt
    assert plan.prompt.startswith("## Context")
    assert "## Task" in plan.prompt
    # symbol inheritance from both endpoints
    assert "fairness" in plan.symbols and "recursion" in plan.symbols


def test_plan_bridge_none_without_surprise():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("x", ["one"], refs=["y"]))
    smc.add_motif(_motif("y", ["one"]))
    assert plan_bridge(smc) is None


def test_execute_captures_and_links():
    smc = _bridged_field()
    tm = ThreadManager(smc)
    plan = plan_bridge(smc)
    result = execute(smc, tm, plan, model="stub-model")
    assert result.motif_id in smc.motifs
    captured = smc.get_motif(result.motif_id)
    assert captured.thread_id == plan.thread_name
    assert captured.content.startswith("[stub:stub-model]")
    assert set(captured.references) >= {"a3", "b1"}


def test_plan_deepen_embeds_content_and_targets_parent():
    smc = _bridged_field()
    plan = plan_deepen(smc, "a1")
    assert plan.kind == "deepen"
    assert plan.targets == ["a1"]
    assert "content of a1" in plan.prompt
    assert plan.symbols == ["justice"]
    assert plan_deepen(smc, "ghost") is None


def test_run_pursuits_is_opt_in():
    smc = _bridged_field()
    tm = ThreadManager(smc)
    assert run_pursuits(smc, tm, ["a1"], cfg=None) == []
    assert run_pursuits(smc, tm, ["a1"], cfg={"enabled": False}) == []


def test_run_pursuits_consumes_queue_within_budget():
    smc = _bridged_field()
    tm = ThreadManager(smc)
    queue = ["a1", "b2"]
    results = run_pursuits(
        smc, tm, queue, cfg={"enabled": True, "max_per_cycle": 2}, model="stub-model"
    )
    fired = [r for r in results if r.motif_id]
    assert [r.kind for r in fired] == ["bridge", "deepen"]
    assert queue == ["b2"]  # unfired entry left for the next cycle
    assert len(smc.motifs) == 8  # 6 + one bridge + one deepen capture


def test_run_pursuits_reports_skip_when_no_surprise():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("x", ["one"], refs=["y"]))
    smc.add_motif(_motif("y", ["one"]))
    tm = ThreadManager(smc)
    results = run_pursuits(smc, tm, [], cfg={"enabled": True})
    assert results[0].kind == "bridge" and results[0].skipped
    assert len(smc.motifs) == 2


def test_template_override_is_used():
    smc = _bridged_field()
    plan = plan_bridge(smc, template="CUSTOM {a_symbols} vs {b_symbols}")
    assert "CUSTOM" in plan.prompt


def _aged(smc: SymbolicMemoryCore, mid: str, hours: float, now) -> None:
    from datetime import timedelta
    smc.get_motif(mid).created_at = (now - timedelta(hours=hours)).isoformat()


def test_recency_damping_prefers_seasoned_surprise():
    from datetime import datetime
    now = datetime.utcnow()
    smc = _bridged_field()
    for mid in smc.motifs:
        _aged(smc, mid, 100.0, now)
    # a second, weaker cross-cluster edge whose endpoints are brand new:
    # raw score would lose to the seasoned bridge only if damping works,
    # so make the FRESH pair the raw winner (reference + cross-thread).
    smc.add_motif(_motif("fresh-a", ["justice"], "A2", refs=["fresh-b"]))
    smc.add_motif(_motif("fresh-b", ["loops"], "B2", refs=["b3", "b2"]))
    _aged(smc, "fresh-a", 0.01, now)
    _aged(smc, "fresh-b", 0.01, now)
    plan = plan_bridge(smc, now=now)
    # the minutes-old fresh edge is damped to near zero; the 100h bridge wins
    assert set(plan.targets) == {"a3", "b1"}


def test_resolved_pairs_are_skipped():
    smc = _bridged_field()
    # a capture already references both bridge endpoints -> pair resolved
    smc.add_motif(_motif("capture", ["justice", "recursion"], "pursue-x",
                         refs=["a3", "b1"]))
    plan = plan_bridge(smc)
    assert plan is None or set(plan.targets) != {"a3", "b1"}


def test_constructed_edges_are_resolved():
    smc = _bridged_field()
    # a pursuit capture linking to one endpoint of each cluster: its own
    # edges are constructed, not surprising — never pursued
    smc.add_motif(_motif("p1", ["justice", "recursion"], "pursue-bridge-x",
                         refs=["a3", "b1"]))
    plan = plan_bridge(smc)
    # p1's REFERENCE edges (to its own targets a3, b1) are constructed and
    # never pursued; a shared-symbol edge to a non-target stays fair game
    if plan is not None:
        assert set(plan.targets) != {"p1", "a3"}
        assert set(plan.targets) != {"p1", "b1"}


def test_recency_config_passes_through_run_pursuits():
    smc = _bridged_field()
    tm = ThreadManager(smc)
    # an absurd half-life leaves every fresh edge damped ~0, but selection
    # still picks the max — the bridge must still fire (damping reorders,
    # it never silences the only candidate)
    results = run_pursuits(
        smc, tm, [], cfg={"enabled": True, "recency_half_life_hours": 1e9},
        model="stub-model",
    )
    assert results[0].motif_id is not None
