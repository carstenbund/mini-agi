from __future__ import annotations

import pytest

import symbolic_recursion.threads.manager as tmgr
from symbolic_recursion.core.flow import load_flow, record_flow, render_trace
from symbolic_recursion.core.model_stub import stub_response
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.pursue import plan_bridge, execute
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


def test_record_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "f.jsonl")
    record_flow({"kind": "chat", "motif_id": "x", "prompt": "p", "response": "r"}, path=path)
    events = load_flow(path)
    assert len(events) == 1 and events[0]["prompt"] == "p" and "ts" in events[0]


def test_execute_writes_flow_entry():
    smc = _bridged_field()
    plan = plan_bridge(smc)
    result = execute(smc, ThreadManager(smc), plan, model="stub-model")
    events = load_flow()  # env-redirected by conftest
    assert len(events) == 1
    e = events[0]
    assert e["motif_id"] == result.motif_id
    assert e["prompt"] == plan.prompt
    assert e["response"].startswith("[stub:stub-model]")
    assert e["targets"] == plan.targets


def test_render_trace_shows_text_and_lineage():
    smc = _bridged_field()
    plan = plan_bridge(smc)
    result = execute(smc, ThreadManager(smc), plan, model="stub-model")
    out = render_trace(smc, result.motif_id)
    assert "### Prompt shown to the model" in out
    assert "## Task" in out                     # the actual prompt text is there
    assert "[stub:stub-model]" in out           # and the response
    assert "## Grew from (references)" in out
    # trace of a parent shows the capture as a child
    parent_trace = render_trace(smc, plan.targets[0])
    assert "## Grew into (referenced by)" in parent_trace
    assert result.motif_id in parent_trace


def test_render_trace_handles_missing_ledger_and_motif():
    smc = _bridged_field()
    out = render_trace(smc, "a1", flow_events=[])
    assert "None in the flow ledger" in out
    assert "No motif with id" in render_trace(smc, "ghost")
