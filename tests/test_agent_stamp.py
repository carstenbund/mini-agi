"""The agent id reaches every record that matters."""
from symbolic_recursion.core.agent import agent_id
from symbolic_recursion.core.flow import load_flow, record_flow, render_trace
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.documents.capture import capture_document
from symbolic_recursion.graph.trajectory import load_events, record_event, render_trajectory
from symbolic_recursion.threads.manager import ThreadManager


def test_agent_id_reads_env_with_fallback(monkeypatch):
    assert agent_id() == "test-agent"
    monkeypatch.setenv("SMC_AGENT", "  loop-nightly ")
    assert agent_id() == "loop-nightly"
    monkeypatch.delenv("SMC_AGENT")
    assert agent_id() == "anonymous"


def test_motif_roundtrip_and_legacy_default():
    m = MotifNode(id="x", symbols=[], content="c", thread_id="t", agent="alice")
    assert MotifNode.from_dict(m.to_dict()).agent == "alice"
    assert MotifNode.from_dict({"id": "old"}).agent == ""


def test_capture_paths_stamp_agent(smc_with_motifs):
    tm = ThreadManager(smc_with_motifs)
    t = tm.new_thread("s")
    m = tm.capture_as_motif(t, ["a"], "text")
    assert m.agent == "test-agent"
    plan = capture_document(smc_with_motifs, "# Doc\n\nbody\n\n## Sec\n\ntext", prefix="d")
    assert all(smc_with_motifs.get_motif(s["id"]).agent == "test-agent" for s in plan["motifs"])


def test_journal_and_ledger_lines_carry_agent(smc_with_motifs):
    record_event(smc_with_motifs, {"type": "capture", "motif_id": "m1"})
    assert load_events()[-1]["agent"] == "test-agent"
    record_flow({"kind": "chat", "motif_id": "m1", "prompt": "p", "response": "r"})
    assert load_flow()[-1]["agent"] == "test-agent"
    # an explicit agent in the entry is kept (a reviewer session recording for another)
    record_flow({"kind": "chat", "motif_id": "m2", "agent": "bob", "prompt": "p", "response": "r"})
    assert load_flow()[-1]["agent"] == "bob"


def test_renderers_show_agent(smc_with_motifs):
    record_event(smc_with_motifs, {"type": "capture", "motif_id": "m1"})
    record_event(smc_with_motifs, {"type": "capture", "motif_id": "m2"})
    record_event(smc_with_motifs, {"type": "capture", "motif_id": "m3"})
    panel = render_trajectory(load_events())
    assert "| agent |" in panel and "| test-agent |" in panel
    record_flow({"kind": "chat", "motif_id": "m1", "prompt": "p", "response": "r"})
    assert "agent test-agent" in render_trace(smc_with_motifs, "m1")
