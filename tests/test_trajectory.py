from __future__ import annotations

import json

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.graph.trajectory import (
    classify_regime,
    load_events,
    record_event,
    render_trajectory,
)


def _motif(mid: str, symbols, thread: str = "test", refs=None) -> MotifNode:
    return MotifNode(
        id=mid, symbols=list(symbols), content=f"content of {mid}",
        thread_id=thread, references=list(refs or []),
    )


def _event(motifs, comms, binding, open_s, etype="capture", target_comms=None):
    e = {"type": etype}
    if target_comms is not None:
        e["target_communities"] = target_comms
    return {
        "ts": "2026-09-12T00:00:00",
        "event": e,
        "metrics": {"motif_count": motifs, "reference_edges": motifs,
                    "community_count": comms, "narrative_binding": binding},
        "open_surprises": open_s,
    }


def test_record_and_load_roundtrip(tmp_path):
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a", ["one"], "A", refs=["b"]))
    smc.add_motif(_motif("b", ["two"], "B"))
    path = str(tmp_path / "traj.jsonl")
    line = record_event(smc, {"type": "capture", "motif_id": "b"}, path=path)
    assert line["metrics"]["motif_count"] == 2
    assert "open_surprises" in line
    events = load_events(path)
    assert len(events) == 1
    assert events[0]["event"]["motif_id"] == "b"
    # append-only
    record_event(smc, {"type": "link", "a": "a", "b": "b"}, path=path)
    assert len(load_events(path)) == 2


def test_pursuit_event_records_target_communities(tmp_path):
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a", ["one"], "A", refs=["b"]))
    smc.add_motif(_motif("b", ["two"], "B"))
    path = str(tmp_path / "traj.jsonl")
    line = record_event(
        smc, {"type": "pursuit", "kind": "bridge", "targets": ["a", "b"]}, path=path
    )
    assert "target_communities" in line["event"]


def test_regime_insufficient_history():
    assert classify_regime([_event(1, 1, 0.0, 0)])["regime"] == "insufficient-history"


def test_regime_fragmenting():
    events = [
        _event(10, 2, 0.50, 4),
        _event(13, 3, 0.42, 6),
        _event(16, 5, 0.31, 9),
    ]
    assert classify_regime(events)["regime"] == "fragmenting"


def test_regime_stagnating():
    events = [
        _event(10, 2, 0.50, 4),
        _event(12, 2, 0.50, 4),
        _event(14, 2, 0.505, 4),
    ]
    v = classify_regime(events)
    assert v["regime"] == "stagnating"
    assert any("no structural consequence" in x for x in v["evidence"])


def test_regime_oscillating_on_dip_and_recover():
    events = [
        _event(10, 2, 0.50, 4),
        _event(11, 3, 0.34, 6, etype="capture"),   # document lands, binding dips
        _event(12, 3, 0.44, 5, etype="pursuit"),   # pursuit weaves it back
        _event(13, 2, 0.52, 4, etype="pursuit"),
    ]
    assert classify_regime(events)["regime"] == "oscillating (healthy)"


def test_regime_concentrating():
    events = [
        _event(10, 2, 0.50, 4),
        _event(11, 2, 0.51, 4, etype="pursuit", target_comms=[1]),
        _event(12, 2, 0.53, 4, etype="pursuit", target_comms=[1]),
        _event(13, 2, 0.55, 3, etype="pursuit", target_comms=[1]),
    ]
    v = classify_regime(events)
    assert v["regime"] == "concentrating"
    assert any("community 1" in x for x in v["evidence"])


def test_regime_idle():
    events = [_event(10, 2, 0.50, 4)] * 3
    assert classify_regime(events)["regime"] == "idle"


def test_render_trajectory_contains_table_and_regime():
    events = [
        _event(10, 2, 0.50, 4),
        _event(13, 3, 0.42, 6),
        _event(16, 5, 0.31, 9),
    ]
    out = render_trajectory(events)
    assert "## Regime: fragmenting" in out
    assert "| ts | event |" in out
    assert render_trajectory([]).count("No journaled events") == 1
