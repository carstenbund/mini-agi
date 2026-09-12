from __future__ import annotations

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.graph import (
    analyze_field,
    build_motif_graph,
    detect_communities,
    render_report,
    surprising_connections,
)


def _motif(mid: str, symbols, thread: str = "test", refs=None) -> MotifNode:
    return MotifNode(
        id=mid,
        symbols=list(symbols),
        content=f"content of {mid}",
        thread_id=thread,
        references=list(refs or []),
    )


def _two_cluster_field() -> SymbolicMemoryCore:
    """Two reference-triangles joined by one bridge edge."""
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a1", ["justice"], "A", refs=["a2", "a3"]))
    smc.add_motif(_motif("a2", ["justice", "fairness"], "A", refs=["a3"]))
    smc.add_motif(_motif("a3", ["fairness"], "A"))
    smc.add_motif(_motif("b1", ["recursion"], "B", refs=["b2", "b3"]))
    smc.add_motif(_motif("b2", ["recursion", "loops"], "B", refs=["b3"]))
    smc.add_motif(_motif("b3", ["loops"], "B"))
    smc.link_motifs("a3", "b1")  # the bridge
    return smc


def test_build_graph_dedupes_and_skips_bad_refs():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("x", ["s"], refs=["y", "y", "x", "ghost"]))
    smc.add_motif(_motif("y", ["t"], refs=["x"]))  # reverse duplicate
    g = build_motif_graph(smc, include_symbol_edges=False)
    assert [(e.a, e.b, e.kind) for e in g.edges] == [("x", "y", "reference")]


def test_symbol_edges_are_weaker_than_references():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("x", ["shared", "extra"]))
    smc.add_motif(_motif("y", ["shared"]))
    g = build_motif_graph(smc)
    assert len(g.edges) == 1
    e = g.edges[0]
    assert e.kind == "shared_symbol"
    assert 0.0 < e.weight < 1.0


def test_communities_split_two_clusters():
    smc = _two_cluster_field()
    g = build_motif_graph(smc, include_symbol_edges=False)
    comm = detect_communities(g)
    assert comm["a1"] == comm["a2"] == comm["a3"]
    assert comm["b1"] == comm["b2"] == comm["b3"]
    assert comm["a1"] != comm["b1"]


def test_communities_are_deterministic():
    smc = _two_cluster_field()
    g = build_motif_graph(smc)
    assert detect_communities(g) == detect_communities(g)


def test_bridge_edge_is_surprising_with_reasons():
    smc = _two_cluster_field()
    g = build_motif_graph(smc, include_symbol_edges=False)
    comm = detect_communities(g)
    surprises = surprising_connections(g, comm, smc)
    assert surprises, "the bridge should surface"
    top = surprises[0]
    assert {top["a"], top["b"]} == {"a3", "b1"}
    assert any("cross-thread" in r for r in top["reasons"])


def test_god_motif_is_the_hub():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("hub", ["h"]))
    for i in range(4):
        smc.add_motif(_motif(f"leaf{i}", [f"l{i}"], refs=["hub"]))
    analysis = analyze_field(smc, include_symbol_edges=False)
    assert analysis["god_motifs"][0]["id"] == "hub"
    assert analysis["god_motifs"][0]["degree"] == 4


def test_isolated_motifs_get_own_communities():
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("solo1", ["one"]))
    smc.add_motif(_motif("solo2", ["two"]))
    g = build_motif_graph(smc)
    comm = detect_communities(g)
    assert comm["solo1"] != comm["solo2"]


def test_metrics_track_history_and_binding():
    smc = _two_cluster_field()
    smc.update_motif("a1", symbols=["justice", "gradient"])  # one revision
    analysis = analyze_field(smc, include_symbol_edges=False)
    m = analysis["metrics"]
    assert m["motif_count"] == 6
    assert m["reference_edges"] == 7
    assert m["symbolic_depth"] > 0.0
    assert 0.0 < m["narrative_binding"] <= 1.0  # the bridge is cross-thread
    assert m["symbol_drift"] > 0.0


def test_report_renders_on_empty_and_full_field():
    empty = render_report(SymbolicMemoryCore())
    assert "field is empty" in empty
    report = render_report(_two_cluster_field())
    for section in ("## Field", "## God motifs", "## Communities", "## Surprising connections"):
        assert section in report
