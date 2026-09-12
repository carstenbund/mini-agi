from __future__ import annotations

from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.documents.capture import (
    capture_document,
    parse_frontmatter,
    plan_capture,
    split_sections,
)

DOC = """---
topic: test-doc
tags: [alpha, beta, missing-tag]
---
Preamble paragraph citing [[Prior Document]].

# First Section
Text about alpha things.

# Second Section
Text about beta and alpha together.
"""

PRIOR = """# Prior Document
Some earlier content about gamma.
"""


def test_frontmatter_and_sections():
    meta, body = parse_frontmatter(DOC)
    assert meta["topic"] == "test-doc"
    assert meta["tags"] == ["alpha", "beta", "missing-tag"]
    preamble, sections = split_sections(body)
    assert "Preamble" in preamble
    assert [h for h, _ in sections] == ["First Section", "Second Section"]


def test_capture_builds_hub_sequence_and_symbols():
    smc = SymbolicMemoryCore()
    plan = capture_document(smc, DOC)
    ids = [m["id"] for m in plan["motifs"]]
    hub, s1, s2 = ids
    assert hub.endswith("-hub")
    assert plan["thread"] == "test-doc"
    # sections reference hub; second also references first (sequence)
    assert smc.get_motif(s1).references == [hub]
    assert smc.get_motif(s2).references == [hub, s1]
    # honest tagging: only frontmatter tags present in the section's text
    assert "alpha" in smc.get_motif(s1).symbols
    assert "missing-tag" not in smc.get_motif(s1).symbols
    assert set(smc.get_motif(s2).symbols) >= {"beta", "alpha"}


def test_wikilinks_resolve_after_prior_capture():
    smc = SymbolicMemoryCore()
    # capturing the cited document first registers its hub
    prior_plan = capture_document(smc, PRIOR, title="Prior Document")
    prior_hub = prior_plan["motifs"][0]["id"]
    plan = capture_document(smc, DOC)
    hub = smc.get_motif(plan["motifs"][0]["id"])
    assert prior_hub in hub.references          # declared ancestry became an edge
    assert plan["motifs"][0]["unresolved_links"] == []


def test_wikilinks_unresolved_without_registry():
    smc = SymbolicMemoryCore()
    plan = plan_capture(smc, DOC)
    assert plan["motifs"][0]["unresolved_links"] == ["Prior Document"]


def test_unstructured_text_falls_back_to_single_motif():
    smc = SymbolicMemoryCore()
    plan = capture_document(smc, "Just a paragraph.\n\nAnother paragraph.",
                            title="Loose Note")
    assert len(plan["motifs"]) == 1
    assert "Just a paragraph." in smc.get_motif(plan["motifs"][0]["id"]).content


def test_plan_is_side_effect_free():
    smc = SymbolicMemoryCore()
    plan_capture(smc, DOC)
    assert smc.list_motifs() == []


def test_relink_pass_wires_links_captured_out_of_order():
    from symbolic_recursion.documents.capture import apply_relink, relink_pass

    smc = SymbolicMemoryCore()
    # DOC cites [[Prior Document]] which is not yet captured
    capture_document(smc, DOC)
    assert relink_pass(smc) == []            # nothing resolvable yet
    prior = capture_document(smc, PRIOR, title="Prior Document")
    prior_hub = prior["motifs"][0]["id"]
    changes = relink_pass(smc)
    assert changes and all(c["add"] == prior_hub for c in changes)
    assert apply_relink(smc, changes) == len(changes)
    hub = smc.get_motif("test-doc-hub")
    assert prior_hub in hub.references
    # idempotent: a second pass finds nothing
    assert relink_pass(smc) == []


def test_relink_respects_thread_filter():
    from symbolic_recursion.documents.capture import relink_pass

    smc = SymbolicMemoryCore()
    capture_document(smc, DOC)
    capture_document(smc, PRIOR, title="Prior Document")
    assert relink_pass(smc, thread="unrelated-thread") == []
