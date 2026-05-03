from __future__ import annotations

import pytest

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.vector_router import VectorRouter


def test_vector_router_persist_is_noop(stub_embeddings):
    router = VectorRouter(stub_embeddings)
    assert router.persist() is None


def test_vector_router_empty_smc_returns_no_hits(stub_embeddings):
    router = VectorRouter(stub_embeddings)
    smc = SymbolicMemoryCore()
    router.rebuild_from_smc(smc)
    assert router.search_text("anything", top_k=3) == []


def test_vector_router_search_finds_relevant_motif(smc_with_motifs, stub_embeddings):
    router = VectorRouter(stub_embeddings)
    router.rebuild_from_smc(smc_with_motifs)
    hits = router.search_text("recursive symbolic structures bind across threads", top_k=3)
    assert hits, "expected at least one hit"
    top_motif, top_score = hits[0]
    assert top_motif.id == "m2"
    assert top_score > 0.0


def test_vector_router_top_k_caps_results(smc_with_motifs, stub_embeddings):
    router = VectorRouter(stub_embeddings)
    router.rebuild_from_smc(smc_with_motifs)
    hits = router.search_text("justice", top_k=1)
    assert len(hits) <= 1


def test_vector_router_add_motif_makes_it_searchable(smc_with_motifs, stub_embeddings):
    router = VectorRouter(stub_embeddings)
    router.rebuild_from_smc(smc_with_motifs)
    new_motif = MotifNode(
        id="m4",
        symbols=["epistemic", "gravity"],
        content="Epistemic gravity bends meaning toward attractors",
        thread_id="test",
    )
    smc_with_motifs.add_motif(new_motif)
    router.add_motif(smc_with_motifs, new_motif)
    hits = router.search_text("Epistemic gravity bends meaning toward attractors", top_k=3)
    assert any(m.id == "m4" for m, _ in hits)


@pytest.mark.xfail(
    reason=(
        "Bug: VectorRouter's legacy sparse fallback (no Embeddings) builds the query "
        "vocabulary fresh from the query text only, so the query vector and the "
        "corpus matrix end up in different vector spaces and search_text fails with "
        "a numpy concatenation/shape error. The vocab needs to be persisted on the "
        "router or projected through a fixed dimension."
    ),
    raises=Exception,
    strict=True,
)
def test_vector_router_legacy_sparse_fallback_is_broken(smc_with_motifs):
    router = VectorRouter(None)
    router.rebuild_from_smc(smc_with_motifs)
    router.search_text("justice", top_k=1)
