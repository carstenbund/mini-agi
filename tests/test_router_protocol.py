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


def test_vector_router_legacy_sparse_search_returns_relevant_motif(smc_with_motifs):
    """Regression: the legacy sparse fallback used to rebuild vocab per call,
    so query and corpus vectors lived in different spaces. Now vocab is
    persisted on the router."""
    router = VectorRouter(None)
    router.rebuild_from_smc(smc_with_motifs)
    hits = router.search_text("justice fairness contexts", top_k=1)
    assert hits, "expected the legacy fallback to return a hit"
    top_motif, top_score = hits[0]
    assert top_motif.id == "m1"
    assert top_score > 0.0


def test_vector_router_legacy_sparse_handles_query_with_unknown_tokens(smc_with_motifs):
    """A query containing tokens not in the corpus vocab must not corrupt the
    corpus matrix (update_vocab=False at query time) and must still return a
    reasonable hit on the known tokens."""
    router = VectorRouter(None)
    router.rebuild_from_smc(smc_with_motifs)
    corpus_dim = router.mat.shape[1]
    hits = router.search_text("justice xyzzy_unknown_token plugh", top_k=1)
    assert router.mat.shape[1] == corpus_dim, "corpus matrix was mutated by a query"
    assert hits and hits[0][0].id == "m1"


def test_vector_router_legacy_sparse_add_motif_with_new_vocab(smc_with_motifs):
    """Adding a motif whose content introduces new tokens must keep the
    corpus matrix and existing motifs searchable."""
    router = VectorRouter(None)
    router.rebuild_from_smc(smc_with_motifs)
    new_motif = MotifNode(
        id="m4",
        symbols=["epistemic"],
        content="Epistemic gravity bends meaning toward attractors",
        thread_id="test",
    )
    smc_with_motifs.add_motif(new_motif)
    router.add_motif(smc_with_motifs, new_motif)

    assert router.mat.shape[0] == 4
    assert "epistemic" in router._voc
    new_hits = router.search_text("Epistemic gravity bends meaning toward attractors", top_k=1)
    assert new_hits and new_hits[0][0].id == "m4"
    old_hits = router.search_text("Justice as fairness across contexts", top_k=1)
    assert old_hits and old_hits[0][0].id == "m1"
