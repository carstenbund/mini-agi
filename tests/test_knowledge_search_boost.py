from __future__ import annotations

from typing import List, Optional, Tuple

import pytest

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.documents.indexer import DocumentIndexer
from symbolic_recursion.documents.loader import load_raw_text
from symbolic_recursion.documents.schema import DocumentChunk
from symbolic_recursion.skills.knowledge_search import knowledge_search

from tests.conftest import FakeRouter


class StubIndexer:
    """Returns hand-controlled chunk hits, isolating boost math from cosine noise."""

    def __init__(self, hits: List[Tuple[DocumentChunk, float]]):
        self._hits = hits

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        return list(self._hits[:top_k])


def _chunk(cid: str, tags, content: Optional[str] = None) -> DocumentChunk:
    return DocumentChunk(
        id=cid,
        document_id="doc1",
        title="t",
        source="s",
        content=content or cid,
        chunk_index=0,
        tags=list(tags),
    )


@pytest.fixture
def smc_with_motifs() -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    smc.add_motif(
        MotifNode(id="m1", symbols=["Justice", "Gradient"], content="x", thread_id="t")
    )
    smc.add_motif(
        MotifNode(id="m2", symbols=["Recursion"], content="y", thread_id="t")
    )
    return smc


def test_no_active_motifs_means_no_boost(smc_with_motifs):
    chunk = _chunk("c1", tags=["justice"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything", smc_with_motifs, router, document_indexer=indexer, top_k=5
    )

    assert len(results) == 1
    assert results[0].kind == "document_chunk"
    assert results[0].score == pytest.approx(0.5)
    assert results[0].metadata["motif_overlap"] == 0.0


def test_overlap_increases_chunk_score(smc_with_motifs):
    chunk = _chunk("c1", tags=["justice"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.2,
        top_k=5,
    )

    assert len(results) == 1
    # m1 has symbols {justice, gradient}; chunk tags {justice}
    # overlap = |{justice}| / |chunk_tags| = 1 / 1 = 1.0
    # boosted = 0.5 + 0.2 * 1.0 = 0.7
    assert results[0].score == pytest.approx(0.7)
    assert results[0].metadata["motif_overlap"] == pytest.approx(1.0)


def test_partial_overlap_partial_boost(smc_with_motifs):
    chunk = _chunk("c1", tags=["justice", "ethics"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.2,
        top_k=5,
    )

    # overlap = |{justice}| / |{justice, ethics}| = 0.5
    # boosted = 0.5 + 0.2 * 0.5 = 0.6
    assert results[0].score == pytest.approx(0.6)
    assert results[0].metadata["motif_overlap"] == pytest.approx(0.5)


def test_boost_weight_zero_disables_boost(smc_with_motifs):
    chunk = _chunk("c1", tags=["justice"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.0,
        top_k=5,
    )

    assert results[0].score == pytest.approx(0.5)
    # overlap is still reported even when weight is 0
    assert results[0].metadata["motif_overlap"] == pytest.approx(1.0)


def test_unknown_motif_id_is_ignored(smc_with_motifs):
    chunk = _chunk("c1", tags=["justice"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["does-not-exist"],
        motif_boost_weight=0.5,
        top_k=5,
    )

    assert results[0].score == pytest.approx(0.5)
    assert results[0].metadata["motif_overlap"] == 0.0


def test_boost_can_change_ranking(smc_with_motifs):
    """A lower-scored chunk that overlaps with active motifs should outrank a
    higher-scored chunk that doesn't, given a sufficient boost weight."""
    relevant = _chunk("c-relevant", tags=["justice"])
    irrelevant = _chunk("c-irrelevant", tags=["unrelated"])
    indexer = StubIndexer([(irrelevant, 0.6), (relevant, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.3,
        top_k=5,
    )

    ids = [r.id for r in results]
    assert ids[0] == "c-relevant"
    assert ids[1] == "c-irrelevant"


def test_motif_results_unaffected_by_boost(smc_with_motifs):
    motif = smc_with_motifs.get_motif("m1")
    router = FakeRouter(hits=[(motif, 0.42)])
    indexer = StubIndexer([])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.5,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].kind == "motif"
    assert results[0].score == pytest.approx(0.42)
    assert "motif_overlap" not in results[0].metadata


def test_top_k_truncates_combined_results(smc_with_motifs):
    motif = smc_with_motifs.get_motif("m1")
    router = FakeRouter(hits=[(motif, 0.9)])
    indexer = StubIndexer([(_chunk("c1", tags=[]), 0.8), (_chunk("c2", tags=[]), 0.7)])

    results = knowledge_search(
        "anything",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        top_k=2,
    )

    assert len(results) == 2
    assert [r.id for r in results] == ["m1", "c1"]


def test_no_document_indexer_returns_only_motifs(smc_with_motifs):
    motif = smc_with_motifs.get_motif("m1")
    router = FakeRouter(hits=[(motif, 0.9)])

    results = knowledge_search("q", smc_with_motifs, router, document_indexer=None)

    assert [r.kind for r in results] == ["motif"]


def test_case_insensitive_overlap(smc_with_motifs):
    """Motif symbols are 'Justice'/'Gradient' (mixed case); chunk tag is 'JUSTICE'."""
    chunk = _chunk("c1", tags=["JUSTICE"])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "q",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.2,
    )

    assert results[0].metadata["motif_overlap"] == pytest.approx(1.0)
    assert results[0].score == pytest.approx(0.7)


def test_empty_chunk_tags_yields_zero_overlap(smc_with_motifs):
    chunk = _chunk("c1", tags=[])
    indexer = StubIndexer([(chunk, 0.5)])
    router = FakeRouter(hits=[])

    results = knowledge_search(
        "q",
        smc_with_motifs,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.5,
    )

    assert results[0].metadata["motif_overlap"] == 0.0
    assert results[0].score == pytest.approx(0.5)


def test_real_document_indexer_with_boost():
    """End-to-end smoke through the real DocumentIndexer + knowledge_search."""
    smc = SymbolicMemoryCore()
    smc.add_motif(
        MotifNode(id="m1", symbols=["justice"], content="justice motif", thread_id="t")
    )

    doc = load_raw_text(
        content="justice as fairness across contexts is a recursive idea",
        source="mem",
        title="d",
        document_id="d1",
    )
    indexer = DocumentIndexer()
    indexer.index_document(doc, tags=["justice"])

    router = FakeRouter(hits=[])
    boosted = knowledge_search(
        "justice",
        smc,
        router,
        document_indexer=indexer,
        active_motif_ids=["m1"],
        motif_boost_weight=0.5,
    )
    plain = knowledge_search(
        "justice",
        smc,
        router,
        document_indexer=indexer,
        active_motif_ids=None,
        motif_boost_weight=0.5,
    )

    assert boosted and plain
    assert boosted[0].score > plain[0].score
    assert boosted[0].metadata["motif_overlap"] > 0.0
