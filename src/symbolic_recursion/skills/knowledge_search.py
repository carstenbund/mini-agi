from __future__ import annotations

from typing import Iterable, List, Optional, Set

from symbolic_recursion.core.router_protocol import RouterProtocol
from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.documents.indexer import DocumentIndexer
from symbolic_recursion.documents.schema import SearchResult


def knowledge_search(
    query: str,
    smc: SymbolicMemoryCore,
    motif_router: RouterProtocol,
    document_indexer: Optional[DocumentIndexer] = None,
    top_k: int = 8,
    motif_k: Optional[int] = None,
    document_k: Optional[int] = None,
    active_motif_ids: Optional[Iterable[str]] = None,
    motif_boost_weight: float = 0.15,
) -> List[SearchResult]:
    mk = motif_k or top_k
    dk = document_k or top_k

    results: List[SearchResult] = []

    motif_hits = motif_router.search_text(query, top_k=mk)
    for motif, score in motif_hits:
        results.append(
            SearchResult(
                kind="motif",
                id=motif.id,
                content=motif.content,
                score=score,
                metadata={
                    "symbols": motif.symbols,
                    "thread_id": motif.thread_id,
                    "references": motif.references,
                    "updated_at": motif.updated_at,
                },
            )
        )

    if document_indexer is not None:
        doc_hits = document_indexer.search(query, top_k=dk)
        boost_symbols: Set[str] = set()
        for mid in active_motif_ids or []:
            motif = smc.get_motif(mid)
            if motif is not None:
                boost_symbols.update(s.lower() for s in motif.symbols)
        for chunk, score in doc_hits:
            overlap = 0.0
            if boost_symbols:
                chunk_symbols = {t.lower() for t in chunk.tags}
                overlap = len(boost_symbols & chunk_symbols) / max(1, len(chunk_symbols))
            boosted_score = score + (motif_boost_weight * overlap)
            results.append(
                SearchResult(
                    kind="document_chunk",
                    id=chunk.id,
                    content=chunk.content,
                    score=boosted_score,
                    metadata={
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "tags": chunk.tags,
                        "updated_at": chunk.updated_at,
                        "motif_overlap": overlap,
                    },
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
