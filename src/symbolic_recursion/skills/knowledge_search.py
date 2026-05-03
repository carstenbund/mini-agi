from __future__ import annotations

from typing import List, Optional

from symbolic_recursion.core.chroma_router import ChromaRouter
from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.documents.indexer import DocumentIndexer
from symbolic_recursion.documents.schema import SearchResult


def knowledge_search(
    query: str,
    smc: SymbolicMemoryCore,
    motif_router: ChromaRouter,
    document_indexer: Optional[DocumentIndexer] = None,
    top_k: int = 8,
    motif_k: Optional[int] = None,
    document_k: Optional[int] = None,
) -> List[SearchResult]:
    """Unified retrieval across motifs + document chunks.

    Returns ranked `SearchResult` items with a stable shape suitable for skill
    wrappers (e.g., OpenClaw-style documentation search endpoints).
    """
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
        for chunk, score in doc_hits:
            results.append(
                SearchResult(
                    kind="document_chunk",
                    id=chunk.id,
                    content=chunk.content,
                    score=score,
                    metadata={
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "tags": chunk.tags,
                        "updated_at": chunk.updated_at,
                    },
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
