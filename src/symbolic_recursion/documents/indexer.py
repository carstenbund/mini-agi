from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

from symbolic_recursion.documents.chunker import chunk_text
from symbolic_recursion.documents.loader import LoadedDocument
from symbolic_recursion.documents.schema import DocumentChunk
from symbolic_recursion.embeddings.embedder import cosine_sparse, embed_text


class DocumentIndexer:
    """In-memory chunk index for document retrieval.

    This mirrors the current motif search approach (sparse term vectors) to keep
    dependencies small while providing a clear API boundary for future Chroma/
    hybrid backends.
    """

    def __init__(self) -> None:
        self._chunks: Dict[str, DocumentChunk] = {}

    def list_chunks(self) -> List[DocumentChunk]:
        return list(self._chunks.values())

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks.get(chunk_id)

    def index_document(
        self,
        doc: LoadedDocument,
        tags: Optional[Iterable[str]] = None,
        chunk_size: int = 900,
        overlap: int = 120,
    ) -> List[DocumentChunk]:
        tags_list = list(tags or [])
        parts = chunk_text(doc.content, chunk_size=chunk_size, overlap=overlap)
        created: List[DocumentChunk] = []
        for idx, part in enumerate(parts):
            cid = f"{doc.document_id}:{idx}"
            chunk = DocumentChunk(
                id=cid,
                document_id=doc.document_id,
                title=doc.title,
                source=doc.source,
                content=part,
                chunk_index=idx,
                tags=tags_list,
            )
            self._chunks[cid] = chunk
            created.append(chunk)
        return created

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        qv = embed_text(query)
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in self._chunks.values():
            score = cosine_sparse(qv, embed_text(chunk.content))
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def as_dict(self) -> List[Dict[str, object]]:
        return [asdict(chunk) for chunk in self._chunks.values()]
