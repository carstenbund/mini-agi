from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    id: str
    document_id: str
    title: str
    source: str
    content: str
    chunk_index: int
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SearchResult:
    kind: str  # "motif" | "document_chunk"
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
