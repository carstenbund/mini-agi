from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np
import pytest

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore


def _motif(mid: str, symbols, content: str) -> MotifNode:
    return MotifNode(
        id=mid,
        symbols=list(symbols),
        content=content,
        thread_id="test",
    )


@pytest.fixture
def smc_with_motifs() -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("m1", ["justice", "gradient"], "Justice as fairness across contexts"))
    smc.add_motif(_motif("m2", ["recursion", "symbol"], "Recursive symbolic structures bind across threads"))
    smc.add_motif(_motif("m3", ["coherence"], "Coherence emerges from repeated motif activation"))
    return smc


class FakeRouter:
    """Hand-controlled router used in skill tests; no embedding noise."""

    def __init__(self, hits: Optional[List[Tuple[MotifNode, float]]] = None):
        self._hits = list(hits or [])
        self.search_calls: List[Tuple[str, int]] = []
        self.persist_calls = 0
        self.added: List[str] = []

    def rebuild_from_smc(self, smc: SymbolicMemoryCore) -> None:
        return None

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]:
        self.search_calls.append((text, top_k))
        return list(self._hits[:top_k])

    def add_motif(self, smc: SymbolicMemoryCore, m: MotifNode) -> None:
        self.added.append(m.id)

    def persist(self) -> None:
        self.persist_calls += 1


class StubEmbeddings:
    """Deterministic, offline embedder satisfying the SBERT `Embeddings` duck type.

    Hashes tokens into a fixed-width vector so encoding is consistent across calls
    (unlike the legacy bag-of-words fallback, which builds vocab per call)."""

    DIM = 64

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or "").lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        out /= norms
        return out

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]


@pytest.fixture
def stub_embeddings() -> StubEmbeddings:
    return StubEmbeddings()
