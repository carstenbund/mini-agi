from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Split text into overlapping character chunks.

    This intentionally keeps implementation simple and deterministic so it can
    be upgraded later (sentence-aware, token-aware, markdown-aware, etc.).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    out: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        part = text[start : start + chunk_size].strip()
        if part:
            out.append(part)
        if start + chunk_size >= len(text):
            break
    return out
