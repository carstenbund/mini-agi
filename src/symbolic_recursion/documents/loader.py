from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LoadedDocument:
    document_id: str
    title: str
    source: str
    content: str


def load_text_file(path: str, title: Optional[str] = None, document_id: Optional[str] = None) -> LoadedDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    doc_id = document_id or p.stem
    doc_title = title or p.name
    return LoadedDocument(document_id=doc_id, title=doc_title, source=str(p), content=text)


def load_raw_text(content: str, source: str, title: str, document_id: str) -> LoadedDocument:
    return LoadedDocument(document_id=document_id, title=title, source=source, content=content)
