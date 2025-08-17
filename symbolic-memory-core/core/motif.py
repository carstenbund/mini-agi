from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Revision:
    timestamp: str
    content: str
    symbols: List[str]

@dataclass
class MotifNode:
    id: str
    symbols: List[str]
    content: str
    thread_id: str
    references: List[str] = field(default_factory=list)
    history: List[Revision] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MotifNode":
        hist = [Revision(**h) for h in d.get("history", [])]
        return MotifNode(
            id=d["id"],
            symbols=list(d.get("symbols", [])),
            content=d.get("content", ""),
            thread_id=d.get("thread_id", ""),
            references=list(d.get("references", [])),
            history=hist,
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )

class SymbolicMemoryCore:
    """
    Minimal in-memory graph for motifs with JSON persistence handled by core.storage.
    Similarity search is delegated to the router layer.
    """
    def __init__(self):
        self.motifs: Dict[str, MotifNode] = {}

    # CRUD
    def add_motif(self, motif: MotifNode) -> None:
        self.motifs[motif.id] = motif

    def get_motif(self, motif_id: str) -> Optional[MotifNode]:
        return self.motifs.get(motif_id)

    def list_motifs(self) -> List[MotifNode]:
        return list(self.motifs.values())

    def update_motif(self, motif_id: str, content: Optional[str] = None, symbols: Optional[List[str]] = None) -> bool:
        m = self.motifs.get(motif_id)
        if not m:
            return False
        # record revision
        m.history.append(Revision(timestamp=datetime.utcnow().isoformat(),
                                  content=m.content,
                                  symbols=m.symbols.copy()))
        if content is not None:
            m.content = content
        if symbols is not None:
            m.symbols = symbols
        m.updated_at = datetime.utcnow().isoformat()
        return True

    # Linking
    def link_motifs(self, a: str, b: str) -> bool:
        if a not in self.motifs or b not in self.motifs:
            return False
        if b not in self.motifs[a].references:
            self.motifs[a].references.append(b)
        return True

    # Simple symbol filter
    def find_by_symbol(self, token: str) -> List[MotifNode]:
        t = token.lower()
        return [m for m in self.motifs.values() if any(t == s.lower() for s in m.symbols)]
