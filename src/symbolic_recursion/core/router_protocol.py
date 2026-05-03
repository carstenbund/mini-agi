from __future__ import annotations

from typing import List, Protocol, Tuple

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore


class RouterProtocol(Protocol):
    """Minimal retrieval/index interface shared across router backends."""

    def rebuild_from_smc(self, smc: SymbolicMemoryCore) -> None: ...

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]: ...

    def add_motif(self, smc: SymbolicMemoryCore, m: MotifNode) -> None: ...

    def persist(self) -> None: ...
