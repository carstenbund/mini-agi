from __future__ import annotations

import os

from symbolic_recursion.core.router_protocol import RouterProtocol
from symbolic_recursion.core.vector_router import VectorRouter


def make_router(kind: str | None = None) -> RouterProtocol:
    router_kind = (kind or os.getenv("SMC_ROUTER", "vector")).lower()
    if router_kind == "chroma":
        from symbolic_recursion.core.chroma_router import ChromaRouter

        return ChromaRouter()
    return VectorRouter(None)
