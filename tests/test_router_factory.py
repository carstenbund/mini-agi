from __future__ import annotations

import builtins
import sys

import pytest

from symbolic_recursion.core.router_factory import make_router
from symbolic_recursion.core.vector_router import VectorRouter


def test_default_returns_vector_router(monkeypatch):
    monkeypatch.delenv("SMC_ROUTER", raising=False)
    router = make_router()
    assert isinstance(router, VectorRouter)


def test_explicit_vector_kind_returns_vector_router(monkeypatch):
    monkeypatch.setenv("SMC_ROUTER", "chroma")
    router = make_router("vector")
    assert isinstance(router, VectorRouter)


def test_env_var_selects_router(monkeypatch):
    monkeypatch.setenv("SMC_ROUTER", "vector")
    assert isinstance(make_router(), VectorRouter)


def test_unknown_kind_falls_back_to_vector(monkeypatch):
    monkeypatch.delenv("SMC_ROUTER", raising=False)
    router = make_router("nonexistent")
    assert isinstance(router, VectorRouter)


def test_chroma_kind_raises_when_chromadb_missing(monkeypatch):
    """The factory should surface a clean RuntimeError when the optional
    `chromadb` dependency is unavailable, not a bare ImportError at import-time."""
    sys.modules.pop("symbolic_recursion.core.chroma_router", None)
    sys.modules.pop("chromadb", None)
    sys.modules.pop("chromadb.config", None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("simulated missing chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="chromadb"):
        make_router("chroma")


def test_returned_router_satisfies_protocol_shape():
    router = make_router("vector")
    for attr in ("rebuild_from_smc", "search_text", "add_motif", "persist"):
        assert callable(getattr(router, attr)), f"missing {attr}"
