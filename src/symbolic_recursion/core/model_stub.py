"""Deterministic model stub.

This module supplies :func:`stub_response` – a tiny, reproducible
stand‑in for a large language model.  It is useful for unit tests or
offline experimentation where calling a real model would be costly or
impossible.

Example
-------
>>> from symbolic_recursion.core.model_stub import stub_response
>>> stub_response("Hello world", model="demo")
'[demo] Hello world'

The optional ``timeout`` argument is ignored and is present only so that
``stub_response`` matches the signature of
:func:`symbolic_recursion.core.ollama_interface.query_ollama`.
"""

from typing import Optional


def stub_response(prompt: str, model: str, timeout: Optional[int] = None) -> str:
    """Return a reproducible response for ``prompt`` and ``model``.

    Parameters
    ----------
    prompt:
        Text prompt being "asked" to the stubbed model.
    model:
        Name of the model.  Included in the output so different models
        produce distinguishable results.
    timeout:
        Ignored placeholder to mirror the interface of the real model
        querying function.

    Returns
    -------
    str
        A deterministic string based solely on ``model`` and ``prompt``.
    """
    return f"[{model}] {prompt}"
