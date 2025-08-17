"""Deterministic stand-in for an LLM backend.

This module provides :func:`stub_response`, a tiny helper that mimics a
language model for development and testing.  The function returns a
reproducible string that depends on both the prompt and model name,
making it suitable for experiments that expect varying outputs without
requiring network access or heavy model downloads.

Example
-------
>>> from symbolic_recursion.core.model_stub import stub_response
>>> stub_response("What is the answer?", "demo-model")
'[stub:demo-model] What is the answer? :: 8e1a9f7c'

The optional ``timeout`` parameter is accepted for API compatibility but
is ignored.
"""

from __future__ import annotations

from typing import Optional
import hashlib

def stub_response(prompt: str, model: str, timeout: Optional[int] = None) -> str:
    """Return a reproducible pseudo-response.

    Args:
        prompt: The prompt text provided to the model.
        model: Identifier for the model being "queried".
        timeout: Included for API compatibility but not used.

    Returns:
        A deterministic string derived from ``prompt`` and ``model``.
    """
    seed = f"{model}:{prompt}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()[:8]
    return f"[stub:{model}] {prompt} :: {digest}"
