"""Agent identity — which session made a move.

Several sessions (people, Claude sessions, loop runs) may work the same
field at once. Nothing else in the pipeline knows who did what, so every
record that matters carries the agent id: motifs on capture, trajectory
journal lines, flow ledger entries, pursuit claims and review verdicts.

Set ``SMC_AGENT`` per session (``SMC_AGENT=carsten``,
``SMC_AGENT=loop-nightly``). Unset, everything is ``anonymous`` — the
pipeline still works, it just cannot tell sessions apart.
"""
from __future__ import annotations

import os

DEFAULT_AGENT = "anonymous"


def agent_id() -> str:
    """The current session's agent id (``SMC_AGENT``, else ``anonymous``)."""
    return (os.environ.get("SMC_AGENT") or DEFAULT_AGENT).strip() or DEFAULT_AGENT
