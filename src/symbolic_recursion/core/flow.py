"""Flow ledger — the textual record of every generation, for debugging.

The trajectory journal records metrics; this ledger records TEXT: the
full prompt a generation was shown (context block included), the model's
response, and the provenance (thread, model, targets, resulting motif).
Append-only JSONL at ``data/flow.jsonl`` (``SMC_FLOW_PATH`` overrides),
same convention as the motif store and the trajectory journal.

``render_trace`` reconstructs the flow of text around one motif: what it
was shown, what it said, where it came from, and what grew out of it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore


def flow_path() -> str:
    return os.path.abspath(
        os.environ.get("SMC_FLOW_PATH", os.path.join("data", "flow.jsonl"))
    )


def record_flow(entry: Dict, path: Optional[str] = None,
                now: Optional[datetime] = None) -> Dict:
    """Append one generation record. Expected keys: ``kind`` (pursuit kind,
    "chat", "cycle-seed", ...), ``thread``, ``model``, ``motif_id``,
    ``targets`` (context/link ids), ``prompt``, ``response``."""
    path = path or flow_path()
    line = {"ts": (now or datetime.utcnow()).isoformat(), **entry}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


def load_flow(path: Optional[str] = None) -> List[Dict]:
    path = path or flow_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def _excerpt(text: str, n: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_trace(
    smc: SymbolicMemoryCore,
    motif_id: str,
    flow_events: Optional[List[Dict]] = None,
    full: bool = True,
) -> str:
    """The flow of text around one motif, as readable markdown.

    Shows the motif's content, its parents (what it grew from), its
    generation record (the exact prompt shown and response given) when
    the flow ledger has one, and its children (what grew from it).
    ``full=False`` excerpts the prompt/response instead of printing them
    whole."""
    m = smc.get_motif(motif_id)
    if m is None:
        return f"No motif with id {motif_id!r} in the field.\n"
    if flow_events is None:
        flow_events = load_flow()

    lines = [f"# Trace — `{m.id}`", "",
             f"thread `{m.thread_id}` · symbols [{', '.join(m.symbols)}]", ""]

    lines += ["## Content", "", m.content.strip(), ""]

    if m.references:
        lines.append("## Grew from (references)")
        for rid in m.references:
            r = smc.get_motif(rid)
            if r:
                lines.append(f"- `{rid}` [{', '.join(r.symbols)}] ({r.thread_id}): {_excerpt(r.content)}")
            else:
                lines.append(f"- `{rid}` (not in field)")
        lines.append("")

    gen = [e for e in flow_events if e.get("motif_id") == motif_id]
    if gen:
        e = gen[-1]
        lines += [f"## Generation record ({e.get('kind', '?')} · model {e.get('model', '?')} · {e.get('ts', '')[:16]})", ""]
        prompt = e.get("prompt", "")
        response = e.get("response", "")
        if full:
            lines += ["### Prompt shown to the model", "", "```", prompt, "```", "",
                      "### Response", "", "```", response, "```", ""]
        else:
            lines += [f"prompt: {_excerpt(prompt, 400)}", "",
                      f"response: {_excerpt(response, 400)}", ""]
    else:
        lines += ["## Generation record", "",
                  "None in the flow ledger — captured before the ledger "
                  "existed, or added manually.", ""]

    children = [c for c in smc.list_motifs() if motif_id in c.references]
    if children:
        lines.append("## Grew into (referenced by)")
        for c in sorted(children, key=lambda x: x.created_at or ""):
            lines.append(f"- `{c.id}` [{', '.join(c.symbols)}] ({c.thread_id}): {_excerpt(c.content)}")
        lines.append("")

    return "\n".join(lines) + "\n"
