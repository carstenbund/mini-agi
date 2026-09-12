"""Document intake — a structured text becomes a motif subgraph.

Automates the hand-capture pattern: decompose by meaning-bearing
structure (markdown headings), keep section text verbatim, tag honestly
from the document's own vocabulary, wire references by structure:

- a HUB motif holds the title and any preamble; every section references
  it (the document's axis)
- each section also references its predecessor (the reading sequence)
- ``[[wiki-links]]`` resolve to the hub motifs of previously captured
  documents via a small registry (``data/doc_registry.json``) — declared
  cross-document ancestry becomes explicit edges, exactly the effect
  that raises narrative binding on arrival

Everything is deterministic and stdlib-only. Unstructured text (no
headings) falls back to a single motif. Symbol derivation is honest by
construction: only frontmatter tags that actually appear in a section's
text, plus the section's own heading words.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from symbolic_recursion.core.agent import agent_id
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore

_STOPWORDS = frozenset(
    "a an and as at be but by for from in into is it its of on or not no the "
    "to with that this those these we you they he she i what which how when "
    "why where who whom whose own same so than then there their them our".split()
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def registry_path() -> str:
    return os.path.abspath(
        os.environ.get("SMC_DOC_REGISTRY", os.path.join("data", "doc_registry.json"))
    )


def _load_registry() -> Dict[str, str]:
    path = registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(reg: Dict[str, str]) -> None:
    path = registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def _norm_title(title: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", title.lower(), re.UNICODE))


def _slug(text: str, max_len: int = 24) -> str:
    words = [w.lower() for w in re.findall(r"[^\W_]+", text, re.UNICODE)
             if w.lower() not in _STOPWORDS]
    slug = "-".join(words) or "section"
    return slug[:max_len].rstrip("-")


def parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """Parse a leading ``---`` block of ``key: value`` lines (YAML-lite,
    no dependency): scalars, quoted strings, and ``[a, b]`` lists."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, object] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:])
        mm = re.match(r"^(\w[\w-]*):\s*(.*)$", line.strip())
        if not mm:
            continue
        key, val = mm.group(1), mm.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        else:
            meta[key] = val.strip("'\"")
    return {}, text  # no closing --- : treat as body


def split_sections(body: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split on markdown ATX headings. Returns (preamble, [(heading, text)]).
    A body without headings yields an empty section list."""
    preamble_lines: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble_lines.append(line)
    return (
        "\n".join(preamble_lines).strip(),
        [(h, "\n".join(ls).strip()) for h, ls in sections if "\n".join(ls).strip() or h],
    )


def derive_symbols(tags: List[str], heading: str, text: str, max_n: int = 5) -> List[str]:
    """Honest tagging: frontmatter tags that actually occur in this
    section's text, then the heading's own significant words."""
    lowered = f"{heading}\n{text}".lower()
    out: List[str] = []
    for t in tags:
        tl = str(t).strip().lower()
        if tl and tl in lowered and tl not in out:
            out.append(tl)
    for w in re.findall(r"[^\W_]+", heading.lower(), re.UNICODE):
        if w not in _STOPWORDS and not w.isdigit() and len(w) > 2 and w not in out:
            out.append(w)
    return out[:max_n]


def resolve_wikilinks(text: str, registry: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Resolve ``[[Doc Title]]`` links against captured documents.
    Returns (resolved hub motif ids, unresolved link texts)."""
    resolved: List[str] = []
    unresolved: List[str] = []
    for link in _WIKILINK_RE.findall(text):
        hub = registry.get(_norm_title(link))
        if hub and hub not in resolved:
            resolved.append(hub)
        elif not hub and link.strip() not in unresolved:
            unresolved.append(link.strip())
    return resolved, unresolved


def plan_capture(
    smc: SymbolicMemoryCore,
    text: str,
    title: Optional[str] = None,
    thread: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Dict:
    """Deterministic capture plan — everything but the writes.

    Returns {"title", "thread", "motifs": [{id, symbols, content, references,
    unresolved_links}], "registered_as"}.
    """
    meta, body = parse_frontmatter(text)
    preamble, sections = split_sections(body)

    if title is None:
        title = str(meta.get("topic") or "") or (sections[0][0] if sections else "untitled")
    thread = thread or _slug(str(meta.get("topic") or title), 40)
    prefix = prefix or _slug(title, 12)
    tags = [str(t) for t in meta.get("tags", [])] if isinstance(meta.get("tags"), list) else []
    registry = _load_registry()

    hub_id = f"{prefix}-hub"
    hub_text = f"{title}\n\n{preamble}".strip() if preamble else title
    hub_links, hub_unresolved = resolve_wikilinks(hub_text, registry)
    motifs = [{
        "id": hub_id,
        "symbols": derive_symbols(tags, title, hub_text or title),
        "content": hub_text,
        "references": hub_links,
        "unresolved_links": hub_unresolved,
    }]

    prev_id = None
    seen = {hub_id}
    for heading, sec_text in sections:
        sid = f"{prefix}-{_slug(heading)}"
        while sid in seen:
            sid += "-x"
        seen.add(sid)
        links, unresolved = resolve_wikilinks(sec_text, registry)
        refs = [hub_id] + ([prev_id] if prev_id else []) + [l for l in links if l not in (hub_id,)]
        motifs.append({
            "id": sid,
            "symbols": derive_symbols(tags, heading, sec_text),
            "content": f"{heading}. {sec_text}".strip() if sec_text else heading,
            "references": refs,
            "unresolved_links": unresolved,
        })
        prev_id = sid

    return {"title": title, "thread": thread, "motifs": motifs,
            "registered_as": _norm_title(title)}


def capture_document(
    smc: SymbolicMemoryCore,
    text: str,
    title: Optional[str] = None,
    thread: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Dict:
    """Execute a capture plan: add the motifs, register the document's hub
    for future wikilink resolution. Returns the plan with motifs added."""
    plan = plan_capture(smc, text, title=title, thread=thread, prefix=prefix)
    for spec in plan["motifs"]:
        smc.add_motif(MotifNode(
            id=spec["id"], symbols=spec["symbols"], content=spec["content"],
            thread_id=plan["thread"],
            references=[r for r in spec["references"] if smc.get_motif(r) is not None],
            agent=agent_id(),
        ))
    registry = _load_registry()
    registry[plan["registered_as"]] = plan["motifs"][0]["id"]
    _save_registry(registry)
    return plan
