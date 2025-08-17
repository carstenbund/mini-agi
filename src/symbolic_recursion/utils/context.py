from __future__ import annotations
from typing import List, Tuple
from symbolic_recursion.core.motif import MotifNode

def render_context(motifs: List[Tuple[MotifNode, float]], max_chars: int = 2000) -> str:
    """
    Turn retrieved motifs into a compact, model-ready context block.
    motifs: list of (MotifNode, similarity_score in [0,1])
    """
    header = "## Context (top motifs)\n"
    lines, used = [], 0
    for m, score in motifs:
        chunk = f"\n### {','.join(m.symbols)} (sim={score:.2f})\n{m.content.strip()}\n"
        if used + len(chunk) > max_chars:
            break
        lines.append(chunk)
        used += len(chunk)
    return header + "".join(lines)

