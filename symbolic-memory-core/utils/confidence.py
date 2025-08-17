from typing import Dict, List
import re
from embeddings.embedder import embed_text, cosine_sparse
from core.motif import SymbolicMemoryCore

HEDGE_RE = re.compile(r"\b(might|maybe|perhaps|possibly|could|unclear|not sure|i think|it seems|appears)\b", re.I)

def _text_quality(text: str) -> float:
    """
    0..1 where higher ~ clearer, less hedging/repetition, some structure.
    Simple, fast, deterministic.
    """
    if not text or not text.strip():
        return 0.0
    t = text.strip()

    # Hedging penalty
    hedges = len(HEDGE_RE.findall(t))
    hedge_pen = min(1.0, hedges / 8.0)  # cap

    # Repetition penalty: repeated 3+ word shingles
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)]
    rep_pen = 0.0
    if len(words) >= 9:
        shingles = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        from collections import Counter
        c = Counter(shingles)
        repeats = sum(v-1 for v in c.values() if v > 1)
        rep_pen = min(1.0, repeats / max(1, len(shingles)//5))

    # Structure bonus: presence of bullets / numbering / headings
    structure = 0.0
    if any(line.strip().startswith(("-", "*", "•")) for line in t.splitlines()):
        structure += 0.3
    if re.search(r"^\s*\d+\.", t, re.M):
        structure += 0.3
    if len(t.splitlines()) >= 3:
        structure += 0.2
    structure = min(0.7, structure)

    # Length reasonableness: very short or extremely long is suspicious
    ln = len(words)
    if ln < 30:
        len_pen = 0.4
    elif ln > 1200:
        len_pen = 0.3
    else:
        len_pen = 0.0

    # Base quality
    q = 1.0 - (0.5*hedge_pen + 0.4*rep_pen + 0.3*len_pen)
    q = max(0.0, min(1.0, q + structure))
    return q

def _context_density(smc: SymbolicMemoryCore, text: str, sim_k: int = 5) -> float:
    """
    0..1 indicating how well the text anchors in existing memory.
    Uses max cosine similarity to any motif and count of near neighbors.
    """
    vec = embed_text(text)
    if not smc.motifs or not vec:
        return 0.0
    sims: List[float] = []
    for m in smc.list_motifs():
        sims.append(cosine_sparse(vec, embed_text(m.content)))
    sims.sort(reverse=True)
    max_sim = sims[0] if sims else 0.0
    near = [s for s in sims[:sim_k] if s >= 0.35]
    density = 0.6*max_sim + 0.4*(len(near)/float(sim_k))
    return max(0.0, min(1.0, density))

def _self_rating_fn(prompt_text: str) -> float:
    """
    Placeholder for a model self-rating step (0..1).
    Keep stub-safe by returning a neutral value; you can replace this by
    calling your LLM with a rating prompt and parsing a number.
    """
    return 0.5

def confidence_score(
    smc: SymbolicMemoryCore,
    generated_text: str,
    use_self_rating: bool = False,
    weights = {"text":0.45, "context":0.45, "self":0.10},
) -> Dict[str, float]:
    tq = _text_quality(generated_text)
    cd = _context_density(smc, generated_text)
    sr = _self_rating_fn(generated_text) if use_self_rating else 0.5  # neutral when off
    score = weights["text"]*tq + weights["context"]*cd + weights["self"]*sr
    score = max(0.0, min(1.0, score))
    return {"text_quality": tq, "context_density": cd, "self_rating": sr, "confidence": score}
