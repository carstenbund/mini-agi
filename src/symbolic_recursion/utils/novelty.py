# utils/novelty.py
from typing import Dict, List, Set
from symbolic_recursion.embeddings.embedder import embed_text, cosine_sparse
from symbolic_recursion.core.motif import SymbolicMemoryCore, MotifNode

def _others(smc: SymbolicMemoryCore, m: MotifNode) -> List[MotifNode]:
    """Prior corpus for scoring m: every motif in smc except m itself.

    Callers typically score a motif *after* it has been captured into the
    graph, so without this exclusion m matches itself with cosine 1.0 and
    every novelty term collapses to zero.
    """
    return [x for x in smc.list_motifs() if x.id != m.id]

def _all_vectors(smc: SymbolicMemoryCore, exclude: MotifNode = None) -> List[Dict[str, float]]:
    motifs = _others(smc, exclude) if exclude is not None else smc.list_motifs()
    return [embed_text(x.content) for x in motifs]

def _max_cosine(vec: Dict[str, float], corpus: List[Dict[str, float]]) -> float:
    if not corpus:
        return 0.0
    best = 0.0
    for v in corpus:
        c = cosine_sparse(vec, v)
        if c > best:
            best = c
    return best

def semantic_novelty(smc: SymbolicMemoryCore, m: MotifNode) -> float:
    """
    1 - max cosine(new, any prior). Higher => more semantically novel.
    """
    new_vec = embed_text(m.content)
    prior_vecs = _all_vectors(smc, exclude=m)
    max_sim = _max_cosine(new_vec, prior_vecs)
    return max(0.0, 1.0 - max_sim)

def symbolic_novelty(smc: SymbolicMemoryCore, m: MotifNode) -> float:
    """
    Fraction of motif symbols that are new to the graph.
    """
    seen: Set[str] = set()
    for x in _others(smc, m):
        for s in x.symbols:
            seen.add(s.lower())
    if not m.symbols:
        return 0.0
    new_syms = [s for s in m.symbols if s.lower() not in seen]
    return len(new_syms) / float(len(m.symbols))

def structural_novelty(smc: SymbolicMemoryCore, m: MotifNode, sim_threshold: float = 0.35) -> float:
    """
    Cheap bridge proxy: proportion of existing motifs similar to m above threshold.
    Normalized to [0,1] by the number of *other* motifs (m itself excluded).
    """
    others = _others(smc, m)
    if not others:
        return 0.0
    mv = embed_text(m.content)
    hits = 0
    for other in others:
        if cosine_sparse(mv, embed_text(other.content)) >= sim_threshold:
            hits += 1
    return hits / float(len(others))

def novelty_index(
    smc: SymbolicMemoryCore,
    m: MotifNode,
    alpha: float = 0.5,   # semantic
    beta: float = 0.2,    # symbolic
    gamma: float = 0.3,   # structural
    sim_threshold: float = 0.35
) -> Dict[str, float]:
    sem = semantic_novelty(smc, m)
    sym = symbolic_novelty(smc, m)
    stc = structural_novelty(smc, m, sim_threshold=sim_threshold)
    score = alpha * sem + beta * sym + gamma * stc
    return {"semantic": sem, "symbolic": sym, "structural": stc, "novelty_index": score}

