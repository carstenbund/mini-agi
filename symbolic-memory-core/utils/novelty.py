from typing import Dict, List, Set
from embeddings.embedder import embed_text, cosine_sparse
from core.motif import SymbolicMemoryCore, MotifNode

def _all_vectors(smc: SymbolicMemoryCore) -> List[Dict[str, float]]:
    return [embed_text(m.content) for m in smc.list_motifs()]

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
    prior_vecs = _all_vectors(smc)
    max_sim = _max_cosine(new_vec, prior_vecs)
    return max(0.0, 1.0 - max_sim)

def symbolic_novelty(smc: SymbolicMemoryCore, m: MotifNode) -> float:
    """
    Fraction of motif symbols that are new to the graph.
    """
    seen: Set[str] = set()
    for x in smc.list_motifs():
        for s in x.symbols:
            seen.add(s.lower())
    if not m.symbols:
        return 0.0
    new_syms = [s for s in m.symbols if s.lower() not in seen]
    return len(new_syms) / float(len(m.symbols))

def structural_novelty(smc: SymbolicMemoryCore, m: MotifNode, sim_threshold: float = 0.35) -> float:
    """
    Cheap bridge proxy: proportion of existing motifs similar to m above threshold.
    Normalized to [0,1] by |V|.
    """
    V = len(smc.motifs)
    if V == 0:
        return 0.0
    mv = embed_text(m.content)
    hits = 0
    for other in smc.list_motifs():
        if other.id == m.id:
            continue
        if cosine_sparse(mv, embed_text(other.content)) >= sim_threshold:
            hits += 1
    return hits / float(V)

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
