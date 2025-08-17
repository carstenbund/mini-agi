from typing import List, Tuple
from symbolic_recursion.embeddings.embedder import embed_text, cosine_sparse
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore

def rank_similar(smc: SymbolicMemoryCore, query_text: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]:
    q = embed_text(query_text)
    scored = []
    for m in smc.list_motifs():
        score = cosine_sparse(q, embed_text(m.content))
        scored.append((m, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def suggest_links(smc: SymbolicMemoryCore, motif_id: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]:
    m = smc.get_motif(motif_id)
    if not m:
        return []
    base = embed_text(m.content)
    scored = []
    for other in smc.list_motifs():
        if other.id == motif_id:
            continue
        score = cosine_sparse(base, embed_text(other.content))
        scored.append((other, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
