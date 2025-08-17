from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from symbolic_recursion.embeddings.sbert import Embeddings
from symbolic_recursion.core.motif import SymbolicMemoryCore, MotifNode
from symbolic_recursion.embeddings.embedder import embed_text as legacy_embed


class VectorRouter:
    """
    Vector search router:
      - Prefers SBERT + FAISS
      - Falls back to SBERT + numpy
      - Finally falls back to legacy sparse embedder if SBERT missing
    """

    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.emb = embeddings
        self.ids: List[str] = []
        self.id_to_motif: Dict[str, MotifNode] = {}
        self.mat: Optional[np.ndarray] = None
        self.index = None  # FAISS index if used

    # --- encoding helpers -------------------------------------------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.emb is not None:
            return self.emb.encode_texts(texts)
        # fallback: legacy sparse -> dense via bag-of-words dict to vector space
        voc: Dict[str, int] = {}
        vecs = []
        for t in texts:
            d = legacy_embed(t)
            for k in d.keys():
                if k not in voc:
                    voc[k] = len(voc)
            vecs.append(d)
        D = len(voc)
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, d in enumerate(vecs):
            for k, v in d.items():
                out[i, voc[k]] = v
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        out /= norms
        return out

    # --- index management -------------------------------------------------
    def rebuild_from_smc(self, smc: SymbolicMemoryCore) -> None:
        motifs = smc.list_motifs()
        self.id_to_motif = {m.id: m for m in motifs}
        self.ids = list(self.id_to_motif.keys())
        texts = [m.content for m in self.id_to_motif.values()]
        if len(texts) == 0:
            self.mat = None
            self.index = None
            return
        mat = self._encode(texts)
        self.mat = mat
        if _HAS_FAISS:
            d = mat.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(mat)
        else:
            self.index = None

    def add_motif(self, smc: SymbolicMemoryCore, m: MotifNode) -> None:
        v = self._encode([m.content])
        self.id_to_motif[m.id] = m
        if self.mat is None:
            self.ids = [m.id]
            self.mat = v
            if _HAS_FAISS:
                d = v.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(v)
            return
        self.ids.append(m.id)
        self.mat = np.vstack([self.mat, v])
        if _HAS_FAISS and self.index is not None:
            self.index.add(v)

    def remove_motif(self, motif_id: str) -> None:
        if self.mat is None:
            return
        if motif_id not in self.ids:
            return
        idx = self.ids.index(motif_id)
        self.ids.pop(idx)
        self.id_to_motif.pop(motif_id, None)
        self.mat = np.delete(self.mat, idx, axis=0)
        if _HAS_FAISS and self.index is not None:
            d = self.mat.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.mat)

    # --- search -----------------------------------------------------------
    def _topk_numpy(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = self.mat @ q.T
        sims = sims.reshape(-1)
        if k >= sims.size:
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            idx = part[np.argsort(-sims[part])]
        return sims[idx], idx

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]:
        if self.mat is None or len(self.ids) == 0:
            return []
        q = self._encode([text])
        if _HAS_FAISS and self.index is not None:
            D, I = self.index.search(q, top_k)
            sims = D[0]
            idxs = I[0]
        else:
            sims, idxs = self._topk_numpy(q, top_k)
        return [
            (self.id_to_motif[self.ids[i]], float(sims[n]))
            for n, i in enumerate(idxs)
        ]

    def suggest_for_motif(
        self, smc: SymbolicMemoryCore, motif_id: str, top_k: int = 5
    ) -> List[Tuple[MotifNode, float]]:
        m = self.id_to_motif.get(motif_id) or smc.get_motif(motif_id)
        if not m or self.mat is None:
            return []
        results = self.search_text(m.content, top_k=top_k + 1)
        out: List[Tuple[MotifNode, float]] = []
        for cand, score in results:
            if cand.id == motif_id:
                continue
            out.append((cand, score))
            if len(out) >= top_k:
                break
        return out

