# src/symbolic_recursion/core/chroma_router.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import threading
import numpy as np

import chromadb
from chromadb.config import Settings

from symbolic_recursion.core.motif import SymbolicMemoryCore, MotifNode
from symbolic_recursion.embeddings.sbert import Embeddings
from symbolic_recursion.embeddings.embedder import embed_text as legacy_embed


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat is None:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return (mat / norms).astype(np.float32)


def _as_sim_from_cosine_distance(dist: float) -> float:
    # Chroma returns distances for cosine; similarity ≈ 1 - dist
    # Clamp to [0,1] for safety.
    if dist is None:
        return 0.0
    try:
        d = float(dist)
    except Exception:
        return 0.0
    if d < 0.0:
        d = 0.0
    if d > 1.0:
        # older builds may report up to 2; keep ordering by best-effort
        d = min(d, 2.0)
    return max(0.0, 1.0 - d)


class ChromaRouter:
    """
    Persistent vector router backed by Chroma, API-compatible with VectorRouter.

    Prefers SBERT.encode_texts (batch) -> np.ndarray.
    Falls back to legacy sparse embedder packed into a dense space (normalized).
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        persist_dir: str = "data/chroma",
        collection_name: str = "motifs",
        metric: str = "cosine",  # "cosine" | "l2" | "ip"
    ):
        self.emb: Optional[Embeddings] = embeddings
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.metric = metric

        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=False)
        )
        self._coll = self._get_or_create_collection()
        self.id_to_motif: Dict[str, MotifNode] = {}
        self.ids: List[str] = []
        self._smc: Optional[SymbolicMemoryCore] = None

    # -------- embedding (mirror your VectorRouter) ------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        # Preferred: SBERT
        if self.emb is not None:
            try:
                mat = self.emb.encode_texts(texts)
                return _l2_normalize(np.asarray(mat, dtype=np.float32))
            except Exception:
                pass

        # Fallback: legacy sparse → dense bag with normalization
        voc: Dict[str, int] = {}
        vecs = []
        for t in texts:
            d = legacy_embed(t)  # dict[token] -> weight
            for k in d.keys():
                if k not in voc:
                    voc[k] = len(voc)
            vecs.append(d)
        D = len(voc)
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, d in enumerate(vecs):
            for k, v in d.items():
                out[i, voc[k]] = v
        return _l2_normalize(out)

    # -------- chroma helpers ---------------------------------------------

    def _get_or_create_collection(self):
        try:
            return self._client.get_collection(self.collection_name)
        except Exception:
            return self._client.create_collection(
                self.collection_name,
                metadata={"hnsw:space": self.metric}
            )

    # -------- public API (compatible) ------------------------------------

    def rebuild_from_smc(self, smc: SymbolicMemoryCore) -> None:
        with self._lock:
            self._smc = smc
            motifs = smc.list_motifs()
            self.id_to_motif = {m.id: m for m in motifs}
            self.ids = list(self.id_to_motif.keys())

            # Recreate collection for a clean rebuild
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._coll = self._client.create_collection(
                self.collection_name, metadata={"hnsw:space": self.metric}
            )

            if not self.ids:
                return

            texts = [m.content for m in self.id_to_motif.values()]
            mat = self._encode(texts)  # (N, D), float32, L2-normalized

            # Chroma needs lists of lists
            embeddings = [row.tolist() for row in mat]
            metadatas = [{"symbols": ",".join(m.symbols)} for m in self.id_to_motif.values()]
            documents = texts
            ids = self.ids

            self._coll.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def add_motif(self, smc: SymbolicMemoryCore, m: MotifNode) -> None:
        with self._lock:
            self._smc = smc
            self.id_to_motif[m.id] = m
            self.ids.append(m.id)
            v = self._encode([m.content])[0].tolist()
            self._coll.upsert(
                ids=[m.id],
                documents=[m.content],
                embeddings=[v],
                metadatas=[{"symbols": ",".join(m.symbols)}],
            )

    def add_many(self, smc: SymbolicMemoryCore, motifs: List[MotifNode]) -> None:
        if not motifs:
            return
        with self._lock:
            self._smc = smc
            for m in motifs:
                self.id_to_motif[m.id] = m
                if m.id not in self.ids:
                    self.ids.append(m.id)
            embs = self._encode([m.content for m in motifs])
            self._coll.upsert(
                ids=[m.id for m in motifs],
                documents=[m.content for m in motifs],
                embeddings=[row.tolist() for row in embs],
                metadatas=[{"symbols": ",".join(m.symbols)} for m in motifs],
            )


    def remove_motif(self, motif_id: str) -> None:
        with self._lock:
            if motif_id in self.id_to_motif:
                self.id_to_motif.pop(motif_id, None)
            if motif_id in self.ids:
                self.ids.remove(motif_id)
            try:
                self._coll.delete(ids=[motif_id])
            except Exception:
                pass

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[MotifNode, float]]:
        if self._smc is None or not self.ids:
            return []
        q = self._encode([text])[0].tolist()
        with self._lock:
            res = self._coll.query(query_embeddings=[q], n_results=max(1, top_k))
        out: List[Tuple[MotifNode, float]] = []
        ids = res.get("ids", [[]])
        dists = res.get("distances", [[]])
        if not ids or not ids[0]:
            return out
        for i in range(len(ids[0])):
            mid = ids[0][i]
            dist = dists[0][i] if dists and dists[0] else None
            sim = _as_sim_from_cosine_distance(dist)
            m = self.id_to_motif.get(mid) or (self._smc.get_motif(mid) if self._smc else None)
            if m is not None:
                out.append((m, sim))
        return out

    def suggest_for_motif(
        self, smc: SymbolicMemoryCore, motif_id: str, top_k: int = 5
    ) -> List[Tuple[MotifNode, float]]:
        m = self.id_to_motif.get(motif_id) or smc.get_motif(motif_id)
        if not m:
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

