import os
from typing import List, Optional
import numpy as np


class _LazySBERT:
    model = None
    error = None

    @classmethod
    def load(cls, name: str):
        if cls.model is not None or cls.error is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            cls.model = SentenceTransformer(name)
        except Exception as e:
            cls.error = e


class Embeddings:
    """
    Wrapper around sentence-transformers (SBERT). If not available,
    raises at first use so caller can fallback.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "SMC_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _LazySBERT.load(self.model_name)

    def _model(self):
        if _LazySBERT.error is not None:
            raise RuntimeError(f"SBERT unavailable: {_LazySBERT.error}")
        if _LazySBERT.model is None:
            raise RuntimeError("SBERT model not loaded")
        return _LazySBERT.model

    def encode_text(self, text: str) -> np.ndarray:
        m = self._model()
        v = m.encode(text or "", convert_to_numpy=True, normalize_embeddings=True)
        return v.astype(np.float32, copy=False)

    def encode_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        m = self._model()
        bs = int(os.getenv("SMC_EMB_BATCH", "32")) if batch_size is None else batch_size
        vs = m.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=bs
        )
        return vs.astype(np.float32, copy=False)

