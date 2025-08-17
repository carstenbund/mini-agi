from collections import Counter
from math import sqrt
import re
from typing import Dict, List

WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]

def embed_text(text: str) -> Dict[str, float]:
    """
    Extremely simple sparse embedding: term frequency.
    Returns a dict[word] -> weight for cosine over sparse vectors.
    """
    toks = tokenize(text)
    if not toks:
        return {}
    counts = Counter(toks)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # dot product
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in keys)
    na = sqrt(sum(v*v for v in a.values()))
    nb = sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
