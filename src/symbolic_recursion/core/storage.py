import json
import os
from typing import Dict
from symbolic_recursion.core.motif import MotifNode


def data_path() -> str:
    """Resolve the motif store path.

    ``SMC_DATA_PATH`` env var wins; otherwise ``data/motifs.json`` relative
    to the current working directory — the same convention as the Chroma
    persist dir (``data/chroma``) and the README. Resolved at call time so
    env/cwd changes take effect without re-import.

    (Previously this was resolved relative to the package directory, which
    silently split the store between ``data/`` and ``src/.../data/``.)
    """
    return os.path.abspath(
        os.environ.get("SMC_DATA_PATH", os.path.join("data", "motifs.json"))
    )


def save_motifs(motif_map: Dict[str, MotifNode]) -> None:
    path = data_path()
    payload = {mid: m.to_dict() for mid, m in motif_map.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_motifs() -> Dict[str, MotifNode]:
    path = data_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: MotifNode.from_dict(v) for k, v in raw.items()}
