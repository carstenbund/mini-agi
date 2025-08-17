import json
import os
from typing import Dict
from symbolic_recursion.core.motif import MotifNode

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "motifs.json")
DATA_PATH = os.path.abspath(DATA_PATH)

def save_motifs(motif_map: Dict[str, MotifNode]) -> None:
    payload = {mid: m.to_dict() for mid, m in motif_map.items()}
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_motifs() -> Dict[str, MotifNode]:
    if not os.path.exists(DATA_PATH):
        return {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: MotifNode.from_dict(v) for k, v in raw.items()}
