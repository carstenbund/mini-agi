import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "motifs.json")

def save_motifs(motifs):
    with open(DATA_PATH, 'w') as f:
        json.dump(motifs, f, indent=2)

def load_motifs():
    if not os.path.exists(DATA_PATH):
        return {}
    with open(DATA_PATH, 'r') as f:
        return json.load(f)
