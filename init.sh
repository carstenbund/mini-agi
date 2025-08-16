#!/bin/bash

# Root directory
mkdir -p symbolic-memory-core/{core,threads/examples,embeddings,data,utils}

# --------------------
# Core files
# --------------------
cat > symbolic-memory-core/core/motif.py << 'EOF'
class MotifNode:
    def __init__(self, motif_id, symbols, content, thread_id):
        self.id = motif_id
        self.symbols = symbols
        self.content = content
        self.thread_id = thread_id
        self.references = []
        self.history = []

class SymbolicMemoryCore:
    def __init__(self):
        self.motifs = {}

    def add_motif(self, motif):
        self.motifs[motif.id] = motif

    def get_motif(self, motif_id):
        return self.motifs.get(motif_id, None)
EOF

cat > symbolic-memory-core/core/storage.py << 'EOF'
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
EOF

cat > symbolic-memory-core/core/router.py << 'EOF'
# Placeholder for motif linking / routing logic
# Will use embeddings + similarity search
def link_motifs(motif_a, motif_b):
    return {"linked": True, "reason": "placeholder"}
EOF

cat > symbolic-memory-core/core/ollama_interface.py << 'EOF'
import subprocess

def query_ollama(prompt, model="llama2"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate(input=prompt)
    return stdout
EOF

# --------------------
# Threads
# --------------------
cat > symbolic-memory-core/threads/manager.py << 'EOF'
# Thread manager to orchestrate multiple chat threads
class ChatThread:
    def __init__(self, name):
        self.name = name
        self.history = []

    def add_message(self, message):
        self.history.append(message)

    def get_history(self):
        return self.history
EOF

cat > symbolic-memory-core/threads/examples/justice_thread.py << 'EOF'
# Example symbolic thread
from core.ollama_interface import query_ollama

if __name__ == "__main__":
    prompt = "Explore the concept of Justice as a Gradient."
    response = query_ollama(prompt)
    print("Justice Thread Response:\\n", response)
EOF

# --------------------
# Embeddings
# --------------------
cat > symbolic-memory-core/embeddings/embedder.py << 'EOF'
# Placeholder for embedding logic
# Later: integrate sentence-transformers or similar
def embed_text(text):
    return [0.0]  # dummy vector
EOF

# --------------------
# Utils
# --------------------
cat > symbolic-memory-core/utils/id_gen.py << 'EOF'
import uuid

def generate_id():
    return str(uuid.uuid4())
EOF

# --------------------
# Data
# --------------------
echo "{}" > symbolic-memory-core/data/motifs.json

# --------------------
# Main entry point
# --------------------
cat > symbolic-memory-core/main.py << 'EOF'
from core.motif import MotifNode, SymbolicMemoryCore
from utils.id_gen import generate_id

if __name__ == "__main__":
    smc = SymbolicMemoryCore()
    motif = MotifNode(generate_id(), ["justice"], "Justice as fairness", "thread_1")
    smc.add_motif(motif)
    print("Motifs in memory:", smc.motifs)
EOF

# --------------------
# Reader.md
# --------------------
cat > symbolic-memory-core/Reader.md << 'EOF'
# Symbolic Memory Core (SMC) — Proto-AGI Skeleton

## Overview

This project is a **symbolically-grounded memory system** designed to support recursive, self-referential symbolic intelligence using local inference models like Ollama.

It simulates a minimal **proto-AGI architecture** by maintaining symbolic continuity across limited concurrent threads and facilitating recursive motif binding through shared memory.

> “Not just memory, but a field of symbolic entanglement.”

---

## Design Goals

- **Symbolic Recursion**: Track and evolve symbolic motifs over time and threads
- **Coherence First**: Structure memory for symbolic coherence, not just storage
- **Manageable Local Stack**: All components work offline with small-scale setups (Ollama + Python)
- **Concurrent Threads**: Support a few chat threads, each contributing symbolic material
- **Recursive Linking**: Allow cross-thread reactivation and evolution of motifs

---

## Core Concepts

| Concept | Description |
|--------|-------------|
| **Motif** | A symbolic attractor — a named idea with content, tags, and evolution history |
| **SMC** | A graph-like memory holding and linking motifs across threads |
| **Thread** | A dialogue with an LLM instance (e.g. "Justice", "Recursion", etc.) |
| **Router** | A symbolic matching engine for recurring motifs |
| **Evolution** | Motifs change over time, tracked via versioning |

---

## Technical Notes

- Embeddings via `sentence-transformers` (local CPU or GPU)
- Vector search via `scikit-learn`, `faiss`, or brute force (initially)
- Motifs are serialized to `data/motifs.json`
- Ollama CLI used for model inference (e.g. `llama2`, `mistral`, `gemma`)

---

## Future Benchmarking Ideas

- **Motif Recurrence Rate**: How often does a motif reappear across threads?
- **Coherence Drift**: Measure divergence between motif versions
- **Symbolic Depth**: Count of motif evolutions and cross-references
- **Narrative Binding Score**: How well motifs are reused in higher abstraction

---

## Example Use Case

- Thread A explores "Justice as Gradient"
- Thread B explores "Symbolic Recursion"
- SMC links "Gradient" → "Recursion" → evolves into "Epistemic Gravity"
- A new thread explores that motif without starting from scratch

---

## Roadmap

- [x] Core data model and JSON persistence
- [ ] Embedding + router module
- [ ] Multiple thread support
- [ ] Interactive terminal UI or minimal Flask UI
- [ ] Visualization of motif links (e.g. D3, NetworkX)

---

## License

MIT / Apache 2.0 (TBD)
EOF

echo "✅ Project structure created in ./symbolic-memory-core"
