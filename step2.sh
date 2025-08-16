#!/bin/bash
set -e

# Root
mkdir -p symbolic-memory-core/{core,threads/examples,embeddings,data,utils}
touch symbolic-memory-core/{core,threads,embeddings,utils}/__init__.py

# --------------------
# core/motif.py
# --------------------
cat > symbolic-memory-core/core/motif.py << 'EOF'
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Revision:
    timestamp: str
    content: str
    symbols: List[str]

@dataclass
class MotifNode:
    id: str
    symbols: List[str]
    content: str
    thread_id: str
    references: List[str] = field(default_factory=list)
    history: List[Revision] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MotifNode":
        hist = [Revision(**h) for h in d.get("history", [])]
        return MotifNode(
            id=d["id"],
            symbols=list(d.get("symbols", [])),
            content=d.get("content", ""),
            thread_id=d.get("thread_id", ""),
            references=list(d.get("references", [])),
            history=hist,
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )

class SymbolicMemoryCore:
    """
    Minimal in-memory graph for motifs with JSON persistence handled by core.storage.
    Similarity search is delegated to the router layer.
    """
    def __init__(self):
        self.motifs: Dict[str, MotifNode] = {}

    # CRUD
    def add_motif(self, motif: MotifNode) -> None:
        self.motifs[motif.id] = motif

    def get_motif(self, motif_id: str) -> Optional[MotifNode]:
        return self.motifs.get(motif_id)

    def list_motifs(self) -> List[MotifNode]:
        return list(self.motifs.values())

    def update_motif(self, motif_id: str, content: Optional[str] = None, symbols: Optional[List[str]] = None) -> bool:
        m = self.motifs.get(motif_id)
        if not m:
            return False
        # record revision
        m.history.append(Revision(timestamp=datetime.utcnow().isoformat(),
                                  content=m.content,
                                  symbols=m.symbols.copy()))
        if content is not None:
            m.content = content
        if symbols is not None:
            m.symbols = symbols
        m.updated_at = datetime.utcnow().isoformat()
        return True

    # Linking
    def link_motifs(self, a: str, b: str) -> bool:
        if a not in self.motifs or b not in self.motifs:
            return False
        if b not in self.motifs[a].references:
            self.motifs[a].references.append(b)
        return True

    # Simple symbol filter
    def find_by_symbol(self, token: str) -> List[MotifNode]:
        t = token.lower()
        return [m for m in self.motifs.values() if any(t == s.lower() for s in m.symbols)]
EOF

# --------------------
# core/storage.py
# --------------------
cat > symbolic-memory-core/core/storage.py << 'EOF'
import json
import os
from typing import Dict
from core.motif import MotifNode

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
EOF

# --------------------
# embeddings/embedder.py
# --------------------
cat > symbolic-memory-core/embeddings/embedder.py << 'EOF'
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
EOF

# --------------------
# core/router.py
# --------------------
cat > symbolic-memory-core/core/router.py << 'EOF'
from typing import List, Tuple
from embeddings.embedder import embed_text, cosine_sparse
from core.motif import MotifNode, SymbolicMemoryCore

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
EOF

# --------------------
# core/ollama_interface.py
# --------------------
cat > symbolic-memory-core/core/ollama_interface.py << 'EOF'
import subprocess
from typing import Optional

def query_ollama(prompt: str, model: str = "llama2", timeout: Optional[int] = None) -> str:
    """
    Calls local Ollama: `ollama run <model>`
    Returns captured stdout as a string.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate(input=prompt, timeout=timeout)
        if proc.returncode != 0:
            return f"[ollama error] {err.strip()}"
        return out
    except FileNotFoundError:
        return "[ollama error] 'ollama' CLI not found."
    except subprocess.TimeoutExpired:
        proc.kill()
        return "[ollama error] request timed out."
EOF

# --------------------
# threads/manager.py
# --------------------
cat > symbolic-memory-core/threads/manager.py << 'EOF'
from typing import List, Optional
from core.motif import MotifNode, SymbolicMemoryCore
from core.ollama_interface import query_ollama
from utils.id_gen import generate_id
from datetime import datetime

class ChatThread:
    def __init__(self, name: str, model: str = "llama2"):
        self.name = name
        self.model = model
        self.history: List[str] = []

    def ask(self, prompt: str) -> str:
        self.history.append(f"[{datetime.utcnow().isoformat()}] USER: {prompt}")
        resp = query_ollama(prompt, model=self.model)
        self.history.append(f"[{datetime.utcnow().isoformat()}] ASSISTANT: {resp}")
        return resp

class ThreadManager:
    def __init__(self, smc: SymbolicMemoryCore):
        self.smc = smc
        self.threads: List[ChatThread] = []

    def new_thread(self, name: str, model: str = "llama2") -> ChatThread:
        t = ChatThread(name=name, model=model)
        self.threads.append(t)
        return t

    def capture_as_motif(self, thread: ChatThread, symbols: List[str], content: str) -> MotifNode:
        m = MotifNode(
            id=generate_id(),
            symbols=symbols,
            content=content,
            thread_id=thread.name
        )
        self.smc.add_motif(m)
        return m

    def list_threads(self) -> List[str]:
        return [t.name for t in self.threads]
EOF

# --------------------
# threads/examples/justice_thread.py
# --------------------
cat > symbolic-memory-core/threads/examples/justice_thread.py << 'EOF'
from core.motif import SymbolicMemoryCore
from threads.manager import ThreadManager

if __name__ == "__main__":
    smc = SymbolicMemoryCore()
    tm = ThreadManager(smc)
    t = tm.new_thread("justice_thread", model="llama2")

    prompt = "In brief, outline 'Justice as a Gradient' in 5 bullet points."
    resp = t.ask(prompt)
    tm.capture_as_motif(thread=t, symbols=["justice", "gradient"], content=resp)
    print("Captured motif. Threads:", tm.list_threads(), "Motifs:", len(smc.list_motifs()))
EOF

# --------------------
# utils/id_gen.py
# --------------------
cat > symbolic-memory-core/utils/id_gen.py << 'EOF'
import uuid

def generate_id() -> str:
    return str(uuid.uuid4())
EOF

# --------------------
# data seed
# --------------------
echo "{}" > symbolic-memory-core/data/motifs.json

# --------------------
# Reader.md (same as before)
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

- Embeddings: tiny local bag-of-words cosine (stdlib only)
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
- [x] Minimal router with cosine similarity
- [x] Multiple thread support (lightweight)
- [ ] Interactive terminal UI or minimal Flask UI
- [ ] Visualization of motif links (e.g. D3, NetworkX)

---

## License

MIT / Apache 2.0 (TBD)
EOF

# --------------------
# main.py with CLI
# --------------------
cat > symbolic-memory-core/main.py << 'EOF'
import argparse
from core.motif import MotifNode, SymbolicMemoryCore
from core.storage import save_motifs, load_motifs
from core.router import rank_similar, suggest_links
from threads.manager import ThreadManager
from utils.id_gen import generate_id

def load_smc() -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    smc.motifs = load_motifs()
    return smc

def save_smc(smc: SymbolicMemoryCore) -> None:
    save_motifs(smc.motifs)

def cmd_add(args):
    smc = load_smc()
    m = MotifNode(
        id=generate_id(),
        symbols=[s.strip() for s in args.symbols.split(",")] if args.symbols else [],
        content=args.content,
        thread_id=args.thread
    )
    smc.add_motif(m)
    save_smc(smc)
    print("Added motif:", m.id)

def cmd_list(args):
    smc = load_smc()
    for m in smc.list_motifs():
        print(f"{m.id} | symbols={m.symbols} | thread={m.thread_id} | refs={len(m.references)}")

def cmd_link(args):
    smc = load_smc()
    ok = smc.link_motifs(args.a, args.b)
    save_smc(smc)
    print("Linked." if ok else "Link failed (check IDs).")

def cmd_query(args):
    smc = load_smc()
    results = rank_similar(smc, args.text, top_k=args.k)
    for m, score in results:
        print(f"{m.id}  score={score:.3f}  symbols={m.symbols}  thread={m.thread_id}")

def cmd_suggest(args):
    smc = load_smc()
    results = suggest_links(smc, args.motif_id, top_k=args.k)
    for m, score in results:
        print(f"{m.id}  score={score:.3f}  symbols={m.symbols}  thread={m.thread_id}")

def cmd_chat(args):
    smc = load_smc()
    tm = ThreadManager(smc)
    t = tm.new_thread(args.name, model=args.model)
    resp = t.ask(args.prompt)
    if args.capture:
        m = tm.capture_as_motif(thread=t, symbols=[s.strip() for s in args.capture.split(",")], content=resp)
        save_smc(smc)
        print("Captured motif:", m.id)
    print("--- Response ---")
    print(resp)

def main():
    p = argparse.ArgumentParser(description="Symbolic Memory Core CLI")
    sub = p.add_subparsers(required=True)

    p_add = sub.add_parser("add", help="Add a motif")
    p_add.add_argument("--symbols", type=str, default="")
    p_add.add_argument("--content", type=str, required=True)
    p_add.add_argument("--thread", type=str, default="manual")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List motifs")
    p_list.set_defaults(func=cmd_list)

    p_link = sub.add_parser("link", help="Link two motifs (A -> B)")
    p_link.add_argument("a", type=str)
    p_link.add_argument("b", type=str)
    p_link.set_defaults(func=cmd_link)

    p_query = sub.add_parser("query", help="Find motifs similar to text")
    p_query.add_argument("--text", type=str, required=True)
    p_query.add_argument("--k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    p_sug = sub.add_parser("suggest", help="Suggest links for a given motif")
    p_sug.add_argument("motif_id", type=str)
    p_sug.add_argument("--k", type=int, default=5)
    p_sug.set_defaults(func=cmd_suggest)

    p_chat = sub.add_parser("chat", help="Ask local Ollama and (optionally) capture as motif")
    p_chat.add_argument("--name", type=str, default="session")
    p_chat.add_argument("--model", type=str, default="llama2")
    p_chat.add_argument("--prompt", type=str, required=True)
    p_chat.add_argument("--capture", type=str, help="Comma-separated symbols to store result as motif")
    p_chat.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
EOF

echo "✅ Project structure with classes created in ./symbolic-memory-core"
echo "Next:"
echo "  cd symbolic-memory-core"
echo "  python3 main.py add --symbols justice,gradient --content 'Justice as fairness across contexts' --thread seed"
echo "  python3 main.py list"
echo "  python3 main.py query --text 'fairness and justice in society'"
echo "  # If Ollama is installed:"
echo "  python3 main.py chat --prompt 'Briefly outline Justice as a Gradient' --capture justice,gradient"
