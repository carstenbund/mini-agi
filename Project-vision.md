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
