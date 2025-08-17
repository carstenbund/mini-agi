# scripts/migrate_to_chroma.py
from __future__ import annotations
from symbolic_recursion.core.motif import SymbolicMemoryCore
from symbolic_recursion.core.storage import load_motifs
from symbolic_recursion.core.chroma_router import ChromaRouter
from symbolic_recursion.embeddings.sbert import Embeddings

def main():
    smc = SymbolicMemoryCore(); smc.motifs = load_motifs()
    router = ChromaRouter(embeddings=Embeddings())  # or None to force legacy fallback
    router.rebuild_from_smc(smc)
    print(f"Indexed {len(smc.motifs)} motifs into Chroma at data/chroma")

if __name__ == "__main__":
    main()

