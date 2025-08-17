from core.motif import MotifNode, SymbolicMemoryCore
from utils.id_gen import generate_id

if __name__ == "__main__":
    smc = SymbolicMemoryCore()
    motif = MotifNode(generate_id(), ["justice"], "Justice as fairness", "thread_1")
    smc.add_motif(motif)
    print("Motifs in memory:", smc.motifs)
