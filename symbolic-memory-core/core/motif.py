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
