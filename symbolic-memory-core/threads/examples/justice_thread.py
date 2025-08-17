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
