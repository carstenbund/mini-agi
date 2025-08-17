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
    p_chat.add_argument("--model", type=str, default="llama3:instruct")
    p_chat.add_argument("--prompt", type=str, required=True)
    p_chat.add_argument("--capture", type=str, help="Comma-separated symbols to store result as motif")
    p_chat.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
