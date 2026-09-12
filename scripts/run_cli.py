import argparse
import os

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.storage import save_motifs, load_motifs
from symbolic_recursion.core.router import rank_similar, suggest_links
from symbolic_recursion.threads.manager import ThreadManager
from symbolic_recursion.utils.id_gen import generate_id
from symbolic_recursion.core.agent import agent_id


def _init_router(kind: str, smc: SymbolicMemoryCore):
    """Instantiate a vector router if requested.

    Parameters
    ----------
    kind: str
        Name of the router backend ("vector" or "chroma").
    smc: SymbolicMemoryCore
        The motif store used to populate the router.
    """
    kind = (kind or "").lower()
    router = None
    if kind == "vector":
        from symbolic_recursion.core.vector_router import VectorRouter

        try:
            from symbolic_recursion.embeddings.sbert import Embeddings

            emb = Embeddings()
            router = VectorRouter(emb)
        except Exception:
            router = VectorRouter(None)
        router.rebuild_from_smc(smc)
    elif kind == "chroma":
        from symbolic_recursion.core.chroma_router import ChromaRouter

        try:
            from symbolic_recursion.embeddings.sbert import Embeddings

            emb = Embeddings()
            router = ChromaRouter(emb)
        except Exception:
            router = ChromaRouter(None)
        router.rebuild_from_smc(smc)
    return router

def load_smc() -> SymbolicMemoryCore:
    smc = SymbolicMemoryCore()
    smc.motifs = load_motifs()
    return smc

def _journal(smc, event: dict) -> None:
    from symbolic_recursion.graph.trajectory import record_event
    record_event(smc, event)

def save_smc(smc: SymbolicMemoryCore) -> None:
    save_motifs(smc.motifs)

def cmd_add(args):
    smc = load_smc()
    m = MotifNode(
        id=generate_id(),
        symbols=[s.strip() for s in args.symbols.split(",")] if args.symbols else [],
        content=args.content,
        thread_id=args.thread,
        agent=agent_id(),
    )
    smc.add_motif(m)
    save_smc(smc)
    _journal(smc, {"type": "capture", "motif_id": m.id, "via": "add"})
    print("Added motif:", m.id)

def cmd_list(args):
    smc = load_smc()
    for m in smc.list_motifs():
        print(f"{m.id} | symbols={m.symbols} | thread={m.thread_id} | refs={len(m.references)}")

def cmd_link(args):
    smc = load_smc()
    ok = smc.link_motifs(args.a, args.b)
    save_smc(smc)
    if ok:
        _journal(smc, {"type": "link", "a": args.a, "b": args.b})
    print("Linked." if ok else "Link failed (check IDs).")

def cmd_query(args):
    smc = load_smc()
    router = _init_router(args.router, smc)
    if router:
        results = router.search_text(args.text, top_k=args.k)
    else:
        results = rank_similar(smc, args.text, top_k=args.k)
    for m, score in results:
        print(f"{m.id}  score={score:.3f}  symbols={m.symbols}  thread={m.thread_id}")

def cmd_suggest(args):
    smc = load_smc()
    router = _init_router(args.router, smc)
    if router:
        results = router.suggest_for_motif(smc, args.motif_id, top_k=args.k)
    else:
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
        _journal(smc, {"type": "capture", "motif_id": m.id, "via": "chat"})
        from symbolic_recursion.core.flow import record_flow
        record_flow({"kind": "chat", "thread": t.name, "model": args.model,
                     "motif_id": m.id, "targets": [],
                     "prompt": args.prompt, "response": resp})
        print("Captured motif:", m.id)
    print("--- Response ---")
    print(resp)

def cmd_capture(args):
    from symbolic_recursion.documents.capture import capture_document, plan_capture

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()
    smc = load_smc()
    if args.dry_run:
        plan = plan_capture(smc, text, title=args.title, thread=args.thread, prefix=args.prefix)
        print(f"title: {plan['title']}  thread: {plan['thread']}  motifs: {len(plan['motifs'])}")
        for spec in plan["motifs"]:
            print(f"  {spec['id']}  [{', '.join(spec['symbols'])}]  refs={spec['references']}")
            if spec["unresolved_links"]:
                print(f"    unresolved links: {spec['unresolved_links']}")
        return
    plan = capture_document(smc, text, title=args.title, thread=args.thread, prefix=args.prefix)
    save_smc(smc)
    for spec in plan["motifs"]:
        _journal(smc, {"type": "capture", "motif_id": spec["id"], "via": "document"})
    unresolved = sorted({u for s in plan["motifs"] for u in s["unresolved_links"]})
    print(f"captured {len(plan['motifs'])} motifs in thread {plan['thread']} "
          f"(registered as '{plan['registered_as']}')")
    if unresolved:
        print("unresolved wiki-links (capture those documents to connect them):",
              ", ".join(unresolved))

def cmd_claims(args):
    from symbolic_recursion.core.claims import active_claims, release

    if args.release:
        release(args.kind, args.release, agent_id())
        print(f"released {args.kind} {args.release} for {agent_id()}")
        return
    active = active_claims()
    if not active:
        print("no active claims")
        return
    print(f"{'kind':<8} {'owner':<16} {'expires':<20} key")
    for (kind, key), r in sorted(active.items(), key=lambda kv: kv[1]["ts"]):
        print(f"{kind:<8} {r['owner']:<16} {r['expires'][:19]:<20} {key}")

def cmd_trace(args):
    from symbolic_recursion.core.flow import render_trace

    smc = load_smc()
    print(render_trace(smc, args.motif_id, full=not args.short), end="")

def cmd_report(args):
    from symbolic_recursion.graph import analyze_field, render_report

    if args.trajectory:
        from symbolic_recursion.graph.trajectory import load_events, render_trajectory
        out = render_trajectory(load_events(), window=args.window)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out)
            print("Trajectory written to", args.out)
        else:
            print(out, end="")
        return

    smc = load_smc()
    analysis = analyze_field(
        smc,
        include_symbol_edges=not args.no_symbol_edges,
        resolution=args.resolution,
    )
    report = render_report(smc, analysis)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print("Report written to", args.out)
    else:
        print(report, end="")

def cmd_pursue(args):
    from symbolic_recursion.core.pursue import plan_bridge, plan_deepen, execute

    smc = load_smc()
    if args.motif:
        plan = plan_deepen(smc, args.motif)
        missing = f"motif {args.motif} not found"
    else:
        plan = plan_bridge(smc)
        missing = "no open, unclaimed surprising cross-community connection to pursue"
    if plan is None:
        print(missing)
        return
    print(f"[{plan.kind}] thread={plan.thread_name} targets={plan.targets} owner={agent_id()}")
    if args.dry_run:
        print("--- prompt the model would receive ---")
        print(plan.prompt)
        return
    if args.stub:
        import symbolic_recursion.core.ollama_interface as oi
        from symbolic_recursion.core.model_stub import stub_response
        import symbolic_recursion.threads.manager as tmgr
        tmgr.query_ollama = lambda prompt, model=args.model, timeout=None: stub_response(prompt, model, timeout)
    review_cfg = None
    if args.review:
        review_cfg = {"enabled": True}
        if args.review_model:
            review_cfg["model"] = args.review_model
    tm = ThreadManager(smc)
    result = execute(smc, tm, plan, model=args.model, review_cfg=review_cfg)
    save_smc(smc)
    event = {"type": "pursuit", "kind": result.kind,
             "motif_id": result.motif_id, "targets": result.targets}
    if result.review:
        event["review"] = result.review
    _journal(smc, event)
    if result.review and result.review != "accept":
        print(f"captured {result.motif_id} — review: {result.review} "
              f"({result.review_evidence}); links withheld, surprise stays open")
    else:
        suffix = " (review: accept)" if result.review else ""
        print(f"captured {result.motif_id} linked -> {result.targets}{suffix}")

def main():
    p = argparse.ArgumentParser(description="Symbolic Memory Core CLI")
    p.add_argument(
        "--router",
        type=str,
        default=os.getenv("SMC_ROUTER", ""),
        help="Vector backend: 'vector', 'chroma', or blank for legacy",
    )
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

    p_pur = sub.add_parser("pursue", help="Fire one pursuit: bridge the top surprise, or deepen a motif")
    p_pur.add_argument("--motif", type=str, help="Motif id to deepen (default: bridge the top surprise)")
    p_pur.add_argument("--model", type=str, default="llama3:instruct")
    p_pur.add_argument("--stub", action="store_true", help="Use the deterministic model stub (no Ollama)")
    p_pur.add_argument("--dry-run", action="store_true", help="Print the assembled prompt, change nothing")
    p_pur.add_argument("--review", action="store_true", help="Reviewer reads the capture; links tied only on accept")
    p_pur.add_argument("--review-model", type=str, help="Different model for the reviewer (default: generator model)")
    p_pur.set_defaults(func=cmd_pursue)

    p_cap = sub.add_parser("capture", help="Capture a structured document as a motif subgraph")
    p_cap.add_argument("file", type=str, help="Path to a markdown/text document")
    p_cap.add_argument("--title", type=str, help="Document title (default: frontmatter topic or first heading)")
    p_cap.add_argument("--thread", type=str, help="Thread id (default: slug of title)")
    p_cap.add_argument("--prefix", type=str, help="Motif id prefix (default: slug of title)")
    p_cap.add_argument("--dry-run", action="store_true", help="Print the capture plan, change nothing")
    p_cap.set_defaults(func=cmd_capture)

    p_cl = sub.add_parser("claims", help="List active pursuit claims (who is working on what)")
    p_cl.add_argument("--release", type=str, metavar="KEY", help="Release this session's claim on KEY")
    p_cl.add_argument("--kind", type=str, default="bridge", help="Claim kind for --release (bridge|deepen)")
    p_cl.set_defaults(func=cmd_claims)

    p_tr = sub.add_parser("trace", help="Show the flow of text around one motif (prompt, response, lineage)")
    p_tr.add_argument("motif_id", type=str)
    p_tr.add_argument("--short", action="store_true", help="Excerpt prompt/response instead of full text")
    p_tr.set_defaults(func=cmd_trace)

    p_rep = sub.add_parser("report", help="Motif field report: communities, god motifs, surprises")
    p_rep.add_argument("--out", type=str, help="Write markdown to this path instead of stdout")
    p_rep.add_argument("--resolution", type=float, default=1.0, help=">1.0 more/smaller communities")
    p_rep.add_argument("--no-symbol-edges", action="store_true", help="Use only explicit references")
    p_rep.add_argument("--trajectory", action="store_true", help="Render the trajectory panel instead of the field report")
    p_rep.add_argument("--window", type=int, default=8, help="Trajectory window (events)")
    p_rep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
