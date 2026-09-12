#!/usr/bin/env python3
"""Autonomous field runner — clone the repo, point at an Ollama port, go.

    python3 scripts/autorun.py --host 127.0.0.1:11434 --model llama3:instruct

No dependencies beyond the standard library (the exhaust path never
imports numpy). One cycle:

    preflight  Ollama reachable, model present
    exhaust    pursue until settled / diminishing-returns / budget,
               reviewer gating every link (a different --review-model
               makes the reviewer a second opinion)
    self       one self-pursuit (--self): the field reads its own spec
               against its own trajectory and proposes a revision
    nursery    custody review: graduate what the field has bound to
    report     data/motif_report.md refreshed; summary printed

With --interval N the cycle repeats every N seconds — sensible values
are hours, not minutes: recency damping means fresh captures must season
before they can be pursued, so tight loops just hit 'settled'. Ctrl-C
stops cleanly. Every generation lands in the flow ledger, every mutation
in the trajectory journal; `scripts/run_cli.py trace <id>` shows any
capture's full textual genesis.

The carrier's part stays yours: proposals from --self cross into code
only through your judgment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from urllib import error, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def preflight(host: str, models: list) -> None:
    url = f"http://{host}/api/tags"
    try:
        opener = request.build_opener(request.ProxyHandler({}))
        with opener.open(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except (error.URLError, error.HTTPError, json.JSONDecodeError, OSError) as e:
        sys.exit(f"preflight failed: Ollama not reachable at {host} ({e})\n"
                 f"start it with: ollama serve   (or systemctl start ollama)")
    available = {m.get("name", "") for m in data.get("models", [])}
    for wanted in models:
        if not any(a == wanted or a.startswith(wanted + ":") or wanted.startswith(a.split(":")[0])
                   for a in available):
            sys.exit(f"preflight failed: model '{wanted}' not present at {host}\n"
                     f"available: {', '.join(sorted(available)) or '(none)'}\n"
                     f"pull it with: ollama pull {wanted}")
    print(f"preflight ok: {host} serving {len(available)} model(s)")


def bind_host(host: str) -> None:
    """Route every generation and review through the given Ollama host."""
    import symbolic_recursion.core.ollama_interface as oi
    import symbolic_recursion.threads.manager as tmgr
    orig = oi.query_ollama

    def bound(prompt, model="llama3:instruct", host_=host, timeout=None, debug=False):
        return orig(prompt, model=model, host=host_, timeout=timeout, debug=debug)

    oi.query_ollama = bound
    tmgr.query_ollama = bound


def one_cycle(args) -> str:
    from symbolic_recursion.core.exhaust import run_exhaust
    from symbolic_recursion.core.motif import SymbolicMemoryCore
    from symbolic_recursion.core.pursue import plan_self, execute
    from symbolic_recursion.core.storage import load_motifs, save_motifs
    from symbolic_recursion.graph import analyze_field, render_report
    from symbolic_recursion.graph.trajectory import classify_regime, load_events, record_event
    from symbolic_recursion.threads.manager import ThreadManager
    from symbolic_recursion.utils.calibration import nursery_pass

    smc = SymbolicMemoryCore()
    smc.motifs = load_motifs()
    tm = ThreadManager(smc)
    review_cfg = {"enabled": True, "model": args.review_model or args.model}
    goal = (args.goal_thread,) if args.goal_thread else None

    def on_fire(result):
        save_motifs(smc.motifs)
        event = {"type": "pursuit", "kind": result.kind,
                 "motif_id": result.motif_id, "targets": result.targets}
        if result.review:
            event["review"] = result.review
        if result.minted:
            event["minted"] = result.minted
        record_event(smc, event)
        tail = f" review={result.review}" if result.review else ""
        tail += f" minted={','.join(result.minted)}" if result.minted else ""
        print(f"  [{datetime.now():%H:%M:%S}] {result.kind}: {result.motif_id}"
              f" -> {result.targets}{tail}")

    report = run_exhaust(
        smc, tm, model=args.model,
        cfg={"max_pursuits": args.max_pursuits, "min_score": args.min_score,
             "patience": args.patience},
        review_cfg=review_cfg, goal_threads=goal, on_fire=on_fire)
    save_motifs(smc.motifs)
    record_event(smc, {"type": "exhaust-stop", "reason": report.stop_reason,
                       "fired": report.fired})
    print(f"exhaust: {report.stop_reason} after {report.fired} pursuit(s)")

    if args.self_pursuit:
        spec = (args.spec_thread,) if args.spec_thread else None
        plan = plan_self(smc, **({"spec_threads": spec} if spec else {}))
        if plan is not None:
            result = execute(smc, tm, plan, model=args.model, review_cfg=review_cfg)
            save_motifs(smc.motifs)
            if not result.skipped:
                on_fire(result)
        else:
            print("self: no unpursued spec motif")

    for row in nursery_pass(smc):
        if row["status"] != "held":
            print(f"nursery: {row['status']} {row['motif']}")

    analysis = analyze_field(smc)
    report_path = os.environ.get("SMC_REPORT_PATH", os.path.join("data", "motif_report.md"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(smc, analysis))
    verdict = classify_regime(load_events())
    met = analysis["metrics"]
    print(f"field: {met['motif_count']} motifs, {met['community_count']} communities, "
          f"binding {met['narrative_binding']} | regime: {verdict['regime']}")
    return report.stop_reason


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.getenv("SMC_OLLAMA_HOST", "127.0.0.1:11434"))
    p.add_argument("--model", default=os.getenv("SMC_MODEL", "llama3:instruct"))
    p.add_argument("--review-model", default=os.getenv("SMC_REVIEW_MODEL"),
                   help="Second-opinion reviewer model (default: same as --model)")
    p.add_argument("--goal-thread", help="Pull pursuit selection toward this thread's vocabulary")
    p.add_argument("--spec-thread", help="Spec thread for the self-pursuit (default: inherited-judgment)")
    p.add_argument("--self", dest="self_pursuit", action="store_true",
                   help="One self-pursuit per cycle: the field proposes its own revision")
    p.add_argument("--max-pursuits", type=int, default=8)
    p.add_argument("--min-score", type=float, default=0.02)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--interval", type=int, default=0,
                   help="Seconds between cycles; 0 runs once. Use hours — fresh captures must season.")
    p.add_argument("--agent", default=os.getenv("SMC_AGENT", "autorun"))
    args = p.parse_args()

    os.environ["SMC_AGENT"] = args.agent
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    models = [args.model] + ([args.review_model] if args.review_model else [])
    preflight(args.host, models)
    bind_host(args.host)

    while True:
        print(f"=== cycle @ {datetime.now():%Y-%m-%d %H:%M:%S} "
              f"(agent {args.agent}, model {args.model}) ===")
        try:
            one_cycle(args)
        except KeyboardInterrupt:
            raise
        if args.interval <= 0:
            break
        print(f"sleeping {args.interval}s (Ctrl-C to stop)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
            break


if __name__ == "__main__":
    main()
