# mini-agi

Symbolic Memory Core (SMC) — a small, local-first experiment in a memory
that grows a graph of ideas and then lets that graph drive its own further
exploration through a local language model.

## What this is, in plain terms

Picture a cork board. Every interesting thought gets written on a card,
tagged with a few sticker words, and pinned up. Related cards are tied
together with string. A local model (via [Ollama](https://ollama.ai/)) can
write new cards when asked; you can also pin cards by hand or paste in a
whole document and have it cut into cards automatically.

Step back and the board shows structure: cards cluster into neighborhoods,
some cards have far more strings than others, and now and then a string runs
clear across the board between two cards that look unrelated. The system
notices those long strings and asks the model: *what bigger idea joins these
two?* The answer becomes a new card, pinned between them. So the board starts
asking its own questions.

Three roles keep this honest, and it matters which is which:

| Role | Decides | How |
|---|---|---|
| **Scores** | *where to look* | structural heuristics: community membership, thread crossing, symbol overlap, edge age. They never read the text. |
| **Model** | *what to say* | fills a human-written prompt template with the field's specifics. |
| **Reviewer** | *whether it held up* | the one component that reads content. Ties the strings only if the synthesis genuinely covers both cards. |

A human remains the steward: the trajectory panel names the direction the
field is taking and shows its evidence, but never steers.

## Project structure

```
src/symbolic_recursion/
  core/
    motif.py            MotifNode + SymbolicMemoryCore (in-memory graph, revisions, links)
    storage.py          JSON persistence (data/motifs.json), merge-on-save, locked
    agent.py            session identity (SMC_AGENT) stamped into every record
    claims.py           claims ledger: which owner is pursuing which target
    router.py           term-frequency similarity + link suggestion (no deps)
    vector_router.py    SBERT/FAISS/numpy vector search, legacy sparse fallback
    chroma_router.py    persistent vector search via Chroma
    router_factory.py   RouterProtocol + backend selection (SMC_ROUTER)
    ollama_interface.py HTTP (localhost:11434) with CLI fallback
    model_stub.py       deterministic offline stand-in for the model
    pursue.py           the pursue step: bridge a surprise / deepen a motif
    review.py           the reviewer: content-aware gate on pursuit links
    flow.py             flow ledger (full prompt/response text) + trace
    runtime_policy.py   link/dedup/novelty thresholds for multi_run
  graph/
    analytics.py        Louvain communities, god motifs, surprises, field metrics
    trajectory.py       append-only journal + regime classifier
  documents/
    capture.py          markdown document -> motif subgraph (hub + sections)
    loader/chunker/indexer   RAG-style chunk index over local text
  skills/knowledge_search.py  unified ranking across motifs and chunks
  embeddings/           term-frequency embedder; optional SBERT wrapper
  threads/manager.py    ChatThread / ThreadManager (ask -> capture)
  utils/                novelty index, confidence heuristic, context rendering
  experiments/          novelty loop, multi-thread runner, reports, scenarios
scripts/run_cli.py      command-line interface
scripts/motif_to_chroma.py   index existing motifs into Chroma
tests/                  pytest suite (72 tests, offline)
```

Persisted state lives under `data/`:

| File | What it holds |
|---|---|
| `motifs.json` | the motif graph (the board itself) |
| `doc_registry.json` | document title → hub motif id, for `[[wiki-link]]` resolution |
| `trajectory.jsonl` | one line per field mutation: event + metrics after it |
| `flow.jsonl` | one line per generation: full prompt, response, provenance, review |
| `claims.jsonl` | which owner is working on which pursuit target, with expiry |
| `motif_report.md` | last rendered field report |
| `chroma/` | Chroma persistence, when that router is used |

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
```

Optional extras: `sbert` (sentence-transformers), `faiss`, `chroma`.
Only `numpy` is required; the graph analytics, capture, pursue, review and
trajectory modules are standard library only.

Run the tests:

```bash
pytest -q
```

If the package is not installed in editable mode, prefix with
`PYTHONPATH=src`. Tests redirect every ledger to a temp directory, so they
never touch `data/`.

## How a cycle flows

1. **Capture.** Text becomes a motif: from a model reply (`chat`,
   `pursue`, the novelty loop), by hand (`add`), or from a structured
   document (`capture`).
2. **Link.** Explicit references are recorded per motif. In the loop, new
   motifs are auto-linked to their nearest neighbors above a similarity
   threshold.
3. **Analyze.** The field is rebuilt as a weighted graph: explicit
   references (weight 1.0) plus implicit shared-symbol edges (0.5 × Jaccard).
   Louvain partitions it into communities; cross-community edges are scored
   for surprise.
4. **Pursue.** The top open surprise, damped by recency, becomes a bridge
   prompt. High-novelty motifs from the queue become deepen prompts.
5. **Review** (opt-in). The reviewer reads the capture against its targets
   and returns accept / revise / reject with evidence. Links are tied only on
   accept.
6. **Journal.** Every mutation appends to the trajectory; every generation
   appends to the flow ledger. The regime classifier reads the recent window.

## CLI reference

All commands read and write `data/motifs.json` in the current directory
(`SMC_DATA_PATH` overrides).

```bash
# board maintenance
python scripts/run_cli.py add --symbols justice,gradient --content "Justice as fairness across contexts"
python scripts/run_cli.py list
python scripts/run_cli.py link <a-id> <b-id>
python scripts/run_cli.py query --text "fairness across contexts" --k 5
python scripts/run_cli.py suggest <motif-id> --k 5

# generation
python scripts/run_cli.py chat --prompt "Briefly outline Justice as a Gradient" --capture justice,gradient
python scripts/run_cli.py capture note.md --dry-run
python scripts/run_cli.py pursue --dry-run
python scripts/run_cli.py pursue --review --review-model qwen2.5

# inspection
python scripts/run_cli.py report
python scripts/run_cli.py report --trajectory --window 8
python scripts/run_cli.py trace <motif-id>
python scripts/run_cli.py claims
```

Pass `--router vector` or `--router chroma` (or set `SMC_ROUTER`) to use a
vector backend for `query` and `suggest`; blank uses the term-frequency
baseline. The Ollama interface prints each request and response to stdout by
default (its `debug` flag).

## Document capture

Structured intake: a markdown document becomes a motif subgraph. A hub motif
holds the title and preamble; each heading becomes one motif with its text
kept verbatim. Every section references the hub (the document's axis) and
its predecessor (the reading sequence). Symbols are derived honestly: only
frontmatter tags that actually occur in the section, plus the heading's own
words. `[[wiki-links]]` resolve to the hubs of previously captured documents
via `data/doc_registry.json`, so declared cross-document ancestry becomes
explicit edges on arrival.

```bash
python scripts/run_cli.py capture note.md --dry-run   # show the plan, change nothing
python scripts/run_cli.py capture note.md --prefix stw --thread stewardship
```

A document without headings becomes a single motif. Unresolved links are
reported so you know which documents to capture next.

## Motif field report

Graph analytics over the motif field: community detection (pure-Python
Louvain), god motifs (dominant attractors), surprising cross-community
connections, and the Project-vision metrics (recurrence rate, symbolic depth,
narrative binding, symbol drift). Deterministic: same field in, same report out.

```bash
python scripts/run_cli.py report                   # print markdown report
python scripts/run_cli.py report --out data/motif_report.md
python scripts/run_cli.py report --resolution 1.5  # more, smaller communities
python scripts/run_cli.py report --no-symbol-edges # explicit references only
```

Surprise scoring is additive and transparent: +2 for an explicit reference
crossing communities (+1 for a shared-symbol edge), +1 for crossing threads,
up to +1 for symbol disjointness, +0.5 for a peripheral motif reaching a hub.
Library API: `symbolic_recursion.graph.analyze_field(smc)` / `render_report(smc)`.

## Pursue step

Motifs initiating their own follow-up exploration. You supply the epistemic
move once as a template; the field fills in the specifics (motifs, symbols,
threads, a context block of graph neighbors) per firing.

- **bridge** fires on the top open surprising connection: *what higher
  abstraction binds A and B?*
- **deepen** fires on a high-novelty motif from the loop's pursue queue:
  *develop this one level of abstraction higher.*

```bash
python scripts/run_cli.py pursue --dry-run          # show the prompt for the top surprise
python scripts/run_cli.py pursue                    # bridge pursuit via Ollama
python scripts/run_cli.py pursue --motif <id>       # deepen one motif
python scripts/run_cli.py pursue --stub             # deterministic stub model (no Ollama)
```

Target selection explores the frontier, not its own tail. Surprises are
damped by the age of their younger endpoint (`recency_half_life_hours`,
default 24: a fresh edge must season before it can be pursued), and pairs
already bridged by a capture referencing both endpoints are skipped as
resolved. Captures land in a new `pursue-*` thread.

In the novelty loop, enable per scenario (off by default):

```json
"pursue": { "enabled": true, "max_per_cycle": 1,
            "recency_half_life_hours": 24,
            "templates": { "bridge": "...{a_symbols}...{b_symbols}..." },
            "review": { "enabled": true, "model": "qwen2.5" } }
```

Bridge templates may use `{a_symbols} {b_symbols} {a_thread} {b_thread}
{reasons}`; deepen templates `{content} {symbols} {thread}`.

## Reviewer

The one component that reads text instead of structure. With `--review`, a
pursuit capture is judged on content: does it name an abstraction that
genuinely covers its targets, or restate them with the sticker words swapped
in? The verdict is `accept`, `revise` or `reject` with one line of evidence
and the extracted testable prediction. Never a scalar: a reason can be argued
with.

The gate ties the **link**, not the capture. The card stays in the field and
the flow ledger either way, but references to the targets are made only on
accept. A rejected bridge therefore leaves the surprise open for a better
attempt. An unparseable reviewer reply degrades to `revise`.

```bash
python scripts/run_cli.py pursue --review                         # same model reviews
python scripts/run_cli.py pursue --review --review-model qwen2.5  # second opinion
```

Use a different model for the reviewer where you can. A model asked to grade
its own synthesis will mostly approve it.

## Trajectory panel

An instrument, not a fence. Every capture, link, pursuit and loop cycle
appends a line to `data/trajectory.jsonl` with the field metrics after it.
The regime classifier reads the recent window and names the direction,
always with its evidence:

| Regime | Reading |
|---|---|
| `fragmenting` | communities up, binding down: accumulating without integrating |
| `stagnating` | field grew with no structural consequence |
| `concentrating` | pursuits locked onto one community: the obsession signature |
| `thrashing` | pursuits fire but the reviewer withholds most links |
| `oscillating (healthy)` | communities split and merge; binding dips on capture, recovers on pursuit |
| `consolidating` | binding rising, communities merging; watch for slide into stagnation |
| `idle`, `mixed`, `insufficient-history` | as named |

```bash
python scripts/run_cli.py report --trajectory --window 8
```

## Flow ledger and trace

The trajectory records numbers; the flow ledger records text. Every
generation appends the full prompt the model saw (context block included),
the response, the provenance, and the review verdict when there was one.
`trace` reconstructs the flow around one motif: what it grew from, what it
was shown, what it said, and what grew out of it.

```bash
python scripts/run_cli.py trace <motif-id>
python scripts/run_cli.py trace <motif-id> --short   # excerpts instead of full text
```

## Chroma persistence

SMC can persist vector search data using [Chroma](https://www.trychroma.com/).

1. Install the optional dependencies (SBERT gives better embeddings):

   ```bash
   pip install chromadb sentence-transformers
   ```

2. Optionally index existing motifs:

   ```bash
   python scripts/motif_to_chroma.py
   ```

3. Select the router by environment variable or flag:

   ```bash
   SMC_ROUTER=chroma python scripts/run_cli.py query --text "hello world"
   python scripts/run_cli.py --router chroma query --text "hello world"
   ```

Chroma data is stored under `data/chroma/`. Without SBERT, both vector
routers fall back to the term-frequency embedder packed into a dense space.

## Experiments

**Novelty loop.** Runs the scenario prompts through the model each cycle,
captures and auto-links the replies, scores each for novelty, optionally
pursues, then records metrics and journals the cycle.

```bash
python -m symbolic_recursion.experiments.run_loop_novelty scenario.json
python -c "from symbolic_recursion.experiments.report_novelty import main; main()"
```

The scenario is loaded by basename from the package's own `experiments/`
directory, so `scenario.json` refers to
`src/symbolic_recursion/experiments/scenario.json` regardless of the path
given. Per-cycle logs land in `experiments/logs/`; the semantic centroid is
carried between runs in `experiments/state.json`. Set `"use_stub": true` to
run without Ollama. The report module has no `__main__` guard, hence the
`-c` invocation.

**Multi-thread runner.** Asks several topics concurrently with a retrieved
context block, then applies a `Policy` (near-duplicate merge, minimum
novelty, capped fan-out) before persisting.

```bash
python -m symbolic_recursion.experiments.multi_run topics.json --workers 3 --router vector
```

**Analyze existing.** Prints graph and semantic metrics plus per-motif
novelty for the current field.

```bash
python -m symbolic_recursion.experiments.analyze_existing
```

## Document layer (RAG seed)

A lightweight retrieval layer that coexists with motif retrieval:
`documents.loader` reads local text, `documents.chunker` splits it
deterministically with overlap, `documents.indexer` holds an in-memory
sparse chunk index, and `skills.knowledge_search` ranks motifs and chunks
together, boosting chunks whose tags overlap the active motifs' symbols.

## Running from several sessions

Several sessions can work the same field at once: a person capturing
documents, a loop run, a second session reviewing. Three pieces make that
safe.

**Merge-on-save.** The store is one JSON file written whole. A save takes
an advisory lock, re-reads the file, merges, and writes atomically. Motifs
only on disk are kept, so another session's captures survive. Motifs
present on both sides take the saving session's content, with the reference
lists unioned so nobody's links are dropped. Deletion is not a persisted
operation. Pass `merge=False` to the save function to overwrite on purpose.

**Agent identity.** Set `SMC_AGENT` per session. It is stamped onto every
motif at capture, every trajectory journal line, every flow ledger entry,
and every pursuit and review. The trajectory panel shows an agent column
and `trace` names the agent in the generation record. Unset, everything is
`anonymous`.

**Claims.** Before a pursuit fires, its owner claims the target in
`data/claims.jsonl` for the length of the model call, then releases it.
The bridge planner skips pairs another owner currently holds, and the loop
skips queued motifs another owner is deepening. Claims expire after an hour
by default, so a crashed session cannot hold a target forever. If another
writer resolves the planned pair between planning and execution, the
pursuit is skipped rather than doubled.

```bash
SMC_AGENT=carsten python scripts/run_cli.py pursue --review
SMC_AGENT=loop     python -m symbolic_recursion.experiments.run_loop_novelty scenario.json
python scripts/run_cli.py claims                       # who is working on what
python scripts/run_cli.py claims --release a|b --kind bridge
```

Content edits still follow last-writer-wins, so two sessions revising the
same motif's text at the same moment will keep only one version. Links and
new motifs are never lost.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SMC_DATA_PATH` | `data/motifs.json` | motif store |
| `SMC_TRAJECTORY_PATH` | `data/trajectory.jsonl` | trajectory journal |
| `SMC_FLOW_PATH` | `data/flow.jsonl` | flow ledger |
| `SMC_DOC_REGISTRY` | `data/doc_registry.json` | captured-document registry |
| `SMC_CLAIMS_PATH` | `data/claims.jsonl` | pursuit claims ledger |
| `SMC_AGENT` | `anonymous` | this session's identity on every record |
| `SMC_ROUTER` | blank | `vector` or `chroma` |
| `SMC_MODEL` | `llama3:instruct` | model for `multi_run` |
| `SMC_EMB_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | SBERT model |
| `SMC_EMB_BATCH` | `32` | SBERT batch size |

## Known gaps

- The confidence heuristic in `utils/confidence.py` is computed nowhere in
  the current loop; the reviewer has taken over the quality role.
- `Project-vision.md` predates most of this and its roadmap is stale.

## License

MIT / Apache 2.0 (TBD)
