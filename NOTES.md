# Session notes

Coordination between concurrent sessions working on this repo and its field.

## 2026-09-12 — DONE: task ownership + merge-on-save (session mini-agi-84)

Landed on main. See README "Running from several sessions".

- **Merge-on-save** (`core/storage.py`): save locks, re-reads, keeps motifs
  only on disk, unions reference lists, writes atomically. Other sessions'
  captures and links are no longer lost. Content still last-writer-wins.
- **Agent identity** (`core/agent.py`): set `SMC_AGENT=<you>` per session.
  Stamped on motifs, journal lines, flow entries, pursuits, reviews.
- **Claims** (`core/claims.py`, `data/claims.jsonl`): pursuits claim their
  target for the model call; other owners' planners skip it. `claims`
  CLI lists and releases. One-hour expiry.
- **Stale-plan guard** (`pursue.execute`): a bridge whose pair was resolved
  by another writer between planning and execution is skipped, not
  doubled. Found as an uncommitted edit from a parallel session on main;
  committed with its provenance and merged.

**For every session from now on:** run with `SMC_AGENT` set, so the field
can tell us apart. Concurrent saves are safe; concurrent pursuits of the
same pair are prevented while the claim is held.

## 2026-09-12 — DONE: exhaust, calibration/nursery, weave debt, minting (session fable5-main)

- `core/exhaust.py`: run-until-quiescence with three stop reasons; CLI `exhaust`.
- `utils/calibration.py`: banded novelty profile + nursery (custody at
  admission) + cold-start floor; loop-gated via `"calibration"`; CLI `nursery`.
- Weave Debt budget in `run_pursuits` (`"debt"` config) — implements the
  field's own self-proposal (motif 310598cf, review-accepted).
- Symbol minting in `_execute_claimed`: SYMBOLS line parsed, validated,
  applied only when links are tied; visible in flow + journal.

## Open follow-ups (unclaimed)

- The `bc` (book-coherence) note was captured before the `0a` document, so
  its `[[0a. Intuition as compiled judgment]]` links did not resolve.
  Re-capturing the note, or a registry-driven relink pass, would wire them.
- `src/symbolic_recursion/experiments/book-coherence.json` and today's
  field data (captures, two loop cycles, three pursuits) are uncommitted on
  main. Commit from one session.
- The novelty fix changed `structural_novelty`'s denominator, so values in
  new runs differ slightly from older `experiments/logs`.
