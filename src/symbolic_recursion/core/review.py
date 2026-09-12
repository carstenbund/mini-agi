"""The reviewer — the one component that reads text instead of structure.

Every existing score (surprise, recency, novelty, field metrics) is
blind to content. The reviewer actually reads a pursuit capture and its
targets and answers the only question the scores cannot: does this name
an abstraction that genuinely covers both, or is it a restatement with
the sticker words swapped in?

Principles (mirroring the trajectory panel's):

- verdict with reasons, not a scalar: ``accept`` / ``revise`` /
  ``reject`` plus one line of evidence, plus the testable prediction
  extracted when present. A reason can be argued with; a 0.62 cannot.
- gate the LINK, not the capture: the card stays in the field and the
  flow ledger either way, but the strings to the targets are only tied
  on ``accept`` — so a rejected bridge does not resolve the surprise,
  and the pair stays an open question for a better attempt.
- feed the journal: the verdict travels with the pursuit event, so the
  regime classifier can see acceptance rate (a field where pursuits
  fire constantly but are mostly rejected is its own failure mode:
  ``thrashing``).

The reviewer may be a DIFFERENT model than the generator (config
``review.model``); an unparseable verdict degrades to ``revise`` —
conservative: capture kept, links withheld, question left open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from symbolic_recursion.core.motif import SymbolicMemoryCore

REVIEW_TEMPLATE = """You are reviewing a synthesis produced inside a symbolic memory field.

{targets_block}

## Candidate synthesis
{candidate}

## Your task
Judge the synthesis on content, not form. Does it name an abstraction that
genuinely covers the target motif(s) above — something that could not be
produced by restating either target with its vocabulary swapped? Ignore
eloquence; look for an actual claim that subsumes both and adds a
consequence neither target already states.

Answer in exactly this format (three lines):
VERDICT: accept | revise | reject
EVIDENCE: <one line naming the strongest reason for your verdict>
PREDICTION: <the testable prediction the synthesis makes, extracted verbatim or tightly paraphrased; or "none">"""


@dataclass
class ReviewResult:
    verdict: str                 # "accept" | "revise" | "reject"
    evidence: str
    prediction: str
    raw: str
    parse_ok: bool


def build_review_prompt(smc: SymbolicMemoryCore, candidate_text: str,
                        target_ids: List[str]) -> str:
    blocks = []
    for i, tid in enumerate(target_ids, 1):
        m = smc.get_motif(tid)
        if m is None:
            continue
        blocks.append(f"## Target {i} — [{', '.join(m.symbols)}] (thread {m.thread_id})\n"
                      f"{m.content.strip()}")
    return REVIEW_TEMPLATE.format(
        targets_block="\n\n".join(blocks) or "## Targets\n(none in field)",
        candidate=candidate_text.strip(),
    )


def parse_review(raw: str) -> ReviewResult:
    """Parse the three-line verdict; degrade to ``revise`` when unparseable."""
    verdict_m = re.search(r"VERDICT:\s*(accept|revise|reject)", raw, re.IGNORECASE)
    evidence_m = re.search(r"EVIDENCE:\s*(.+)", raw)
    prediction_m = re.search(r"PREDICTION:\s*(.+)", raw)
    if verdict_m:
        return ReviewResult(
            verdict=verdict_m.group(1).lower(),
            evidence=(evidence_m.group(1).strip() if evidence_m else ""),
            prediction=(prediction_m.group(1).strip() if prediction_m else "none"),
            raw=raw, parse_ok=True,
        )
    return ReviewResult(verdict="revise",
                        evidence="reviewer output unparseable; links withheld",
                        prediction="none", raw=raw, parse_ok=False)


def review_capture(
    smc: SymbolicMemoryCore,
    candidate_text: str,
    target_ids: List[str],
    model: str,
    query_fn: Optional[Callable] = None,
) -> ReviewResult:
    """Run the reviewer. ``query_fn(prompt, model)`` may be injected; the
    default goes through the same Ollama interface as generation — pass a
    different ``model`` to make the reviewer a second opinion."""
    prompt = build_review_prompt(smc, candidate_text, target_ids)
    if query_fn is None:
        from symbolic_recursion.core.ollama_interface import query_ollama
        query_fn = lambda p, m: query_ollama(p, model=m)
    raw = query_fn(prompt, model)
    return parse_review(raw)
