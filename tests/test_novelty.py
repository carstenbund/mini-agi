"""Novelty scoring must not let a motif match itself.

Both loop callers score a motif *after* it has been captured into the SMC.
Before the fix, the motif's own vector was in the prior corpus, so semantic
novelty was always ~0 and the combined index could never clear the pursue
threshold.
"""

import pytest

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.utils.novelty import (
    novelty_index,
    semantic_novelty,
    structural_novelty,
    symbolic_novelty,
)


def _motif(mid, symbols, content):
    return MotifNode(id=mid, symbols=list(symbols), content=content, thread_id="t")


def test_semantic_novelty_ignores_self_after_add(smc_with_motifs):
    m = _motif("new", ["quokka", "tide"], "Quokka photographs surge before the lunar tide")
    smc_with_motifs.add_motif(m)  # score AFTER add, like the loop does

    sem = semantic_novelty(smc_with_motifs, m)
    assert sem == pytest.approx(1.0, abs=1e-9)


def test_semantic_novelty_same_before_and_after_add(smc_with_motifs):
    m = _motif("new", ["coherence"], "Coherence emerges from repeated activation of motifs")
    before = semantic_novelty(smc_with_motifs, m)
    smc_with_motifs.add_motif(m)
    after = semantic_novelty(smc_with_motifs, m)
    assert after == pytest.approx(before)
    assert 0.0 < after < 1.0  # overlaps m3 but is not identical


def test_symbolic_and_structural_ignore_self(smc_with_motifs):
    m = _motif("new", ["quokka", "tide"], "Quokka photographs surge before the lunar tide")
    smc_with_motifs.add_motif(m)
    assert symbolic_novelty(smc_with_motifs, m) == pytest.approx(1.0)
    assert structural_novelty(smc_with_motifs, m) == pytest.approx(0.0)


def test_unrelated_motif_clears_default_pursue_threshold(smc_with_motifs):
    m = _motif("new", ["quokka", "tide"], "Quokka photographs surge before the lunar tide")
    smc_with_motifs.add_motif(m)
    n = novelty_index(smc_with_motifs, m)
    assert n["semantic"] == pytest.approx(1.0)
    assert n["novelty_index"] >= 0.55


def test_only_motif_in_graph_is_not_scored_against_itself():
    smc = SymbolicMemoryCore()
    m = _motif("solo", ["alpha"], "the only motif here")
    smc.add_motif(m)
    assert semantic_novelty(smc, m) == pytest.approx(1.0)
    assert structural_novelty(smc, m) == 0.0


def test_semantic_novelty_still_sees_real_overlap():
    """Excluding self must not exclude a genuine duplicate elsewhere."""
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a", ["fruit"], "apples and oranges"))
    dup = _motif("b", ["fruit"], "apples and oranges")
    smc.add_motif(dup)
    assert semantic_novelty(smc, dup) == pytest.approx(0.0)


def test_symbolic_novelty_counts_only_others_symbols():
    """Half of the new motif's symbols are already in the field."""
    smc = SymbolicMemoryCore()
    smc.add_motif(_motif("a", ["fruit"], "x"))
    new = _motif("b", ["zebra", "fruit"], "y")
    smc.add_motif(new)
    assert symbolic_novelty(smc, new) == pytest.approx(0.5)
