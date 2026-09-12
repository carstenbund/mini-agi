"""Merge-on-save: two sessions saving the same store must not lose each
other's captures. conftest points SMC_DATA_PATH at a tmp file."""
import os

from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.storage import (
    data_path, load_motifs, merge_payloads, save_motifs,
)


def _m(mid, refs=None, content=None):
    return MotifNode(id=mid, symbols=["s"], content=content or f"content {mid}",
                     thread_id="t", references=list(refs or []))


def _seed(*ids):
    smc = SymbolicMemoryCore()
    for i in ids:
        smc.add_motif(_m(i))
    save_motifs(smc.motifs)
    return smc


def _session():
    smc = SymbolicMemoryCore()
    smc.motifs = load_motifs()
    return smc


def test_concurrent_sessions_keep_each_others_motifs():
    _seed("m0")
    a, b = _session(), _session()       # both loaded the same store
    a.add_motif(_m("a1"))
    b.add_motif(_m("b1"))
    save_motifs(a.motifs)
    save_motifs(b.motifs)               # would have dropped a1 before
    assert set(load_motifs()) == {"m0", "a1", "b1"}


def test_reference_lists_are_unioned():
    _seed("m0", "m1", "m2")
    a, b = _session(), _session()
    a.link_motifs("m0", "m1")
    b.link_motifs("m0", "m2")
    save_motifs(a.motifs)
    save_motifs(b.motifs)
    assert set(load_motifs()["m0"].references) == {"m1", "m2"}


def test_last_writer_wins_on_content():
    _seed("m0")
    a, b = _session(), _session()
    a.update_motif("m0", content="from a")
    b.update_motif("m0", content="from b")
    save_motifs(a.motifs)
    save_motifs(b.motifs)
    assert load_motifs()["m0"].content == "from b"


def test_merge_can_be_disabled():
    _seed("m0", "m1")
    only = SymbolicMemoryCore()
    only.add_motif(_m("x"))
    save_motifs(only.motifs, merge=False)
    assert set(load_motifs()) == {"x"}


def test_save_leaves_no_scratch_file():
    _seed("m0")
    path = data_path()
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_merge_payloads_is_pure_and_prefers_mine():
    mine = {"a": {"content": "mine", "references": ["x"]}}
    disk = {"a": {"content": "disk", "references": ["y", "x"]},
            "b": {"content": "only on disk", "references": []}}
    out = merge_payloads(mine, disk)
    assert out["a"]["content"] == "mine"
    assert out["a"]["references"] == ["x", "y"]
    assert out["b"]["content"] == "only on disk"
    assert mine["a"]["references"] == ["x"]      # input untouched
