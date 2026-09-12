"""Motif store persistence — one JSON file, safe for several sessions.

The store is written whole on every save. Before this module merged, two
sessions saving within the same moment silently lost one side's captures:
whoever wrote last won. Now a save takes an advisory lock, re-reads what
is on disk, merges, and writes atomically:

- motifs only on disk are kept (another session's captures)
- motifs in both take THIS session's content and symbols (last writer
  wins on content, which is the honest answer for a revision), with the
  reference lists unioned so nobody's links are dropped
- deletion is not a persisted operation: a motif removed from memory but
  still on disk comes back

``merge=False`` restores plain overwrite for callers that really mean it.
"""
from __future__ import annotations

import json
import os
from typing import Dict

from symbolic_recursion.core.motif import MotifNode

try:  # POSIX advisory lock; absent on Windows, where we fall back to no lock
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def data_path() -> str:
    """Resolve the motif store path.

    ``SMC_DATA_PATH`` env var wins; otherwise ``data/motifs.json`` relative
    to the current working directory — the same convention as the Chroma
    persist dir (``data/chroma``) and the README. Resolved at call time so
    env/cwd changes take effect without re-import.

    (Previously this was resolved relative to the package directory, which
    silently split the store between ``data/`` and ``src/.../data/``.)
    """
    return os.path.abspath(
        os.environ.get("SMC_DATA_PATH", os.path.join("data", "motifs.json"))
    )


class _StoreLock:
    """Exclusive advisory lock on ``<store>.lock`` for the read-merge-write."""

    def __init__(self, store_path: str):
        self.path = store_path + ".lock"
        self.fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
        return False


def _read_raw(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_payloads(mine: Dict[str, dict], on_disk: Dict[str, dict]) -> Dict[str, dict]:
    """Pure merge of two serialized stores; see the module docstring."""
    out: Dict[str, dict] = {mid: dict(d) for mid, d in mine.items()}
    for mid, disk in on_disk.items():
        if mid not in out:
            out[mid] = disk
            continue
        refs = list(out[mid].get("references", []))
        for r in disk.get("references", []):
            if r not in refs:
                refs.append(r)
        out[mid]["references"] = refs
    return out


def save_motifs(motif_map: Dict[str, MotifNode], merge: bool = True) -> None:
    path = data_path()
    payload = {mid: m.to_dict() for mid, m in motif_map.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _StoreLock(path):
        if merge:
            payload = merge_payloads(payload, _read_raw(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)


def load_motifs() -> Dict[str, MotifNode]:
    raw = _read_raw(data_path())
    return {k: MotifNode.from_dict(v) for k, v in raw.items()}
