"""Claims ledger — which owner is working on which pursuit target.

Two sessions pursuing the same open surprise waste a model call and
produce a duplicate bridge. Before a pursuit executes, its owner records
a claim on the target; target selection skips targets claimed by another
owner until the claim is released or expires.

Append-only JSONL at ``data/claims.jsonl`` (``SMC_CLAIMS_PATH`` overrides),
same convention as the other ledgers. Every record has ``kind``
(``bridge`` | ``deepen``), ``key`` (a sorted pair ``a|b`` for a bridge, a
motif id for a deepen), ``owner``, ``ts``, ``expires`` and ``released``.
The latest record per (kind, key) wins, so a release is just one more
line. Expiry keeps a crashed session from holding a target forever.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

DEFAULT_TTL_HOURS = 1.0


def claims_path() -> str:
    return os.path.abspath(
        os.environ.get("SMC_CLAIMS_PATH", os.path.join("data", "claims.jsonl"))
    )


def pair_key(a: str, b: str) -> str:
    """Order-independent key for a bridge pair."""
    return "|".join(sorted((a, b)))


def target_key(targets: List[str]) -> str:
    """Key for a pursuit's targets: pair key for two, the id for one."""
    if len(targets) == 2:
        return pair_key(targets[0], targets[1])
    return "|".join(targets)


def _append(rec: Dict, path: Optional[str]) -> Dict:
    path = path or claims_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_claims(path: Optional[str] = None) -> List[Dict]:
    path = path or claims_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def claim(
    kind: str,
    key: str,
    owner: str,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    now: Optional[datetime] = None,
    path: Optional[str] = None,
) -> Dict:
    """Record that ``owner`` is working on (kind, key) until the claim is
    released or ``ttl_hours`` pass."""
    now = now or datetime.utcnow()
    return _append({
        "ts": now.isoformat(),
        "kind": kind,
        "key": key,
        "owner": owner,
        "expires": (now + timedelta(hours=ttl_hours)).isoformat(),
        "released": False,
    }, path)


def release(
    kind: str,
    key: str,
    owner: str,
    now: Optional[datetime] = None,
    path: Optional[str] = None,
) -> Dict:
    """Record that ``owner`` is done with (kind, key)."""
    now = now or datetime.utcnow()
    return _append({
        "ts": now.isoformat(),
        "kind": kind,
        "key": key,
        "owner": owner,
        "expires": now.isoformat(),
        "released": True,
    }, path)


def active_claims(
    now: Optional[datetime] = None, path: Optional[str] = None
) -> Dict[Tuple[str, str], Dict]:
    """{(kind, key): latest record} for claims neither released nor expired."""
    now = now or datetime.utcnow()
    latest: Dict[Tuple[str, str], Dict] = {}
    for r in load_claims(path):
        latest[(r.get("kind", ""), r.get("key", ""))] = r
    out: Dict[Tuple[str, str], Dict] = {}
    for k, r in latest.items():
        if r.get("released"):
            continue
        try:
            expires = datetime.fromisoformat(r["expires"])
        except (KeyError, TypeError, ValueError):
            continue
        if expires > now:
            out[k] = r
    return out


def claimed_by_other(
    kind: str,
    key: str,
    owner: str,
    now: Optional[datetime] = None,
    path: Optional[str] = None,
    active: Optional[Dict[Tuple[str, str], Dict]] = None,
) -> Optional[str]:
    """The other owner currently holding (kind, key), or None. Pass
    ``active`` (from ``active_claims``) to avoid re-reading the ledger."""
    if active is None:
        active = active_claims(now, path)
    r = active.get((kind, key))
    if r and r.get("owner") != owner:
        return r.get("owner")
    return None
