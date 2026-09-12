"""Claims ledger and its use by target selection."""
from datetime import datetime, timedelta

import symbolic_recursion.threads.manager as tmgr
from symbolic_recursion.core.claims import (
    active_claims, claim, claimed_by_other, load_claims, pair_key, release, target_key,
)
from symbolic_recursion.core.model_stub import stub_response
from symbolic_recursion.core.pursue import execute, plan_bridge, run_pursuits
from symbolic_recursion.threads.manager import ThreadManager
from tests.test_pursue import _bridged_field


def _stub(monkeypatch):
    monkeypatch.setattr(tmgr, "query_ollama",
                        lambda prompt, model="stub", timeout=None: stub_response(prompt, model, timeout))


def test_keys_are_order_independent():
    assert pair_key("b", "a") == pair_key("a", "b") == "a|b"
    assert target_key(["b", "a"]) == "a|b"
    assert target_key(["solo"]) == "solo"


def test_claim_is_active_until_released_or_expired():
    now = datetime(2026, 9, 12, 12, 0, 0)
    claim("bridge", "a|b", "alice", ttl_hours=1.0, now=now)
    assert ("bridge", "a|b") in active_claims(now=now + timedelta(minutes=30))
    assert ("bridge", "a|b") not in active_claims(now=now + timedelta(hours=2))
    release("bridge", "a|b", "alice", now=now + timedelta(minutes=5))
    assert ("bridge", "a|b") not in active_claims(now=now + timedelta(minutes=10))


def test_claimed_by_other_ignores_own_claim():
    now = datetime(2026, 9, 12, 12, 0, 0)
    claim("bridge", "a|b", "alice", now=now)
    assert claimed_by_other("bridge", "a|b", "alice", now=now) is None
    assert claimed_by_other("bridge", "a|b", "bob", now=now) == "alice"
    assert claimed_by_other("deepen", "a|b", "bob", now=now) is None   # kind matters


def test_plan_bridge_skips_pair_claimed_by_another_owner():
    smc = _bridged_field()
    top = plan_bridge(smc, owner="me")
    assert set(top.targets) == {"a3", "b1"}
    claim("bridge", target_key(top.targets), "someone-else")
    other = plan_bridge(smc, owner="me")
    assert other is None or set(other.targets) != {"a3", "b1"}
    # my own claim does not block me; nor does anyone's when claims are ignored
    claim("bridge", target_key(top.targets), "me")
    assert set(plan_bridge(smc, owner="me").targets) == {"a3", "b1"}
    assert set(plan_bridge(smc, owner="bob", respect_claims=False).targets) == {"a3", "b1"}


def test_execute_claims_during_and_releases_after(monkeypatch):
    _stub(monkeypatch)
    smc = _bridged_field()
    plan = plan_bridge(smc, owner="me")
    result = execute(smc, ThreadManager(smc), plan, model="stub-model", owner="me")
    assert result.owner == "me"
    records = [r for r in load_claims() if r["key"] == target_key(plan.targets)]
    assert [r["released"] for r in records] == [False, True]
    assert all(r["owner"] == "me" for r in records)
    assert ("bridge", target_key(plan.targets)) not in active_claims()


def test_execute_releases_even_when_model_fails(monkeypatch):
    def boom(prompt, model="stub", timeout=None):
        raise RuntimeError("model down")
    monkeypatch.setattr(tmgr, "query_ollama", boom)
    smc = _bridged_field()
    plan = plan_bridge(smc, owner="me")
    try:
        execute(smc, ThreadManager(smc), plan, model="x", owner="me")
    except RuntimeError:
        pass
    assert ("bridge", target_key(plan.targets)) not in active_claims()


def test_run_pursuits_skips_queue_entry_claimed_by_other(monkeypatch):
    _stub(monkeypatch)
    smc = _bridged_field()
    claim("deepen", "a1", "someone-else")
    results = run_pursuits(smc, ThreadManager(smc), ["a1", "b2"],
                           cfg={"enabled": True, "max_per_cycle": 3},
                           model="stub-model", owner="me")
    deepens = [r for r in results if r.kind == "deepen"]
    assert deepens[0].skipped and "someone-else" in deepens[0].skipped
    assert deepens[1].motif_id is not None and deepens[1].targets == ["b2"]


def test_default_owner_is_the_session_agent(monkeypatch):
    _stub(monkeypatch)
    smc = _bridged_field()
    plan = plan_bridge(smc)
    result = execute(smc, ThreadManager(smc), plan, model="stub-model")
    assert result.owner == "test-agent"          # SMC_AGENT from conftest
