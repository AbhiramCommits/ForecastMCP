"""Budget ceilings and typed recovery behaviour."""

from __future__ import annotations

import time

import pytest

from forecastmcp.mcp.budgets import BudgetExceeded, BudgetGuard


def test_per_call_timeout_triggers_typed_error():
    """A slow tool call is cut off and reported as a recoverable budget error."""
    guard = BudgetGuard(per_call_timeout_seconds=0.05)
    with pytest.raises(BudgetExceeded) as exc:
        guard.run(lambda: time.sleep(0.3))
    assert exc.value.reason == "per_call_timeout"
    assert guard.denied == 1


def test_max_calls_ceiling():
    """The call ceiling fires before the next call."""
    guard = BudgetGuard(max_calls=1)
    guard.record(0.0)
    with pytest.raises(BudgetExceeded) as exc:
        guard.check()
    assert exc.value.reason == "max_calls"


def test_cost_ceiling_and_wall_clock():
    """Cost and wall-clock budgets both raise typed errors."""
    guard = BudgetGuard(max_cost_usd=0.01)
    guard.record(0.009)
    with pytest.raises(BudgetExceeded) as exc:
        guard.check(estimated_cost=0.005)
    assert exc.value.reason == "max_cost"

    ticks = iter([0.0, 100.0, 100.0])
    clock_guard = BudgetGuard(max_wall_clock_seconds=10.0, clock=lambda: next(ticks, 100.0))
    with pytest.raises(BudgetExceeded) as exc:
        clock_guard.check()
    assert exc.value.reason == "max_wall_clock"


def test_snapshot_reports_usage():
    """Snapshots expose usage for reports and agent evaluation."""
    guard = BudgetGuard(max_calls=3)
    guard.record(0.002)
    snap = guard.snapshot()
    assert snap["calls"] == 1
    assert snap["cost_usd"] == pytest.approx(0.002)
    assert snap["max_calls"] == 3
