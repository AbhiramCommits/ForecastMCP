"""Replay must re-execute traced calls and compare byte-identical outputs."""

from __future__ import annotations

import json

from forecastmcp.mcp.budgets import BudgetGuard
from forecastmcp.mcp.tools import ToolRuntime
from forecastmcp.obs.replay import replay


def _make_run(tmp_path):
    runtime = ToolRuntime(run_id="replay-test", base_dir=tmp_path, guard=BudgetGuard(max_calls=10))
    runtime.execute("list_series", {"dataset": "synthetic_supply", "filter": {"limit": 2}})
    runtime.execute("sql", {"query": "SELECT count(*) AS n FROM series"})
    runtime.close()
    return tmp_path / "replay-test"


def test_replay_is_byte_identical(tmp_path):
    """Successful tool calls reproduce exactly."""
    _make_run(tmp_path)
    report = replay("replay-test", base_dir=tmp_path)
    assert report.ok
    assert report.matched == 2
    assert report.mismatched == 0


def test_replay_detects_tampered_output(tmp_path):
    """A modified cached output is detected as a mismatch."""
    run_dir = _make_run(tmp_path)
    call_files = sorted((run_dir / "calls").glob("*.json"))
    assert call_files
    payload = json.loads(call_files[0].read_text())
    payload["tampered"] = True
    call_files[0].write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    report = replay("replay-test", base_dir=tmp_path)
    assert not report.ok
    assert report.mismatched >= 1
