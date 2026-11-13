"""Re-execute a traced run and assert byte-identical tool outputs."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from forecastmcp.mcp.budgets import BudgetGuard
from forecastmcp.mcp.tools import ToolRuntime
from forecastmcp.obs.tracing import RUNS_DIR, read_trace, tool_calls


@dataclass
class ReplayReport:
    """Outcome of replaying one run."""

    run_id: str
    matched: int = 0
    mismatched: int = 0
    skipped: int = 0
    non_tool_spans: int = 0
    missing_outputs: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every replayable call matched its cached output byte-for-byte."""
        return self.mismatched == 0 and self.missing_outputs == 0

    def to_dict(self) -> dict:
        """Serialise for CLI/JSON output."""
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "skipped_failed_calls": self.skipped,
            "non_tool_spans": self.non_tool_spans,
            "missing_outputs": self.missing_outputs,
            "details": self.details,
        }


def replay(run_id: str, base_dir: Path | str | None = None) -> ReplayReport:
    """Re-execute each successful tool call from trace.jsonl and compare bytes.

    The original call outputs are never overwritten: replay writes into a
    temporary run directory. Failed calls are re-executed and expected to fail.
    """
    runs_root = Path(base_dir or RUNS_DIR)
    run_dir = runs_root / run_id
    records = read_trace(run_dir)
    calls = tool_calls(run_dir)
    report = ReplayReport(run_id=run_id)
    if not calls:
        report.details.append("trace contains no tool calls")
        return report

    guard = BudgetGuard(
        max_calls=10_000,
        max_wall_clock_seconds=86_400,
        max_cost_usd=1_000.0,
        per_call_timeout_seconds=600.0,
    )
    with tempfile.TemporaryDirectory(prefix=f"replay-{run_id}-") as tmp:
        runtime = ToolRuntime(
            run_id=run_id,
            role=str(calls[0]["attributes"].get("tool.role", "analyst")),
            guard=guard,
            base_dir=Path(tmp),
        )
        try:
            for call in calls:
                attrs = call["attributes"]
                tool = attrs["tool.name"]
                call_id = attrs["call.id"]
                arguments = json.loads(attrs["call.input_json"])
                original_path = Path(attrs.get("call.output_path", ""))
                original_sha = attrs.get("call.output_sha256")
                success = bool(attrs.get("call.success", False))
                if not success:
                    try:
                        runtime.execute(tool, arguments, call_id=call_id)
                    except Exception:
                        report.matched += 1
                    else:
                        report.skipped += 1
                        report.details.append(f"{call_id} {tool}: expected failure but succeeded")
                    continue
                if not original_path.exists():
                    report.missing_outputs += 1
                    report.details.append(f"{call_id} {tool}: original output missing")
                    continue
                try:
                    runtime.execute(tool, arguments, call_id=call_id)
                except Exception as exc:
                    report.mismatched += 1
                    report.details.append(f"{call_id} {tool}: replay raised {type(exc).__name__}: {exc}")
                    continue
                replay_path = Path(tmp) / run_id / "calls" / f"{call_id}.json"
                if not replay_path.exists():
                    report.mismatched += 1
                    report.details.append(f"{call_id} {tool}: replay produced no output")
                    continue
                original_bytes = original_path.read_bytes()
                replay_bytes = replay_path.read_bytes()
                if original_bytes == replay_bytes:
                    report.matched += 1
                else:
                    report.mismatched += 1
                    report.details.append(
                        f"{call_id} {tool}: bytes differ "
                        f"(original_sha={original_sha}, replay_sha={_sha(replay_bytes)})"
                    )
        finally:
            runtime.close()
    report.non_tool_spans = len(records) - len(calls)
    return report


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
