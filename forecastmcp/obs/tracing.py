"""OpenTelemetry tracing with a JSONL span exporter.

Every run writes ``artifacts/runs/<run_id>/trace.jsonl`` (one span per line) and
``artifacts/runs/<run_id>/calls/<call_id>.json`` for the stable serialized tool
output. Trace attributes carry tool name, validated-input hash, model and
revision, latency, cost, success/error and the output pointer needed by replay.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from forecastmcp.config import RUNS_DIR

_PRIMITIVES = (str, bool, int, float)


def canonical_json(payload: Any) -> str:
    """Canonical, sorted, compact JSON used for hashing and byte comparisons."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JsonlSpanExporter(SpanExporter):
    """Append every ended span as one JSON line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        """Write spans; failures are reported rather than raised into user code."""
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                for span in spans:
                    fh.write(json.dumps(_span_to_dict(span), sort_keys=True) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:  # pragma: no cover - disk errors should not break tools
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """No persistent resources to release."""


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    attrs = {}
    for key, value in (span.attributes or {}).items():
        if isinstance(value, _PRIMITIVES):
            attrs[key] = value
        else:
            attrs[key] = str(value)
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else None,
        "span_id": format(ctx.span_id, "016x") if ctx else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "status": span.status.status_code.name if span.status else "UNSET",
        "attributes": attrs,
    }


class RunTrace:
    """Tracer scoped to one run directory."""

    def __init__(self, run_id: str, base_dir: Path | str | None = None):
        self.run_id = run_id
        self.run_dir = Path(base_dir or RUNS_DIR) / run_id
        self.calls_dir = self.run_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.trace_path.touch(exist_ok=True)
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(self.trace_path)))
        self.tracer = self.provider.get_tracer("forecastmcp", "0.1.0")

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        """Create a span and set primitive attributes on it."""
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is None:
                    continue
                span.set_attribute(key, value if isinstance(value, _PRIMITIVES) else str(value))
            yield span

    def event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a short-lived point-in-time span (compaction, downgrade, denial)."""
        with self.span(name, attributes):
            pass

    def write_call_output(self, call_id: str, payload: Any) -> tuple[Path, str]:
        """Write the stable output bytes and return (path, sha256)."""
        text = canonical_json(payload)
        path = self.calls_dir / f"{call_id}.json"
        path.write_text(text, encoding="utf-8")
        return path, sha256_text(text)

    def close(self) -> None:
        """Flush and shut down the tracer provider."""
        with contextlib.suppress(Exception):
            self.provider.shutdown()


def read_trace(run_dir: Path | str) -> list[dict[str, Any]]:
    """Read all span records from a run's trace.jsonl."""
    path = Path(run_dir) / "trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no trace at {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tool_calls(run_dir: Path | str) -> list[dict[str, Any]]:
    """Return only tool-call span records, in sequence order."""
    calls = [s for s in read_trace(run_dir) if s["name"].startswith("tool.")]
    return sorted(calls, key=lambda s: s.get("start_time_ns") or 0)
