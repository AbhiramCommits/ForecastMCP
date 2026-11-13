"""Per-run budgets: call count, wall clock, cost and per-call timeout."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, TypeVar

T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """Typed budget failure the agent can recover from (e.g. retry cheaper)."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"BudgetExceeded[{reason}]: {message}")


class BudgetGuard:
    """Enforces the four budget dimensions from ``configs/costs.yaml``."""

    def __init__(
        self,
        max_calls: int = 40,
        max_wall_clock_seconds: float = 900.0,
        max_cost_usd: float = 0.50,
        per_call_timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_calls = int(max_calls)
        self.max_wall_clock_seconds = float(max_wall_clock_seconds)
        self.max_cost_usd = float(max_cost_usd)
        self.per_call_timeout_seconds = float(per_call_timeout_seconds)
        self._clock = clock
        self.started = clock()
        self.calls = 0
        self.cost_usd = 0.0
        self.denied = 0

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return self._clock() - self.started

    def check(self, estimated_cost: float = 0.0) -> None:
        """Raise ``BudgetExceeded`` when the next call would violate a budget."""
        if self.calls >= self.max_calls:
            self.denied += 1
            raise BudgetExceeded("max_calls", f"{self.calls} calls already made (max {self.max_calls})")
        if self.elapsed() >= self.max_wall_clock_seconds:
            self.denied += 1
            raise BudgetExceeded(
                "max_wall_clock",
                f"{self.elapsed():.1f}s elapsed (max {self.max_wall_clock_seconds:.1f}s)",
            )
        if self.cost_usd + estimated_cost > self.max_cost_usd + 1e-12:
            self.denied += 1
            raise BudgetExceeded(
                "max_cost",
                f"${self.cost_usd:.6f} spent, next ${estimated_cost:.6f} would exceed "
                f"${self.max_cost_usd:.6f}",
            )

    def run(self, fn: Callable[[], T], timeout: float | None = None) -> T:
        """Run ``fn`` with the per-call timeout, converting timeouts to budget errors."""
        limit = self.per_call_timeout_seconds if timeout is None else float(timeout)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecastmcp-call")
        future = executor.submit(fn)
        try:
            return future.result(timeout=limit)
        except FutureTimeout as exc:
            self.denied += 1
            future.cancel()
            raise BudgetExceeded(
                "per_call_timeout", f"call exceeded {limit:.1f}s timeout"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def record(self, cost_usd: float = 0.0) -> None:
        """Account for one successfully started/finished call."""
        self.calls += 1
        self.cost_usd += float(cost_usd)

    def snapshot(self) -> dict[str, Any]:
        """Current budget state for traces/reports."""
        return {
            "calls": self.calls,
            "max_calls": self.max_calls,
            "elapsed_seconds": round(self.elapsed(), 3),
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "cost_usd": round(self.cost_usd, 8),
            "max_cost_usd": self.max_cost_usd,
            "per_call_timeout_seconds": self.per_call_timeout_seconds,
            "denied": self.denied,
        }
