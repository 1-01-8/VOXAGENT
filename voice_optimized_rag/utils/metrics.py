"""Latency instrumentation and metrics tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyRecord:
    """A single latency measurement."""
    component: str
    operation: str
    duration_ms: float
    timestamp: float


class MetricsCollector:
    """Collects and reports latency metrics for all system components."""

    def __init__(self) -> None:
        self._records: list[LatencyRecord] = []
        self._counters: dict[str, int] = defaultdict(int)

    def record_latency(self, component: str, operation: str, duration_ms: float) -> None:
        self._records.append(LatencyRecord(
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
        ))

    def increment(self, counter_name: str, amount: int = 1) -> None:
        self._counters[counter_name] += amount

    def get_counter(self, counter_name: str) -> int:
        return self._counters.get(counter_name, 0)

    @property
    def cache_hit_rate(self) -> float:
        hits = self._counters.get("cache_hit", 0)
        misses = self._counters.get("cache_miss", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def get_avg_latency(self, component: str, operation: str | None = None) -> float:
        filtered = [
            r for r in self._records
            if r.component == component
            and (operation is None or r.operation == operation)
        ]
        if not filtered:
            return 0.0
        return sum(r.duration_ms for r in filtered) / len(filtered)

    def get_p99_latency(self, component: str, operation: str | None = None) -> float:
        filtered = sorted(
            (r.duration_ms for r in self._records
             if r.component == component
             and (operation is None or r.operation == operation))
        )
        if not filtered:
            return 0.0
        idx = int(len(filtered) * 0.99)
        return filtered[min(idx, len(filtered) - 1)]

    def summary(self) -> dict[str, Any]:
        components = {r.component for r in self._records}
        result: dict[str, Any] = {
            "counters": dict(self._counters),
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            "latency": {},
        }
        for comp in sorted(components):
            comp_records = [r for r in self._records if r.component == comp]
            operations = {r.operation for r in comp_records}
            result["latency"][comp] = {}
            for op in sorted(operations):
                result["latency"][comp][op] = {
                    "avg_ms": round(self.get_avg_latency(comp, op), 2),
                    "p99_ms": round(self.get_p99_latency(comp, op), 2),
                    "count": sum(1 for r in comp_records if r.operation == op),
                }
        return result

    def reset(self) -> None:
        self._records.clear()
        self._counters.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics: MetricsCollector, component: str, operation: str) -> None:
        self._metrics = metrics
        self._component = component
        self._operation = operation
        self._start: float = 0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._metrics.record_latency(self._component, self._operation, elapsed_ms)
        self.elapsed_ms = elapsed_ms
