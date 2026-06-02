"""Trace Capture Middleware.

Wraps any ReasoningProvider so every solve() emits a Trace with raw text,
counts, and a hash id. Capture is intentionally pre-segmentation — the
segmenter and graph builder run downstream so capture remains the
single, cheap, side-effect-free hook that intercepts an LLM call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from .providers import ReasoningProvider, ProviderResponse
from .types import Trace


@dataclass
class TraceCapture:
    """Decorate a provider with capture hooks.

    Each call to .solve(problem) returns (Trace, ProviderResponse) so the
    caller controls whether to persist, segment, or both. The capture itself
    is pure: no I/O, no DB writes. That contract keeps it safe to insert
    into any reasoning loop without surprising side effects.
    """
    provider: ReasoningProvider
    on_capture: Optional[Callable[[Trace], None]] = None
    captured: list[Trace] = field(default_factory=list)

    def solve(self, problem: str, *, scaffold: Optional[str] = None) -> tuple[Trace, ProviderResponse]:
        resp = self.provider.solve(problem, scaffold=scaffold)
        trace = Trace(
            problem=problem,
            reasoning=resp.reasoning,
            answer=resp.answer,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "scaffolded": scaffold is not None,
            },
            token_count=resp.output_tokens,
        )
        self.captured.append(trace)
        if self.on_capture is not None:
            self.on_capture(trace)
        return trace, resp


@contextmanager
def capture(provider: ReasoningProvider) -> Iterator[TraceCapture]:
    """Convenience context manager so a caller can use the middleware
    with a clean 'with' block without juggling state."""
    cap = TraceCapture(provider=provider)
    try:
        yield cap
    finally:
        pass
