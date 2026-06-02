"""Trace Splicer.

Picks a 'subtree' of a retrieved trace to inject into a new prompt as
scaffolding. The pitch promises "splice a relevant subtree" — a subtree
here means a contiguous winning-path slice plus the decision-point
that anchors it.

Selection rules:
- Prefer compressed reasoning (dead-ends dropped) — that's what makes
  scaffolding cheap.
- Prefer segments around the decision point closest to the new problem
  (heuristic: the highest-similarity decision point if multiple exist;
  for v0 we take the first one — works with the mock).
- Cap output by token budget so the splicer never blows up the prompt.

The splicer is also responsible for the prompt formatting: it returns a
ready-to-insert string, not a Trace object. The caller decides whether
to insert it as a system message, a prefix, or a tool result.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Segment, SegmentKind, Trace


@dataclass
class SpliceResult:
    text: str
    source_trace_id: str
    similarity: float
    segments_used: int
    tokens_estimate: int


class TraceSplicer:
    def __init__(self, max_tokens: int = 400, drop_dead_ends: bool = True) -> None:
        self.max_tokens = max_tokens
        self.drop_dead_ends = drop_dead_ends

    def splice(self, trace: Trace, similarity: float) -> SpliceResult:
        kept = self._select_segments(trace.segments)
        # Trim to budget by removing segments from the end, preserving
        # the decision shape (setup + decision + answer) at the front.
        text = self._format(trace, kept)
        while self._tok_estimate(text) > self.max_tokens and len(kept) > 1:
            kept = kept[:-1]
            text = self._format(trace, kept)
        return SpliceResult(
            text=text,
            source_trace_id=trace.trace_id,
            similarity=similarity,
            segments_used=len(kept),
            tokens_estimate=self._tok_estimate(text),
        )

    def splice_top_k(self, hits: list[tuple[Trace, float]]) -> str:
        """Concatenate splices for the top-k hits in similarity order.
        Caller already filtered by min_similarity in retrieval."""
        blocks: list[str] = []
        budget = self.max_tokens
        for trace, sim in hits:
            block_splicer = TraceSplicer(max_tokens=budget, drop_dead_ends=self.drop_dead_ends)
            block = block_splicer.splice(trace, sim)
            blocks.append(block.text)
            budget -= block.tokens_estimate
            if budget <= 0:
                break
        return "\n\n---\n\n".join(blocks)

    def _select_segments(self, segments: list[Segment]) -> list[Segment]:
        if self.drop_dead_ends:
            kept = [s for s in segments if not s.is_dead_end and s.kind != SegmentKind.BACKTRACK]
        else:
            kept = list(segments)
        return kept

    @staticmethod
    def _format(trace: Trace, kept: list[Segment]) -> str:
        header = f"[Past trace excerpt — problem was: {trace.problem.strip()[:200]}]\n"
        body = "\n\n".join(s.text for s in kept)
        footer = f"\n[Past answer: {trace.answer.strip()[:200]}]"
        return header + body + footer

    @staticmethod
    def _tok_estimate(text: str) -> int:
        # ~4 chars/token is the standard rough estimate for English.
        return max(1, len(text) // 4)
