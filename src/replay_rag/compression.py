"""Trace Compression Layer.

Compresses a captured trace before storage. Two operations, both
deterministic and free at write-time (no LLM calls):

1. Drop dead-end segments and the BACKTRACK markers that cancel them.
   These are the segments the splicer also drops — but the splicer
   operates per-retrieval, while compression is one-time at write.
   Compressing at write makes retrieval cheaper, the index smaller,
   and the splicer's job simpler.

2. Token-budget cap. If the compressed reasoning still exceeds a budget,
   keep the SETUP, all DECISION segments, the final VERIFY, and the
   ANSWER. Drop EXPLORE segments from the middle outward. This preserves
   the strategic skeleton — exactly what the embedding cares about.

The original raw reasoning is preserved on `trace.metadata["raw_reasoning"]`
so a debugger can see what was dropped. Only `trace.reasoning` is the
compressed view.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Segment, SegmentKind, Trace


@dataclass
class CompressionStats:
    original_tokens: int
    compressed_tokens: int
    segments_dropped: int

    @property
    def reduction_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


class TraceCompressor:
    def __init__(self, max_tokens: int = 600, drop_dead_ends: bool = True) -> None:
        self.max_tokens = max_tokens
        self.drop_dead_ends = drop_dead_ends

    def compress(self, trace: Trace) -> CompressionStats:
        """Mutates trace.reasoning, trace.segments, and trace.graph in place.

        After compression the segments are re-indexed (so .index values
        remain contiguous 0..N-1) and the graph is rebuilt's edges-only.
        We don't need to re-run the segmenter — kinds were already assigned.
        """
        original = list(trace.segments)
        original_tokens = self._tok_estimate(trace.reasoning)
        kept = self._select(original)
        kept = self._budget_trim(kept)
        # Re-index
        for new_i, seg in enumerate(kept):
            seg.index = new_i
        new_reasoning = "\n\n".join(s.text for s in kept)
        trace.metadata.setdefault("raw_reasoning", trace.reasoning)
        trace.reasoning = new_reasoning
        trace.segments = kept
        if trace.graph is not None:
            trace.graph.nodes = kept
            trace.graph.edges = [(i, i + 1, "next") for i in range(len(kept) - 1)]
            # Decision points referencing dropped indices are stale; clear
            # them rather than keep dangling pointers. The splicer reads
            # segments directly, so this is safe.
            trace.graph.decision_points = []
        return CompressionStats(
            original_tokens=original_tokens,
            compressed_tokens=self._tok_estimate(new_reasoning),
            segments_dropped=len(original) - len(kept),
        )

    def _select(self, segments: list[Segment]) -> list[Segment]:
        if not self.drop_dead_ends:
            return list(segments)
        return [s for s in segments if not s.is_dead_end and s.kind != SegmentKind.BACKTRACK]

    def _budget_trim(self, kept: list[Segment]) -> list[Segment]:
        text = "\n\n".join(s.text for s in kept)
        if self._tok_estimate(text) <= self.max_tokens:
            return kept

        # Priority of kinds to keep when over budget. Higher = more essential.
        priority = {
            SegmentKind.ANSWER: 5,
            SegmentKind.VERIFY: 4,
            SegmentKind.DECISION: 3,
            SegmentKind.SETUP: 2,
            SegmentKind.EXPLORE: 1,
            SegmentKind.BACKTRACK: 0,
        }
        # Keep all high-priority segments, then add EXPLORE from the start
        # outward until we hit the budget.
        essential = [s for s in kept if priority[s.kind] >= 2]
        explores = [s for s in kept if s.kind == SegmentKind.EXPLORE]
        out = list(essential)
        # Sort essential back into original order
        out.sort(key=lambda s: s.index)
        # Add explores in order until budget exhausted
        for e in explores:
            trial = sorted(out + [e], key=lambda s: s.index)
            if self._tok_estimate("\n\n".join(s.text for s in trial)) <= self.max_tokens:
                out = trial
            else:
                break
        if not out:
            # Worst case: nothing fits. Keep the answer-bearing segment.
            answer_seg = next((s for s in kept if s.kind == SegmentKind.ANSWER), None)
            return [answer_seg] if answer_seg else kept[:1]
        return out

    @staticmethod
    def _tok_estimate(text: str) -> int:
        return max(1, len(text) // 4)
