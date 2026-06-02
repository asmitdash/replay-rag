from replay_rag import (
    DecisionGraphBuilder,
    MockReasoningProvider,
    TraceCompressor,
    TraceSegmenter,
)
from replay_rag.types import SegmentKind, Trace


def _make_trace(problem: str) -> Trace:
    resp = MockReasoningProvider().solve(problem)
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    return Trace(problem=problem, reasoning=resp.reasoning, answer=resp.answer, segments=segs, graph=g)


def test_compress_drops_dead_ends_and_backtrack_markers():
    trace = _make_trace("What is the sum of 7 and 3?")
    n_before = len(trace.segments)
    stats = TraceCompressor(max_tokens=10000).compress(trace)
    assert stats.segments_dropped >= 2  # 1 dead-end + 1 backtrack at minimum
    kinds = {s.kind for s in trace.segments}
    assert SegmentKind.BACKTRACK not in kinds
    assert all(not s.is_dead_end for s in trace.segments)
    assert len(trace.segments) < n_before


def test_compress_preserves_raw_reasoning_in_metadata():
    trace = _make_trace("What is the sum of 7 and 3?")
    raw = trace.reasoning
    TraceCompressor(max_tokens=10000).compress(trace)
    assert trace.metadata["raw_reasoning"] == raw
    assert trace.reasoning != raw  # something was dropped


def test_compress_reindexes_segments_contiguously():
    trace = _make_trace("What is the sum of 7 and 3?")
    TraceCompressor(max_tokens=10000).compress(trace)
    assert [s.index for s in trace.segments] == list(range(len(trace.segments)))


def test_compress_under_budget_keeps_essentials():
    trace = _make_trace("What is the sum of 7 and 3?")
    TraceCompressor(max_tokens=20).compress(trace)
    kinds = [s.kind for s in trace.segments]
    # Even at very tight budget, we must keep the answer-bearing segment.
    assert SegmentKind.ANSWER in kinds or SegmentKind.DECISION in kinds


def test_compression_reduction_ratio_positive():
    trace = _make_trace("What is the sum of 7 and 3?")
    stats = TraceCompressor(max_tokens=10000).compress(trace)
    assert stats.reduction_ratio > 0.0
