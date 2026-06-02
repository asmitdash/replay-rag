from replay_rag import (
    DecisionGraphBuilder,
    MockReasoningProvider,
    TraceSegmenter,
    TraceSplicer,
)
from replay_rag.types import Trace


def _make_trace(problem: str) -> Trace:
    resp = MockReasoningProvider().solve(problem)
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    return Trace(problem=problem, reasoning=resp.reasoning, answer=resp.answer, segments=segs, graph=g)


def test_splicer_drops_dead_ends_by_default():
    trace = _make_trace("What is the sum of 7 and 3?")
    out = TraceSplicer(max_tokens=2000).splice(trace, similarity=0.9)
    # The wrong-path text contains "subtract"; that should be dropped.
    assert "subtract" not in out.text.lower()
    # The right calculation should survive.
    assert "10" in out.text or "add" in out.text.lower()


def test_splicer_respects_token_budget():
    trace = _make_trace("What is the sum of 7 and 3?")
    out = TraceSplicer(max_tokens=20).splice(trace, similarity=0.9)
    assert out.tokens_estimate <= 60  # header/footer overhead beyond budget


def test_splicer_includes_problem_and_answer_metadata():
    trace = _make_trace("What is the sum of 7 and 3?")
    out = TraceSplicer(max_tokens=2000).splice(trace, similarity=0.9)
    assert "sum of 7 and 3" in out.text.lower()
    assert "10" in out.text


def test_splicer_top_k_concatenates_with_separators():
    traces = [
        (_make_trace("What is the sum of 7 and 3?"), 0.9),
        (_make_trace("What is the sum of 5 and 6?"), 0.7),
    ]
    out = TraceSplicer(max_tokens=2000).splice_top_k(traces)
    assert "---" in out
    # Both source problems should appear
    assert "7 and 3" in out
    assert "5 and 6" in out


def test_splicer_can_keep_dead_ends_when_disabled():
    trace = _make_trace("What is the sum of 7 and 3?")
    out = TraceSplicer(max_tokens=4000, drop_dead_ends=False).splice(trace, similarity=0.9)
    # When dead-ends are kept, the wrong-path "subtract" reasoning shows up.
    assert "subtract" in out.text.lower()
