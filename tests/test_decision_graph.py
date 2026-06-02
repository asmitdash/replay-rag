from replay_rag import (
    DecisionGraphBuilder,
    MockReasoningProvider,
    TraceSegmenter,
)
from replay_rag.types import SegmentKind


def test_graph_has_sequential_next_edges():
    text = "Setup A.\n\nDecide B.\n\nFinal answer: C."
    segs = TraceSegmenter().segment(text)
    g = DecisionGraphBuilder().build(segs)
    next_edges = [(a, b) for a, b, k in g.edges if k == "next"]
    assert next_edges == [(0, 1), (1, 2)]


def test_graph_creates_backtrack_edge_to_pre_dead_end():
    resp = MockReasoningProvider().solve("What is the sum of 7 and 3?")
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    bt_edges = [(a, b) for a, b, k in g.edges if k == "backtrack"]
    assert bt_edges, "expected at least one backtrack edge"
    # source must be a BACKTRACK segment, target must NOT be a dead-end
    for src, dst in bt_edges:
        assert segs[src].kind == SegmentKind.BACKTRACK
        assert segs[dst].is_dead_end is False


def test_graph_extracts_decision_points_with_options():
    resp = MockReasoningProvider().solve("What is the sum of 7 and 3?")
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    assert len(g.decision_points) >= 1
    dp = g.decision_points[0]
    assert len(dp.options_considered) >= 2
    assert dp.chosen is not None


def test_winning_path_excludes_dead_ends():
    resp = MockReasoningProvider().solve("What is the sum of 7 and 3?")
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    winning_idxs = g.winning_path()
    for idx in winning_idxs:
        assert segs[idx].is_dead_end is False
