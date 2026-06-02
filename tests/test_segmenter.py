from replay_rag import MockReasoningProvider, TraceSegmenter
from replay_rag.types import SegmentKind


def test_segmenter_handles_empty_trace():
    seg = TraceSegmenter()
    assert seg.segment("") == []
    assert seg.segment("   \n  \n  ") == []


def test_segmenter_finds_setup_and_answer():
    text = (
        "Setup: I have two numbers.\n\n"
        "Final answer: 42."
    )
    segs = TraceSegmenter().segment(text)
    assert len(segs) == 2
    assert segs[0].kind == SegmentKind.SETUP
    assert segs[-1].kind == SegmentKind.ANSWER


def test_segmenter_finds_backtrack_and_marks_dead_end():
    # Real arithmetic trace from the mock — has explore -> backtrack pattern.
    resp = MockReasoningProvider().solve("What is the sum of 7 and 3?")
    segs = TraceSegmenter().segment(resp.reasoning)
    kinds = [s.kind for s in segs]
    assert SegmentKind.BACKTRACK in kinds
    # the segment immediately before the BACKTRACK should be flagged dead-end
    bt_idx = kinds.index(SegmentKind.BACKTRACK)
    assert bt_idx > 0
    assert segs[bt_idx - 1].is_dead_end is True


def test_segmenter_offsets_are_within_source():
    text = "Setup: foo.\n\nLet me try one approach.\n\nFinal answer: bar."
    segs = TraceSegmenter().segment(text)
    for s in segs:
        assert text[s.start_char:s.end_char] == s.text


def test_segmenter_summary_truncates():
    long_line = "x" * 500
    text = f"Setup line.\n\n{long_line}\n\nFinal answer: ok."
    segs = TraceSegmenter().segment(text)
    assert all(len(s.summary) <= 160 for s in segs)
