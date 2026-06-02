import os
import tempfile

from replay_rag import (
    DecisionGraphBuilder,
    HashEmbedder,
    MockReasoningProvider,
    RetrievalEngine,
    TraceSegmenter,
)
from replay_rag.retrieval import InMemoryTraceStore
from replay_rag.types import Trace


def _make_trace(problem: str) -> Trace:
    resp = MockReasoningProvider().solve(problem)
    segs = TraceSegmenter().segment(resp.reasoning)
    g = DecisionGraphBuilder().build(segs)
    return Trace(problem=problem, reasoning=resp.reasoning, answer=resp.answer, segments=segs, graph=g)


def test_empty_store_returns_no_hits():
    eng = RetrievalEngine(embedder=HashEmbedder(dim=256))
    assert eng.retrieve("anything", k=3) == []
    assert len(eng) == 0


def test_indexed_traces_are_retrievable():
    eng = RetrievalEngine(embedder=HashEmbedder(dim=512), store=InMemoryTraceStore())
    eng.index(_make_trace("What is the sum of 5 and 7?"))
    eng.index(_make_trace("What is the sum of 100 and 200?"))
    eng.index(_make_trace("What is the difference of 10 and 4?"))
    hits = eng.retrieve("What is the sum of 8 and 12?", k=2)
    assert len(hits) <= 2
    assert hits, "expected at least one hit above min_similarity"
    # The top hit should be one of the addition problems, not subtraction.
    top_problem = hits[0][0].problem.lower()
    assert "sum" in top_problem


def test_top_k_respected():
    eng = RetrievalEngine(embedder=HashEmbedder(dim=256), min_similarity=0.0)
    for i in range(5):
        eng.index(_make_trace(f"What is the sum of {i} and {i+1}?"))
    hits = eng.retrieve("sum of two numbers", k=3)
    assert len(hits) == 3


def test_save_and_load_roundtrip():
    store = InMemoryTraceStore()
    eng = RetrievalEngine(embedder=HashEmbedder(dim=256), store=store)
    eng.index(_make_trace("What is the sum of 5 and 7?"))
    eng.index(_make_trace("What is the difference of 10 and 4?"))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "store.json")
        store.save(path)
        store2 = InMemoryTraceStore()
        store2.load(path)
        assert len(store2) == 2
        eng2 = RetrievalEngine(embedder=HashEmbedder(dim=256), store=store2, min_similarity=0.0)
        hits = eng2.retrieve("sum of numbers", k=1)
        assert hits and "sum" in hits[0][0].problem.lower()
