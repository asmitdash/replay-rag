import numpy as np
import pytest

from replay_rag import (
    DecisionGraphBuilder,
    HashEmbedder,
    MockReasoningProvider,
    TraceSegmenter,
)
from replay_rag.embedding import build_index_text
from replay_rag.types import Trace


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(dim=128)
    a = e.embed("the quick brown fox")
    b = e.embed("the quick brown fox")
    assert np.allclose(a, b)


def test_hash_embedder_unit_norm():
    e = HashEmbedder(dim=128)
    v = e.embed("some text with several words")
    assert v.shape == (128,)
    assert pytest.approx(float(np.linalg.norm(v)), abs=1e-5) == 1.0


def test_hash_embedder_empty_returns_zero_vec():
    e = HashEmbedder(dim=64)
    v = e.embed("")
    assert v.shape == (64,)
    assert float(np.linalg.norm(v)) == 0.0


def test_hash_embedder_similar_texts_have_higher_cosine():
    e = HashEmbedder(dim=512)
    a = e.embed("sum of two numbers seven and three")
    b = e.embed("add two integers five and nine")  # arithmetic-similar
    c = e.embed("recipe for chocolate cake with butter and sugar")
    sim_ab = float(a @ b)
    sim_ac = float(a @ c)
    assert sim_ab > sim_ac


def test_build_index_text_uses_problem_and_decision_summaries():
    resp = MockReasoningProvider().solve("What is the sum of 7 and 3?")
    segs = TraceSegmenter().segment(resp.reasoning)
    graph = DecisionGraphBuilder().build(segs)
    t = Trace(problem="What is the sum of 7 and 3?", reasoning=resp.reasoning, answer=resp.answer, segments=segs, graph=graph)
    text = build_index_text(t)
    assert "sum of 7 and 3" in text
    assert "considered" in text or "chose" in text
    # Crucially: raw reasoning text is NOT included verbatim — that's the design.
    assert "Setup: I have two numbers" not in text
