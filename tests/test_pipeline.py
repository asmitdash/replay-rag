from replay_rag import (
    HashEmbedder,
    MockReasoningProvider,
    ReplayPipeline,
)
from replay_rag.retrieval import InMemoryTraceStore


def test_pipeline_first_solve_has_no_retrieval():
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=256),
        store=InMemoryTraceStore(),
    )
    run = pipe.solve("What is the sum of 7 and 3?")
    assert run.retrieved_count == 0
    assert run.scaffold is None
    assert "10" in run.trace.answer


def test_pipeline_subsequent_solve_retrieves_and_scaffolds():
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=512),
        store=InMemoryTraceStore(),
    )
    pipe.solve("What is the sum of 7 and 3?")
    pipe.solve("What is the sum of 5 and 6?")
    run = pipe.solve("What is the sum of 12 and 4?")
    assert run.retrieved_count > 0
    assert run.scaffold is not None
    assert run.top_similarity > 0.15


def test_pipeline_scaffolded_run_uses_fewer_tokens():
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=512),
        store=InMemoryTraceStore(),
    )
    cold = pipe.solve("What is the sum of 7 and 3?")
    warm = pipe.solve("What is the sum of 50 and 60?")
    assert warm.scaffold is not None
    assert warm.output_tokens < cold.output_tokens


def test_pipeline_replay_disabled_skips_retrieval():
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=256),
        store=InMemoryTraceStore(),
    )
    pipe.solve("What is the sum of 7 and 3?")
    run = pipe.solve("What is the sum of 50 and 60?", replay=False)
    assert run.scaffold is None
    assert run.retrieved_count == 0


def test_pipeline_indexes_compressed_traces():
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=256),
        store=InMemoryTraceStore(),
    )
    run = pipe.solve("What is the sum of 7 and 3?")
    # After compression, raw_reasoning should be on the metadata and
    # the live reasoning should be shorter.
    assert "raw_reasoning" in run.trace.metadata
    assert len(run.trace.reasoning) < len(run.trace.metadata["raw_reasoning"])
