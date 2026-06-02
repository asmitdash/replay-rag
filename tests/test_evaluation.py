from replay_rag import (
    EvalCase,
    EvaluationEngine,
    HashEmbedder,
    MockReasoningProvider,
)


def test_evaluation_runs_baseline_and_replay():
    cases = [
        EvalCase("What is the sum of 12 and 7?", "19"),
        EvalCase("What is the sum of 100 and 50?", "150"),
        EvalCase("What is the sum of 3 and 4?", "7"),
    ]
    warmup = [
        EvalCase("What is the sum of 5 and 6?"),
        EvalCase("What is the sum of 10 and 20?"),
    ]
    engine = EvaluationEngine(
        provider_factory=MockReasoningProvider,
        embedder=HashEmbedder(dim=512),
    )
    result = engine.run(cases, warmup=warmup)

    assert len(result.baseline) == 3
    assert len(result.replay) == 3
    # Mock answers are computed correctly, so accuracy should be 100% in both.
    assert result.baseline_accuracy == 1.0
    assert result.replay_accuracy == 1.0


def test_replay_reduces_tokens_when_warmup_is_relevant():
    cases = [EvalCase(f"What is the sum of {i} and {i+1}?", str(2*i+1)) for i in range(5)]
    warmup = [EvalCase(f"What is the sum of {10*i} and {10*i+5}?") for i in range(5)]
    engine = EvaluationEngine(
        provider_factory=MockReasoningProvider,
        embedder=HashEmbedder(dim=1024),
    )
    result = engine.run(cases, warmup=warmup)
    # The wedge: scaffolded solves should produce shorter traces.
    assert result.replay_mean_tokens < result.baseline_mean_tokens
    assert result.token_reduction > 0.0


def test_summary_runs():
    cases = [EvalCase("What is the sum of 1 and 1?", "2")]
    engine = EvaluationEngine(provider_factory=MockReasoningProvider, embedder=HashEmbedder(dim=128))
    result = engine.run(cases)
    s = result.summary()
    assert "baseline acc" in s
    assert "replay acc" in s


def test_eval_case_normalization_handles_punctuation_and_spaces():
    c = EvalCase("Q?", "  19. ")
    assert c.matches("19")
    assert c.matches("19.")
    assert not c.matches("20")
