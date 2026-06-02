"""End-to-end demo on the mock provider.

Run: python examples/quickstart.py
"""
from replay_rag import (
    EvalCase,
    EvaluationEngine,
    HashEmbedder,
    MockReasoningProvider,
    ReplayPipeline,
)
from replay_rag.retrieval import InMemoryTraceStore


def demo_pipeline() -> None:
    print("=" * 60)
    print("Pipeline demo: cold solve -> warm solve")
    print("=" * 60)
    pipe = ReplayPipeline(
        provider=MockReasoningProvider(),
        embedder=HashEmbedder(dim=512),
        store=InMemoryTraceStore(),
    )
    cold = pipe.solve("What is the sum of 7 and 3?")
    print(f"cold solve  | tokens={cold.output_tokens}  | scaffold=None")
    warm = pipe.solve("What is the sum of 50 and 60?")
    print(f"warm solve  | tokens={warm.output_tokens}  | scaffold={warm.scaffold[:60]+'...' if warm.scaffold else None}")
    print(f"reduction   | {1 - warm.output_tokens / cold.output_tokens:.1%}")
    print()


def demo_eval() -> None:
    print("=" * 60)
    print("Eval demo: 5 cases, 5 warmup, A/B baseline vs replay")
    print("=" * 60)
    cases = [EvalCase(f"What is the sum of {i*2} and {i*3}?", str(i*5)) for i in range(1, 6)]
    warmup = [EvalCase(f"What is the sum of {10*i} and {5*i}?") for i in range(1, 6)]
    engine = EvaluationEngine(
        provider_factory=MockReasoningProvider,
        embedder=HashEmbedder(dim=1024),
    )
    result = engine.run(cases, warmup=warmup)
    print(result.summary())


if __name__ == "__main__":
    demo_pipeline()
    demo_eval()
