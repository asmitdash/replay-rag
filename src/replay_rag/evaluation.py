"""Evaluation Engine.

Measures the wedge: does scaffolded solving cost fewer output tokens at
similar accuracy? On a list of cases, run two passes:

1. baseline_pass — every case solved cold (replay disabled). Captures
   baseline output tokens and accuracy.
2. replay_pass — same cases solved with retrieval+splice on. Captures
   reduced-output tokens and accuracy.

To make replay actually do something, the eval also supports a 'warmup'
list: extra problems solved before the replay pass to populate the trace
store with relevant prior reasoning.

Reports:
- baseline accuracy / replay accuracy / accuracy delta
- baseline mean output tokens / replay mean output tokens / token reduction %
- per-case retrieval stats (top-k similarity, scaffold size)

This is offline-friendly: with the MockReasoningProvider, it runs in
sub-second time and gives meaningful numbers because the mock returns
the same arithmetic answer in fewer reasoning tokens when it gets a
scaffold (the scaffold replaces the explore-and-backtrack section).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .embedding import Embedder
from .pipeline import PipelineRun, ReplayPipeline
from .providers import ReasoningProvider
from .retrieval import InMemoryTraceStore


@dataclass
class EvalCase:
    problem: str
    expected_answer: Optional[str] = None  # exact-match check; None disables accuracy

    def matches(self, given: str) -> bool:
        if self.expected_answer is None:
            return True
        return self._norm(self.expected_answer) == self._norm(given)

    @staticmethod
    def _norm(s: str) -> str:
        return "".join(ch for ch in s.strip().lower() if not ch.isspace() and ch != ".")


@dataclass
class _CaseStats:
    problem: str
    answer: str
    correct: bool
    output_tokens: int
    retrieved_count: int
    top_similarity: float
    scaffold_tokens: int


@dataclass
class EvalResult:
    baseline: list[_CaseStats] = field(default_factory=list)
    replay: list[_CaseStats] = field(default_factory=list)

    @property
    def baseline_accuracy(self) -> float:
        return self._acc(self.baseline)

    @property
    def replay_accuracy(self) -> float:
        return self._acc(self.replay)

    @property
    def baseline_mean_tokens(self) -> float:
        return self._mean([c.output_tokens for c in self.baseline])

    @property
    def replay_mean_tokens(self) -> float:
        return self._mean([c.output_tokens for c in self.replay])

    @property
    def token_reduction(self) -> float:
        b = self.baseline_mean_tokens
        if b == 0:
            return 0.0
        return 1.0 - (self.replay_mean_tokens / b)

    def summary(self) -> str:
        return (
            f"cases: {len(self.baseline)}\n"
            f"baseline acc: {self.baseline_accuracy:.1%}  | replay acc: {self.replay_accuracy:.1%}\n"
            f"baseline tokens: {self.baseline_mean_tokens:.1f}/case  | "
            f"replay tokens: {self.replay_mean_tokens:.1f}/case  | "
            f"reduction: {self.token_reduction:.1%}"
        )

    @staticmethod
    def _acc(rows: list[_CaseStats]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.correct) / len(rows)

    @staticmethod
    def _mean(xs: list[int]) -> float:
        if not xs:
            return 0.0
        return sum(xs) / len(xs)


class EvaluationEngine:
    """A/B test driver.

    Notes on fairness:
    - Baseline and replay use *fresh* pipelines so cross-run state never
      leaks. Same provider instance though — provider behavior must be
      deterministic for the comparison to mean anything (the mock is).
    - Warmup problems are solved with replay enabled in the replay pass
      *only*, after baseline finishes, so the baseline run isn't
      contaminated by a different number of solves.
    """

    def __init__(
        self,
        provider_factory: Callable[[], ReasoningProvider],
        embedder: Optional[Embedder] = None,
    ) -> None:
        self._provider_factory = provider_factory
        self._embedder = embedder

    def run(
        self,
        cases: Iterable[EvalCase],
        warmup: Optional[Iterable[EvalCase]] = None,
    ) -> EvalResult:
        cases = list(cases)
        warmup = list(warmup or [])
        result = EvalResult()

        # --- baseline: replay disabled, fresh store
        baseline_pipe = ReplayPipeline(
            provider=self._provider_factory(),
            embedder=self._embedder,
            store=InMemoryTraceStore(),
        )
        for c in cases:
            run = baseline_pipe.solve(c.problem, replay=False)
            result.baseline.append(self._row(c, run))

        # --- replay: warmup populates the store, then real cases run with replay on
        replay_pipe = ReplayPipeline(
            provider=self._provider_factory(),
            embedder=self._embedder,
            store=InMemoryTraceStore(),
        )
        for w in warmup:
            replay_pipe.solve(w.problem, replay=True)
        for c in cases:
            run = replay_pipe.solve(c.problem, replay=True)
            result.replay.append(self._row(c, run))
        return result

    @staticmethod
    def _row(case: EvalCase, run: PipelineRun) -> _CaseStats:
        scaffold_tokens = max(1, len(run.scaffold) // 4) if run.scaffold else 0
        return _CaseStats(
            problem=case.problem,
            answer=run.trace.answer,
            correct=case.matches(run.trace.answer),
            output_tokens=run.output_tokens,
            retrieved_count=run.retrieved_count,
            top_similarity=run.top_similarity,
            scaffold_tokens=scaffold_tokens,
        )
