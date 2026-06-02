"""Reasoning model providers.

Provider returns (reasoning, answer) so capture can store both. Mock generates
realistic traces (with backtrack/decision markers) so the segmenter and graph
builder have meaningful input under test. Bedrock-Claude provider uses
extended thinking blocks as the trace.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderResponse:
    reasoning: str
    answer: str
    input_tokens: int = 0
    output_tokens: int = 0


class ReasoningProvider(ABC):
    @abstractmethod
    def solve(self, problem: str, *, scaffold: Optional[str] = None) -> ProviderResponse:
        ...


class MockReasoningProvider(ReasoningProvider):
    """Deterministic mock that emits backtracking traces.

    Behavior:
    - For arithmetic word problems with two numbers and an operator keyword
      (sum/total/add/difference/subtract/product/multiply), it computes the
      right answer and emits a trace with one wrong-attempt-then-backtrack
      so the segmenter has a real branch to find.
    - For everything else, it emits a generic 4-segment trace with a single
      decision point and no dead end.

    Token counts are estimated as len(text) // 4. Good enough for the
    relative measurements the eval engine reports.
    """

    _NUM = re.compile(r"-?\d+(?:\.\d+)?")

    def solve(self, problem: str, *, scaffold: Optional[str] = None) -> ProviderResponse:
        nums = [float(x) for x in self._NUM.findall(problem)]
        op = self._infer_op(problem)
        if op and len(nums) >= 2:
            return self._arith_trace(problem, nums[0], nums[1], op, scaffold)
        return self._generic_trace(problem, scaffold)

    @staticmethod
    def _infer_op(problem: str) -> Optional[str]:
        p = problem.lower()
        if any(k in p for k in ("sum", "total", "add", "plus", "altogether", "combined")):
            return "+"
        if any(k in p for k in ("difference", "subtract", "minus", "less than", "fewer", "left")):
            return "-"
        if any(k in p for k in ("product", "multiply", "times", "twice", "thrice")):
            return "*"
        if any(k in p for k in ("divide", "quotient", "per", "split", "share")):
            return "/"
        return None

    def _arith_trace(self, problem: str, a: float, b: float, op: str, scaffold: Optional[str]) -> ProviderResponse:
        correct = self._compute(a, b, op)
        wrong = self._compute(a, b, self._wrong_op(op))
        # If the scaffold mentions the right operation, simulate a real
        # reasoning model that uses the scaffold to skip the wrong-path
        # exploration. This is the wedge in miniature: scaffolded -> shorter trace.
        # Look for either the operator symbol (survives compression as a math
        # expression) or the verb form.
        scaffold_helps = scaffold is not None and (
            self._verb(op) in scaffold.lower() or f" {op} " in scaffold
        )
        if scaffold_helps:
            reasoning = (
                f"Setup: two numbers {a} and {b}; from prior similar work, the move is to {self._verb(op)}.\n\n"
                f"So {a} {op} {b} = {correct}.\n\n"
                f"Final answer: {self._fmt(correct)}."
            )
        else:
            scaffold_block = ""
            if scaffold:
                scaffold_block = (
                    "Reviewing prior work on a similar problem before starting:\n"
                    f"{scaffold}\n\n"
                )
            reasoning = (
                f"{scaffold_block}"
                f"Setup: I have two numbers, {a} and {b}, and need to find a value derived from them.\n\n"
                f"Let me try: maybe I should {self._verb(self._wrong_op(op))} them. "
                f"That gives {wrong}.\n\n"
                f"Wait — actually that's not what the question asks. Let me re-read.\n\n"
                f"Backtrack: the question is asking me to {self._verb(op)} the two numbers.\n\n"
                f"So the calculation is {a} {op} {b} = {correct}.\n\n"
                f"Verify: {a} {op} {b} = {correct}. That matches.\n\n"
                f"Final answer: {self._fmt(correct)}."
            )
        answer = self._fmt(correct)
        return self._wrap(reasoning, answer, problem)

    def _generic_trace(self, problem: str, scaffold: Optional[str]) -> ProviderResponse:
        scaffold_block = ""
        if scaffold:
            scaffold_block = f"Prior trace excerpt:\n{scaffold}\n\n"
        reasoning = (
            f"{scaffold_block}"
            f"Setup: read the question — \"{problem.strip()}\".\n\n"
            f"Explore: a few angles come to mind. The most direct one is to answer in plain language.\n\n"
            f"Decide: I'll just answer directly without further branching.\n\n"
            f"Verify: the response addresses the question as posed.\n\n"
            f"Final answer: (mock) acknowledged."
        )
        answer = "(mock) acknowledged."
        return self._wrap(reasoning, answer, problem)

    @staticmethod
    def _wrap(reasoning: str, answer: str, problem: str) -> ProviderResponse:
        return ProviderResponse(
            reasoning=reasoning,
            answer=answer,
            input_tokens=max(1, len(problem) // 4),
            output_tokens=max(1, len(reasoning) // 4),
        )

    @staticmethod
    def _compute(a: float, b: float, op: str) -> float:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0 else float("nan")
        raise ValueError(op)

    @staticmethod
    def _wrong_op(op: str) -> str:
        return {"+": "-", "-": "+", "*": "/", "/": "*"}[op]

    @staticmethod
    def _verb(op: str) -> str:
        return {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[op]

    @staticmethod
    def _fmt(x: float) -> str:
        return str(int(x)) if x == int(x) else f"{x:g}"


class BedrockClaudeProvider(ReasoningProvider):
    """Claude via Bedrock with extended thinking enabled.

    Extended-thinking blocks are exactly the artifact replay-rag wants: an
    explicit reasoning trace separate from the final answer. We concatenate
    all thinking blocks as `reasoning` and all text blocks as `answer`.

    Imports anthropic only inside __init__ so the package stays usable
    offline. Requires `replay-rag[bedrock]`.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7-20251201-v1:0",
        region: str = "us-east-1",
        max_tokens: int = 16000,
        thinking_budget: int = 8000,
    ) -> None:
        try:
            from anthropic import AnthropicBedrock  # type: ignore
        except ImportError as e:
            raise ImportError(
                "BedrockClaudeProvider requires `pip install replay-rag[bedrock]`"
            ) from e
        self._client = AnthropicBedrock(aws_region=region)
        self._model = model_id
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

    def solve(self, problem: str, *, scaffold: Optional[str] = None) -> ProviderResponse:
        prompt = problem
        if scaffold:
            prompt = (
                "Below is a reasoning excerpt from a similar past problem. "
                "Use it as scaffolding only — verify before relying on any step.\n\n"
                f"--- past reasoning ---\n{scaffold}\n--- end ---\n\n"
                f"Now solve:\n{problem}"
            )
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "enabled", "budget_tokens": self._thinking_budget},
            messages=[{"role": "user", "content": prompt}],
        )
        thinking = []
        text = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                thinking.append(getattr(block, "thinking", ""))
            elif btype == "text":
                text.append(getattr(block, "text", ""))
        usage = getattr(msg, "usage", None)
        return ProviderResponse(
            reasoning="\n\n".join(thinking).strip(),
            answer="\n\n".join(text).strip(),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )
