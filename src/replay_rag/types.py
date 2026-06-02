"""Core data types for replay-rag.

The contract: a Trace is a (problem, reasoning_text, answer) tuple plus the
structured decomposition built from it — segments, decision points, and the
graph that connects them. All downstream components operate on these types.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SegmentKind(str, Enum):
    SETUP = "setup"
    EXPLORE = "explore"
    DECISION = "decision"
    BACKTRACK = "backtrack"
    VERIFY = "verify"
    ANSWER = "answer"


@dataclass
class Segment:
    """A contiguous chunk of reasoning text with one rhetorical role."""
    index: int
    kind: SegmentKind
    text: str
    start_char: int
    end_char: int
    summary: Optional[str] = None
    is_dead_end: bool = False

    def __post_init__(self) -> None:
        if self.summary is None:
            self.summary = self._auto_summary()

    def _auto_summary(self) -> str:
        first_line = self.text.strip().split("\n", 1)[0]
        return first_line[:160]


@dataclass
class DecisionPoint:
    """A branching/choice moment in the reasoning."""
    segment_index: int
    options_considered: list[str]
    chosen: Optional[str] = None
    led_to_dead_end: bool = False


@dataclass
class DecisionGraph:
    """Graph over segments. Edges represent reasoning flow; backtrack edges
    are explicit so dead-ends are traceable for compression."""
    nodes: list[Segment] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)  # (from, to, kind)
    decision_points: list[DecisionPoint] = field(default_factory=list)

    def successors(self, idx: int) -> list[int]:
        return [b for a, b, _ in self.edges if a == idx]

    def winning_path(self) -> list[int]:
        """Indices of segments not on a dead-end branch, in order."""
        return [n.index for n in self.nodes if not n.is_dead_end]


@dataclass
class Trace:
    """A captured reasoning episode."""
    problem: str
    reasoning: str
    answer: str
    trace_id: str = ""
    segments: list[Segment] = field(default_factory=list)
    graph: Optional[DecisionGraph] = None
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = self._hash_id()

    def _hash_id(self) -> str:
        h = hashlib.sha256()
        h.update(self.problem.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.reasoning[:512].encode("utf-8"))
        return h.hexdigest()[:16]


@dataclass
class Tuple:
    """The (problem → reasoning → answer) tuple as a portable record."""
    problem: str
    reasoning: str
    answer: str
    trace_id: Optional[str] = None
