"""Trace Segmenter — split raw reasoning text into typed segments.

The novelty in the project pitch is "embedding over decision-point summaries,
not raw traces." That requires a stable segmentation. v0 uses heuristic
markers reasoning models actually emit (R1's "wait", o1's "actually", common
"let me try / let me re-read" patterns) plus structural cues (blank lines).
A learned segmenter is a v2 problem.

Strategy:
1. Split the trace on blank lines into raw chunks.
2. Classify each chunk by marker keywords into a SegmentKind.
3. Mark a chunk as a backtrack-target dead-end if the *next* chunk is a
   backtrack/correction segment.
"""
from __future__ import annotations

import re
from typing import Iterable

from .types import Segment, SegmentKind


_BACKTRACK_PATTERNS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\bthat's not\b",
    r"\bthat is not\b",
    r"\bbacktrack\b",
    r"\blet me re-?(?:read|check|consider|think|try)\b",
    r"\bon second thought\b",
    r"\bnope\b",
    r"\bhmm,? no\b",
    r"\bscratch that\b",
]
_DECISION_PATTERNS = [
    r"\b(?:i'll|i will|let me|let's)\b",
    r"\bdecide\b",
    r"\bchoose\b",
    r"\bgo with\b",
    r"\bopt for\b",
]
_VERIFY_PATTERNS = [
    r"\bverify\b",
    r"\bcheck\b",
    r"\bdouble[- ]?check\b",
    r"\bconfirm\b",
    r"\bsanity check\b",
    r"\bthat matches\b",
]
_ANSWER_PATTERNS = [
    r"\bfinal answer\b",
    r"\btherefore\b",
    r"\bso the answer\b",
    r"\b=\s*[-\d]",
    r"\banswer:\b",
]
_SETUP_PATTERNS = [
    r"\bsetup\b",
    r"\bgiven\b",
    r"\bproblem\b",
    r"\bi have\b",
    r"\bi need\b",
    r"\bread the question\b",
]
_EXPLORE_PATTERNS = [
    r"\bexplore\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bcould\b",
    r"\bmight\b",
    r"\bone option\b",
    r"\ba few angles\b",
    r"\btry\b",
]


def _matches(text: str, patterns: Iterable[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


class TraceSegmenter:
    """Heuristic segmenter. Stateless — re-instantiation cost is zero."""

    _SPLIT = re.compile(r"\n\s*\n")

    def segment(self, reasoning: str) -> list[Segment]:
        if not reasoning.strip():
            return []
        chunks = self._split_with_offsets(reasoning)
        segments: list[Segment] = []
        for i, (text, start, end) in enumerate(chunks):
            kind = self._classify(text, is_first=(i == 0), is_last=(i == len(chunks) - 1))
            segments.append(
                Segment(
                    index=i,
                    kind=kind,
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )
        self._mark_dead_ends(segments)
        return segments

    def _split_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Split on blank lines, preserving char offsets into the original."""
        out: list[tuple[str, int, int]] = []
        cursor = 0
        for chunk in self._SPLIT.split(text):
            stripped = chunk.strip()
            if not stripped:
                cursor += len(chunk)
                continue
            start = text.index(stripped, cursor)
            end = start + len(stripped)
            out.append((stripped, start, end))
            cursor = end
        return out

    @staticmethod
    def _classify(text: str, *, is_first: bool, is_last: bool) -> SegmentKind:
        if _matches(text, _BACKTRACK_PATTERNS):
            return SegmentKind.BACKTRACK
        if _matches(text, _ANSWER_PATTERNS) and is_last:
            return SegmentKind.ANSWER
        if _matches(text, _ANSWER_PATTERNS):
            return SegmentKind.DECISION
        if _matches(text, _VERIFY_PATTERNS):
            return SegmentKind.VERIFY
        if _matches(text, _DECISION_PATTERNS):
            return SegmentKind.DECISION
        if is_first or _matches(text, _SETUP_PATTERNS):
            return SegmentKind.SETUP
        if _matches(text, _EXPLORE_PATTERNS):
            return SegmentKind.EXPLORE
        return SegmentKind.EXPLORE

    @staticmethod
    def _mark_dead_ends(segments: list[Segment]) -> None:
        """A segment is a dead end if it is immediately followed by a
        BACKTRACK, and the segment itself is EXPLORE or DECISION (i.e., it
        was a candidate path that got rejected). Setup and Verify are never
        dead ends."""
        for i in range(len(segments) - 1):
            cur = segments[i]
            nxt = segments[i + 1]
            if nxt.kind == SegmentKind.BACKTRACK and cur.kind in (
                SegmentKind.EXPLORE,
                SegmentKind.DECISION,
            ):
                cur.is_dead_end = True
