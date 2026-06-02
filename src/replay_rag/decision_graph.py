"""Decision Graph Builder.

Builds a directed graph over segments. Edges:
- "next" — sequential reasoning flow
- "backtrack" — from a BACKTRACK segment back to the segment before the
  dead-end it abandoned

DecisionPoints are derived: every EXPLORE/DECISION segment whose neighbor
is a BACKTRACK is treated as one option in a choice; the kept path is the
segment(s) after the BACKTRACK up to the next BACKTRACK or ANSWER.

The graph is what the splicer walks to extract a 'winning path' for
scaffolding. Compression also reads it to drop dead-end segments.
"""
from __future__ import annotations

from .types import DecisionGraph, DecisionPoint, Segment, SegmentKind


class DecisionGraphBuilder:
    def build(self, segments: list[Segment]) -> DecisionGraph:
        graph = DecisionGraph(nodes=list(segments))
        for i in range(len(segments) - 1):
            graph.edges.append((i, i + 1, "next"))
        # Backtrack edges: BACKTRACK -> the segment immediately preceding
        # the dead-end it cancels (so a graph walker can rewind to the
        # last good state).
        for i, seg in enumerate(segments):
            if seg.kind != SegmentKind.BACKTRACK:
                continue
            j = i - 1
            while j >= 0 and segments[j].is_dead_end:
                j -= 1
            if j >= 0:
                graph.edges.append((i, j, "backtrack"))
        graph.decision_points = self._extract_decision_points(segments)
        return graph

    @staticmethod
    def _extract_decision_points(segments: list[Segment]) -> list[DecisionPoint]:
        points: list[DecisionPoint] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg.is_dead_end:
                # Walk forward over consecutive dead-ends and the BACKTRACK
                # to the next non-backtrack segment — that's the chosen
                # option of this decision point.
                options = [seg.summary or seg.text[:80]]
                j = i + 1
                while j < len(segments) and segments[j].is_dead_end:
                    options.append(segments[j].summary or segments[j].text[:80])
                    j += 1
                # Skip the BACKTRACK marker itself.
                if j < len(segments) and segments[j].kind == SegmentKind.BACKTRACK:
                    j += 1
                chosen = None
                if j < len(segments):
                    chosen = segments[j].summary or segments[j].text[:80]
                points.append(
                    DecisionPoint(
                        segment_index=seg.index,
                        options_considered=options + ([chosen] if chosen else []),
                        chosen=chosen,
                        led_to_dead_end=False,
                    )
                )
                i = j + 1
            else:
                i += 1
        return points
