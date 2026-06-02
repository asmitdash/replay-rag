"""End-to-end pipeline glue.

Wires the components into one user-facing object so callers don't have to
manually compose capture -> segment -> graph -> compress -> index -> retrieve
-> splice. This is the path the evaluation engine uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .capture import TraceCapture
from .compression import TraceCompressor
from .decision_graph import DecisionGraphBuilder
from .embedding import Embedder
from .providers import ReasoningProvider
from .retrieval import RetrievalEngine, TraceStore
from .segmenter import TraceSegmenter
from .splicer import TraceSplicer
from .types import Trace


@dataclass
class PipelineRun:
    """One end-to-end pass: retrieve -> splice -> solve -> capture -> compress -> index."""
    trace: Trace
    scaffold: Optional[str]
    retrieved_count: int
    top_similarity: float
    input_tokens: int
    output_tokens: int


@dataclass
class ReplayPipeline:
    provider: ReasoningProvider
    embedder: Optional[Embedder] = None
    store: Optional[TraceStore] = None
    segmenter: TraceSegmenter = field(default_factory=TraceSegmenter)
    graph_builder: DecisionGraphBuilder = field(default_factory=DecisionGraphBuilder)
    compressor: TraceCompressor = field(default_factory=TraceCompressor)
    splicer: TraceSplicer = field(default_factory=TraceSplicer)
    top_k: int = 3
    capture: TraceCapture = field(init=False)
    retrieval: RetrievalEngine = field(init=False)

    def __post_init__(self) -> None:
        self.capture = TraceCapture(provider=self.provider)
        # Pass through None vs explicit value cleanly — RetrievalEngine
        # treats None as "use default" and any concrete instance as the
        # user's choice, even when empty.
        self.retrieval = RetrievalEngine(embedder=self.embedder, store=self.store)

    def solve(self, problem: str, *, replay: bool = True) -> PipelineRun:
        scaffold: Optional[str] = None
        retrieved_count = 0
        top_sim = 0.0
        if replay and len(self.retrieval) > 0:
            hits = self.retrieval.retrieve(problem, k=self.top_k)
            retrieved_count = len(hits)
            if hits:
                top_sim = hits[0][1]
                scaffold = self.splicer.splice_top_k(hits)
        trace, resp = self.capture.solve(problem, scaffold=scaffold)
        # Annotate, segment, graph, compress, then index.
        trace.segments = self.segmenter.segment(trace.reasoning)
        trace.graph = self.graph_builder.build(trace.segments)
        self.compressor.compress(trace)
        self.retrieval.index(trace)
        return PipelineRun(
            trace=trace,
            scaffold=scaffold,
            retrieved_count=retrieved_count,
            top_similarity=top_sim,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
