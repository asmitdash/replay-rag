"""replay-rag — reasoning-trace memory for o1/R1-style agents."""
from .types import (
    Trace,
    Segment,
    SegmentKind,
    DecisionPoint,
    DecisionGraph,
    Tuple as ReplayTuple,
)
from .capture import TraceCapture, capture
from .segmenter import TraceSegmenter
from .decision_graph import DecisionGraphBuilder
from .embedding import Embedder, HashEmbedder, SentenceTransformerEmbedder
from .retrieval import TraceStore, RetrievalEngine
from .splicer import TraceSplicer
from .compression import TraceCompressor
from .evaluation import EvaluationEngine, EvalResult, EvalCase
from .pipeline import ReplayPipeline, PipelineRun
from .providers import ReasoningProvider, MockReasoningProvider, ProviderResponse

__all__ = [
    "Trace",
    "Segment",
    "SegmentKind",
    "DecisionPoint",
    "DecisionGraph",
    "ReplayTuple",
    "TraceCapture",
    "capture",
    "TraceSegmenter",
    "DecisionGraphBuilder",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "TraceStore",
    "RetrievalEngine",
    "TraceSplicer",
    "TraceCompressor",
    "EvaluationEngine",
    "EvalResult",
    "EvalCase",
    "ReplayPipeline",
    "PipelineRun",
    "ReasoningProvider",
    "MockReasoningProvider",
    "ProviderResponse",
]
__version__ = "0.1.0"
