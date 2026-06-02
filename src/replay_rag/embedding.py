"""Trace Embedding Model.

The pitch's novelty: embed *decision-point summaries*, not raw traces.
Why — raw reasoning text is dominated by surface form (numbers, names,
restated problem). Decision-point summaries strip surface form and keep
the strategic shape ("considered subtraction, chose addition, verified
by recomputation"), which is what should generalize across problems.

Two backends:
- HashEmbedder: deterministic, dependency-free, n-gram hashing into a
  fixed-dimension float vector. Cosine similarity is meaningful because
  shared n-grams collide on the same axes. Used by tests and as a fallback
  when sentence-transformers isn't installed.
- SentenceTransformerEmbedder: real semantic embeddings via
  all-MiniLM-L6-v2 (384-dim), only loaded if the optional dep is present.

Either backend exposes the same .embed(text) -> np.ndarray contract.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from .types import DecisionPoint, Segment, SegmentKind, Trace


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))


class HashEmbedder(Embedder):
    """Hashed n-gram embedder. Deterministic, offline, ~free.

    Cosine similarity over hashed-ngram vectors is a well-understood baseline
    (a la HashingVectorizer in scikit-learn). Good enough for retrieval
    over hundreds-to-low-thousands of traces in a 3-week MVP.
    """
    _TOKEN = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.dim = dim
        self._ngram_range = ngram_range

    def embed(self, text: str) -> np.ndarray:
        tokens = [t.lower() for t in self._TOKEN.findall(text)]
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vec
        lo, hi = self._ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                idx = self._hash(gram) % self.dim
                sign = 1.0 if (self._hash(gram + "#sign") & 1) == 0 else -1.0
                vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _hash(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class SentenceTransformerEmbedder(Embedder):
    """all-MiniLM-L6-v2 embeddings. Imported lazily."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires `pip install replay-rag[embed]`"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def build_index_text(trace: Trace) -> str:
    """The text that gets embedded for retrieval — the project's central design choice.

    We concatenate:
    - the problem statement (anchors topical similarity)
    - decision-point summaries (anchors strategic similarity)
    - segment summaries for VERIFY and ANSWER segments (captures the shape
      of a successful solution)

    Crucially: we do NOT include the raw reasoning text. That keeps the
    embedding focused on shape, not surface form, which is what makes
    cross-problem retrieval generalize.
    """
    parts: list[str] = [trace.problem.strip()]
    if trace.graph is not None:
        for dp in trace.graph.decision_points:
            opts = " | ".join(dp.options_considered)
            chose = dp.chosen or ""
            parts.append(f"considered: {opts}; chose: {chose}")
    for seg in trace.segments:
        if seg.kind in (SegmentKind.VERIFY, SegmentKind.ANSWER):
            parts.append(f"{seg.kind.value}: {seg.summary}")
    return "\n".join(p for p in parts if p)


def default_embedder() -> Embedder:
    """Pick the best available embedder without forcing an install."""
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return HashEmbedder()
