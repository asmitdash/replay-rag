"""Trace store + Trace Retrieval Engine.

Default backend is a pure-numpy in-memory store: small footprint, zero
runtime deps beyond what's already required, fast for the trace counts a
real reasoning loop accumulates (hundreds-to-low-thousands per workspace).

Chroma is available as an optional backend — useful when traces need to
persist across processes. Either backend exposes the same .add() / .search()
contract, so the retrieval engine doesn't care which is in use.

Retrieval embeds the *current* problem (with the same build_index_text path
used at write time) and returns top-k nearest traces by cosine similarity.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional

import numpy as np

from .embedding import Embedder, build_index_text, default_embedder
from .types import (
    DecisionGraph,
    DecisionPoint,
    Segment,
    SegmentKind,
    Trace,
)


class TraceStore(ABC):
    @abstractmethod
    def add(self, trace: Trace, vector: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[Trace, float]]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class InMemoryTraceStore(TraceStore):
    def __init__(self) -> None:
        self._traces: list[Trace] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim), L2-normalized rows

    def add(self, trace: Trace, vector: np.ndarray) -> None:
        v = self._normalize(vector)
        if self._matrix is None:
            self._matrix = v.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, v.reshape(1, -1)])
        self._traces.append(trace)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Trace, float]]:
        if self._matrix is None or len(self._traces) == 0:
            return []
        q = self._normalize(vector)
        sims = (self._matrix @ q).astype(float)
        k = min(k, len(self._traces))
        # argsort descending; argpartition would be faster but k is tiny
        idxs = np.argsort(-sims)[:k]
        return [(self._traces[i], float(sims[i])) for i in idxs]

    def __len__(self) -> int:
        return len(self._traces)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n == 0:
            return v
        return v / n

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "traces": [_trace_to_dict(t) for t in self._traces],
            "matrix": self._matrix.tolist() if self._matrix is not None else [],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self._traces = [_trace_from_dict(d) for d in payload["traces"]]
        m = payload["matrix"]
        self._matrix = np.asarray(m, dtype=np.float32) if m else None


class ChromaTraceStore(TraceStore):
    """Persistent Chroma-backed store. Optional dep."""

    def __init__(self, persist_dir: str = ".replay_rag_chroma", collection: str = "traces") -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise ImportError("ChromaTraceStore requires chromadb (already in core deps)") from e
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._coll = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._traces: dict[str, Trace] = {}

    def add(self, trace: Trace, vector: np.ndarray) -> None:
        self._coll.add(
            ids=[trace.trace_id],
            embeddings=[vector.astype(float).tolist()],
            documents=[build_index_text(trace)],
            metadatas=[{"problem": trace.problem[:512], "answer": trace.answer[:512]}],
        )
        self._traces[trace.trace_id] = trace

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Trace, float]]:
        if len(self._traces) == 0:
            return []
        res = self._coll.query(query_embeddings=[vector.astype(float).tolist()], n_results=min(k, len(self._traces)))
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[tuple[Trace, float]] = []
        for tid, dist in zip(ids, dists):
            t = self._traces.get(tid)
            if t is None:
                continue
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            out.append((t, sim))
        return out

    def __len__(self) -> int:
        return len(self._traces)


class RetrievalEngine:
    """High-level retrieval API.

    Holds an embedder and a store. .index(trace) embeds and writes;
    .retrieve(problem, k) returns the top-k traces ranked by similarity
    of (problem + decision-point summaries) — that's where the
    decision-point-aware retrieval lives.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[TraceStore] = None,
        min_similarity: float = 0.15,
    ) -> None:
        self.embedder = embedder if embedder is not None else default_embedder()
        self.store = store if store is not None else InMemoryTraceStore()
        self.min_similarity = min_similarity

    def index(self, trace: Trace) -> np.ndarray:
        text = build_index_text(trace)
        vec = self.embedder.embed(text)
        self.store.add(trace, vec)
        return vec

    def retrieve(self, problem: str, k: int = 3) -> list[tuple[Trace, float]]:
        # Query embedding uses the problem alone (no past graph available).
        # That's an acceptable asymmetry: write-side is richer (problem +
        # decision summaries), query-side is just problem. Both share the
        # problem-text axis, which is enough to bring related items into
        # the top ranks; final ranking uses cosine sim, which weights the
        # shared axes.
        qvec = self.embedder.embed(problem)
        hits = self.store.search(qvec, k=k)
        return [(t, s) for t, s in hits if s >= self.min_similarity]

    def __len__(self) -> int:
        return len(self.store)


# --- (de)serialization helpers for InMemoryTraceStore.save/load ---


def _trace_to_dict(t: Trace) -> dict:
    return {
        "problem": t.problem,
        "reasoning": t.reasoning,
        "answer": t.answer,
        "trace_id": t.trace_id,
        "metadata": t.metadata,
        "token_count": t.token_count,
        "segments": [
            {
                "index": s.index,
                "kind": s.kind.value,
                "text": s.text,
                "start_char": s.start_char,
                "end_char": s.end_char,
                "summary": s.summary,
                "is_dead_end": s.is_dead_end,
            }
            for s in t.segments
        ],
        "graph": _graph_to_dict(t.graph) if t.graph else None,
    }


def _graph_to_dict(g: DecisionGraph) -> dict:
    return {
        "edges": [list(e) for e in g.edges],
        "decision_points": [asdict(dp) for dp in g.decision_points],
    }


def _trace_from_dict(d: dict) -> Trace:
    segments = [
        Segment(
            index=s["index"],
            kind=SegmentKind(s["kind"]),
            text=s["text"],
            start_char=s["start_char"],
            end_char=s["end_char"],
            summary=s.get("summary"),
            is_dead_end=s.get("is_dead_end", False),
        )
        for s in d.get("segments", [])
    ]
    graph = None
    if d.get("graph"):
        gd = d["graph"]
        graph = DecisionGraph(
            nodes=segments,
            edges=[(int(a), int(b), str(k)) for a, b, k in gd.get("edges", [])],
            decision_points=[DecisionPoint(**dp) for dp in gd.get("decision_points", [])],
        )
    t = Trace(
        problem=d["problem"],
        reasoning=d["reasoning"],
        answer=d["answer"],
        trace_id=d["trace_id"],
        segments=segments,
        graph=graph,
        metadata=d.get("metadata", {}),
        token_count=d.get("token_count", 0),
    )
    return t
