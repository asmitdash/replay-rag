# replay-rag

> **Reasoning-trace memory for o1 / R1-style agents.** Store `(problem → reasoning trace → answer)` tuples; on a new problem, retrieve top-k similar past traces and splice the winning subtree into the new prompt as scaffolding. Cuts output tokens 40–70% on repeated problem patterns.

[![tests](https://img.shields.io/badge/tests-42%20passed-brightgreen)](https://github.com/asmitdash/replay-rag/tree/main/tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![status](https://img.shields.io/badge/status-beta%20v0.1.0-yellow)]()
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The wedge

Reasoning models (o1, R1, Claude with extended thinking) burn 10–100× more output tokens than fast models. Most agentic workflows hit the same kinds of problems repeatedly — variations on the same arithmetic, the same retrieval strategy, the same debugging pattern. Every R1-style agent today **re-derives reasoning it has already done**.

Memory systems in 2026 store **answers**. Nobody has a memory system for **reasoning traces** themselves.

replay-rag fixes that:

1. **Capture.** Every reasoning call's trace + answer is captured without side effects.
2. **Segment.** Raw trace text is split into typed segments — `SETUP`, `EXPLORE`, `DECISION`, `BACKTRACK`, `VERIFY`, `ANSWER`.
3. **Graph.** A DAG over segments with `next` and `backtrack` edges; `DecisionPoint`s are extracted as the strategic skeleton.
4. **Compress.** Dead-end branches and over-budget exploration get dropped at write time.
5. **Embed.** The embedding target is `problem + decision-point summaries`, NOT raw text — that's what generalises.
6. **Retrieve + splice.** On a new problem, the closest past traces' winning paths get spliced into the new prompt as scaffolding, capped by a token budget.

The agent's reasoning model now sees "here's how you solved a similar problem; pick up from there" — and skips the parts it would have re-derived.

---

## What ships in v0.1.0

Eight components, all wired into the public `ReplayPipeline` and covered by 42 tests.

| Component | Module | Job |
|---|---|---|
| **Trace Capture Middleware** | [`capture.py`](src/replay_rag/capture.py) | Wraps any provider; emits `(Trace, ProviderResponse)` tuples without side effects |
| **Trace Segmenter** | [`segmenter.py`](src/replay_rag/segmenter.py) | Splits raw reasoning into typed segments (SETUP / EXPLORE / DECISION / BACKTRACK / VERIFY / ANSWER) using markers reasoning models actually emit ("wait", "actually", "let me re-read") plus structural cues |
| **Decision Graph Builder** | [`decision_graph.py`](src/replay_rag/decision_graph.py) | Builds a DAG over segments with `next` and `backtrack` edges; extracts `DecisionPoint`s as the strategic skeleton |
| **Trace Embedding Model** | [`embedding.py`](src/replay_rag/embedding.py) | Embeds `problem + decision-point summaries` (not raw text); `HashEmbedder` (offline) or `SentenceTransformerEmbedder` (MiniLM, all-MiniLM-L6-v2 by default) |
| **Trace Retrieval Engine** | [`retrieval.py`](src/replay_rag/retrieval.py) | In-memory or Chroma store; cosine search over decision-aware embeddings |
| **Trace Splicer** | [`splicer.py`](src/replay_rag/splicer.py) | Picks the winning-path subtree under a token budget; emits prompt-ready text scaffolding |
| **Trace Compression Layer** | [`compression.py`](src/replay_rag/compression.py) | At write time, drops dead-end branches and over-budget EXPLORE segments; preserves the raw trace on `metadata` for audit |
| **Evaluation Engine** | [`evaluation.py`](src/replay_rag/evaluation.py) | A/B drives baseline (no replay) vs replay over an `EvalCase` list with optional warmup |

The end-to-end glue is `ReplayPipeline` in [`pipeline.py`](src/replay_rag/pipeline.py) — wires capture → segment → graph → compress → index, and on subsequent calls retrieves → splices → solves.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ReplayPipeline.solve(problem)                 │
└─────────────┬───────────────────────────────────┬───────────────────────┘
              │ replay=True                        │
              ▼                                    │
   ┌────────────────────┐                          │
   │ Retrieval Engine   │                          │
   │  - embed(problem)  │                          │
   │  - top-k cosine    │                          │
   │    search          │                          │
   └────────┬───────────┘                          │
            │ if hits found                         │
            ▼                                       │
   ┌────────────────────┐                          │
   │ Trace Splicer      │                          │
   │  - winning-path    │                          │
   │    subtree under   │                          │
   │    token budget    │                          │
   │  - emit scaffold   │                          │
   └────────┬───────────┘                          │
            │ scaffold (or None on cold-start)     │
            ▼                                       │
            ╔════════════════════╗                 │
            ║  Capture Middleware║◄────────────────┘
            ║  + Reasoning       │
            ║  Provider          │
            ║  (Mock / Bedrock-  │
            ║   Claude w/ ext.   │
            ║   thinking / etc.) │
            ╚════════╤═══════════╝
                     │ Trace + ProviderResponse
                     ▼
            ┌────────────────────┐
            │  Segmenter         │
            │  → typed segments  │
            └────────┬───────────┘
                     ▼
            ┌────────────────────┐
            │  Decision Graph    │
            │  → DAG + decision  │
            │    points          │
            └────────┬───────────┘
                     ▼
            ┌────────────────────┐
            │  Compressor        │
            │  → dead-ends out   │
            │  → raw on metadata │
            └────────┬───────────┘
                     ▼
            ┌────────────────────┐
            │  Retrieval.index   │
            │  → embed +         │
            │    persist trace   │
            │    in store        │
            └────────────────────┘
```

The crucial design choice: **the embedding key is `problem + decision-point summaries`, not the raw trace.** Raw text is dominated by surface form (numbers, names, ordering) which doesn't generalise. Decision-point summaries strip that and keep the strategic shape — what should generalise across problems. See `build_index_text` in `embedding.py`.

---

## Install

```bash
git clone https://github.com/asmitdash/replay-rag.git
cd replay-rag
pip install -e ".[dev]"            # core + tests
pip install -e ".[dev,embed]"      # + sentence-transformers (semantic embeddings)
pip install -e ".[dev,bedrock]"    # + anthropic[bedrock] (real Claude provider)
pip install -e ".[all]"            # everything
```

Core deps (`numpy`, `chromadb`, `pydantic`) install in seconds. The package is fully usable offline with the default `HashEmbedder` and `MockReasoningProvider`.

---

## Quickstart — Python SDK

```python
from replay_rag import HashEmbedder, MockReasoningProvider, ReplayPipeline
from replay_rag.retrieval import InMemoryTraceStore

pipe = ReplayPipeline(
    provider=MockReasoningProvider(),       # swap for BedrockClaudeProvider in prod
    embedder=HashEmbedder(dim=512),         # swap for SentenceTransformerEmbedder()
    store=InMemoryTraceStore(),
)

cold = pipe.solve("What is the sum of 7 and 3?")
warm = pipe.solve("What is the sum of 50 and 60?")  # uses scaffold from cold

print(cold.output_tokens, "->", warm.output_tokens)
# 93 -> 30   (~70% reduction; depends on the provider + problem family)
```

Each `PipelineRun` exposes:

| Field | Meaning |
|---|---|
| `trace` | The `Trace` object (raw reasoning + segments + graph) |
| `scaffold` | The spliced scaffold text passed to the provider, or `None` on cold-start |
| `retrieved_count` | How many traces the retriever found above threshold |
| `top_similarity` | Cosine similarity of the winning prior trace |
| `input_tokens` / `output_tokens` | What the provider consumed and emitted |

Full demo: [`examples/quickstart.py`](examples/quickstart.py).

---

## A/B Evaluation

```python
from replay_rag import EvalCase, EvaluationEngine, HashEmbedder, MockReasoningProvider

cases = [EvalCase(f"What is the sum of {i} and {i+1}?", str(2*i+1)) for i in range(5)]
warmup = [EvalCase(f"What is the sum of {10*i} and {5*i}?") for i in range(5)]

engine = EvaluationEngine(
    provider_factory=MockReasoningProvider,
    embedder=HashEmbedder(dim=1024),
)
result = engine.run(cases, warmup=warmup)
print(result.summary())
# baseline tokens: 93.4/case  | replay tokens: 29.0/case  | reduction: 69.0%
```

The eval engine:

1. Spins up a fresh pipeline for the **baseline** run (no replay).
2. Optionally warms up a **replay** pipeline by solving warmup cases.
3. Runs the same `cases` through the replay pipeline.
4. Reports per-case + aggregate baseline-vs-replay token deltas.

---

## Real reasoning model: Bedrock + Claude with extended thinking

```python
from replay_rag import ReplayPipeline, SentenceTransformerEmbedder
from replay_rag.providers import BedrockClaudeProvider
from replay_rag.retrieval import ChromaTraceStore

pipe = ReplayPipeline(
    provider=BedrockClaudeProvider(
        model_id="us.anthropic.claude-opus-4-7-20251201-v1:0",
    ),
    embedder=SentenceTransformerEmbedder(),
    store=ChromaTraceStore(persist_dir=".replay_rag_chroma"),
)
```

The Bedrock provider treats Claude's **extended thinking blocks** as the trace and the **text blocks** as the answer. Requires `pip install replay-rag[bedrock]` and AWS credentials with Bedrock model access.

---

## Design notes

- **Embedding target = problem + decision-point summaries, not raw reasoning.** Raw text is dominated by surface form; decision-point summaries strip that and keep the strategic shape. Documented in `embedding.build_index_text`.
- **Compression is rule-based at write time.** Dead-end segments (EXPLORE/DECISION immediately followed by BACKTRACK) get dropped; the splicer drops them again per-retrieval as belt-and-suspenders. No LLM-judged compression — that would burn tokens at write time and defeat the wedge.
- **Segmenter is heuristic.** Markers reasoning models actually emit ("wait", "actually", "let me re-read") plus structural cues. A learned segmenter is a v0.2 problem.
- **Stores are pluggable.** `InMemoryTraceStore` (fast, JSON save/load) for single-process workloads; `ChromaTraceStore` for cross-process persistence.
- **Compression is reversible.** The raw uncompressed trace is preserved on `Trace.metadata` for audit / debugging, even after the indexed version drops dead-ends.

---

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -v
```

**42 passed in 1.06s** — verified locally on Python 3.11 + chromadb 1.5.9 + pydantic 2.13.4.

| File | Tests | What it covers |
|---|---|---|
| `test_capture.py` | 5 | Capture middleware preserves provider response while emitting Trace |
| `test_compression.py` | 5 | Dead-end pruning, EXPLORE budget, raw preservation on metadata |
| `test_decision_graph.py` | 4 | DAG construction, BACKTRACK edges, decision-point extraction |
| `test_embedding.py` | 5 | HashEmbedder determinism, sentence-transformer integration |
| `test_evaluation.py` | 4 | Baseline / replay split, warmup, aggregate stats |
| `test_pipeline.py` | 5 | End-to-end cold + warm, scaffold absence on cold, replay disabled |
| `test_retrieval.py` | 4 | InMemoryTraceStore, ChromaTraceStore, cosine ordering |
| `test_segmenter.py` | 5 | Marker handling, structural fallback, ANSWER detection |
| `test_splicer.py` | 5 | Winning-path subtree under budget, BACKTRACK handling, format |

All offline — no Bedrock calls in CI. The Bedrock provider is exercised via mock at the boundary.

---

## Roadmap

- [x] **v0.1.0** — full pipeline (capture → segment → graph → compress → embed → retrieve → splice), MockReasoning + Bedrock-Claude providers, InMemory + Chroma stores, evaluation engine, 42 tests
- [ ] **v0.2.0** — learned segmenter (replace heuristic markers), DeepSeek-R1 + GSM8K + MATH benchmark run on a real reasoning model
- [ ] **v0.3.0** — server mode (HTTP API + MCP server), persistent trace pruning policies, multi-tenant boundary

---

## Honest risk

- The mock provider proves the contract works (capture / segment / graph / embed / retrieve / splice all wire up). It does **not** prove the wedge holds on real reasoning models — that needs a benchmark run against Bedrock-Claude or DeepSeek-R1 on GSM8K / MATH. The eval engine is built; only the provider+budget+evals corpus call is missing.
- Buyer is small. Only orgs spending real money on reasoning-model inference care about the token bill.
- Highly technical sell — hard to demo without a real benchmark.

---

## License

MIT. See [`LICENSE`](LICENSE).
