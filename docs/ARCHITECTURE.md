# Architecture — Suyven RAG Pipeline

## System Overview

```
+-----------------------------------------------------------------+
|                        INGESTION PIPELINE                        |
+-----------------------------------------------------------------+
|                                                                  |
|   Documents (MD/TXT/PDF)                                         |
|         |                                                        |
|         v                                                        |
|   +-----------+    ThreadPoolExecutor (8 workers)                |
|   |  Loader   |--- Parallel file reading                         |
|   +-----------+                                                  |
|         |                                                        |
|         v                                                        |
|   +-----------+    600-char chunks, 80-char overlap              |
|   |  Chunker  |--- Character-based text splitting                |
|   +-----------+                                                  |
|         |                                                        |
|         v                                                        |
|   +--------------------------------------+                       |
|   |     GPU Embedding (FP16)             |                       |
|   |                                      |                       |
|   |  Model: all-MiniLM-L6-v2            |                       |
|   |  Precision: FP16 (half)             |                       |
|   |  Batch size: 256 (benchmark-optimal)|                       |
|   |  Device: CUDA (RTX 5070)            |                       |
|   |                                      |                       |
|   |  Throughput: ~2960 chunks/s          |                       |
|   |  Speedup: 38x vs CPU               |                       |
|   |  VRAM: 671 MB (-48% vs FP32)        |                       |
|   +--------------------------------------+                       |
|         |                                                        |
|         v                                                        |
|   +-----------+    Batch inserts (500 chunks/call)               |
|   | ChromaDB  |--- Persistent vector store                       |
|   +-----------+                                                  |
|                                                                  |
+-----------------------------------------------------------------+

+-----------------------------------------------------------------+
|                        QUERY PIPELINE                            |
+-----------------------------------------------------------------+
|                                                                  |
|   User Query                                                     |
|         |                                                        |
|         v                                                        |
|   +----------------+                                             |
|   |  Orchestrator  |--- Deterministic routing (answer/summary/   |
|   |  (RoutePlan)   |    code) based on keywords + query length   |
|   +----------------+                                             |
|         |                                                        |
|         v                                                        |
|   +--------------+    Same FP16 model (cached in GPU)            |
|   | Query Embed  |--- Single embedding, <1ms                     |
|   +--------------+                                               |
|         |                                                        |
|         v                                                        |
|   +--------------+    Cosine similarity -- overfetch x4          |
|   |  Stage 1:    |--- Bi-encoder fast retrieval                  |
|   |  Retriever   |    Top-4k candidates from ChromaDB            |
|   +--------------+                                               |
|         |                                                        |
|         v                                                        |
|   +--------------------------------------+                       |
|   |     Stage 2: Cross-Encoder (FP16)   |                       |
|   |                                      |                       |
|   |  Model: ms-marco-MiniLM-L-6-v2     |                       |
|   |  Precision: FP16 (half)             |                       |
|   |  Device: CUDA (RTX 5070)            |                       |
|   |                                      |                       |
|   |  Pairwise (query, chunk) scoring    |                       |
|   |  Re-ranks candidates -> top-k       |                       |
|   +--------------------------------------+                       |
|         |                                                        |
|         v                                                        |
|   +--------------------------------------+                       |
|   |  LLM Provider Abstraction            |                       |
|   |                                      |                       |
|   |  Provider: Groq (cloud)             |                       |
|   |  Model: llama-3.3-70b-versatile     |                       |
|   |  Protocol: OpenAI-compatible API    |                       |
|   |  Streaming: SSE token-by-token      |                       |
|   |  Avg latency: 3.17s (incl retrieval)|                       |
|   +--------------------------------------+                       |
|         |                                                        |
|         v                                                        |
|   Answer + Source References + Route Badge (mode)                |
|                                                                  |
+-----------------------------------------------------------------+

+-----------------------------------------------------------------+
|                     GPU METRICS DASHBOARD                        |
+-----------------------------------------------------------------+
|                                                                  |
|   pynvml (NVML bindings)                                         |
|         |                                                        |
|         v                                                        |
|   +--------------------------------------+                       |
|   |  Real-time GPU Monitoring            |                       |
|   |                                      |                       |
|   |  * Device name (RTX 5070)           |                       |
|   |  * Temperature (C)                  |                       |
|   |  * GPU utilization (%)              |                       |
|   |  * VRAM used / total (GB + bar)     |                       |
|   +--------------------------------------+                       |
|         |                                                        |
|         v                                                        |
|   React sidebar + Streamlit sidebar                              |
|                                                                  |
+-----------------------------------------------------------------+
```

## Key Design Decisions

| Decision | Rationale | Validated By |
|----------|-----------|--------------|
| FP16 bi-encoder | +109% throughput, -48% VRAM | `BENCHMARK_RESULTS.md` |
| Batch size = 256 | Optimal from sweep (256-2048) | `benchmark.py` |
| FP16 quality validated | 99.3% Recall@10 maintained | `QUALITY_VALIDATION.md` |
| 2-stage retrieval | Cross-encoder reranks bi-encoder candidates | `RERANKER_QUALITY.md` |
| Overfetch x4 | Ensures reranker has enough candidates to promote | Retrieval experiments |
| FP16 cross-encoder | GPU-accelerated pairwise reranking | `RERANKER_BENCHMARK.md` |
| Parallel file I/O | Eliminates read bottleneck | Phase 1 timing |
| Groq cloud for LLM | 70B model quality, GPU VRAM reserved for embed+reranker | `LLM_EVAL.md` |
| LLM provider abstraction | Swap between Ollama (local) and any OpenAI-compatible API | `rag/llm.py` |
| Orchestrator routing | Deterministic mode classification (answer/summary/code) | `validate_v2_1.py`, 48 unit tests |
| Model/index registries | Decouple pipeline from concrete model/collection instances | `rag/model_registry.py`, `rag/index_registry.py` |
| sentence-transformers | Direct GPU control, no HTTP overhead | vs Ollama API |
| pynvml GPU dashboard | Real-time VRAM / temp / utilization monitoring | Sidebar UI |

## File Structure

```
suyven-v1/
├── api.py                      # FastAPI backend (SSE streaming, orchestrator)
├── app.py                      # Streamlit UI (ingest + query + GPU dashboard)
├── ingest.py                   # CLI ingestion
├── requirements.txt            # Python dependencies
│
├── rag/
│   ├── config.py               # Centralized configuration (env vars)
│   ├── orchestrator.py         # RoutePlan + deterministic query routing
│   ├── model_registry.py       # Embed/reranker model registry (singleton)
│   ├── index_registry.py       # ChromaDB collection registry
│   ├── llm.py                  # LLM provider abstraction (Ollama + OpenAI)
│   ├── store.py                # Embedding + ChromaDB storage (FP16 GPU)
│   ├── loader.py               # File reader (MD, TXT, PDF)
│   ├── chunker.py              # Text chunking
│   ├── pipeline.py             # Shared read+chunk logic
│   └── monitoring.py           # GPU metrics via pynvml
│
├── tests/
│   ├── test_config.py          # Config loading and types
│   ├── test_orchestrator.py    # Mode detection, RoutePlan, edge cases
│   ├── test_model_registry.py  # Registry contents and interface
│   └── test_index_registry.py  # Routing and index info
│
├── suyven front/               # React frontend (Vite)
│   └── src/
│       ├── App.jsx             # Chat UI + sidebar + route badges
│       └── index.css           # Styling
│
├── benchmark.py                # Bi-encoder GPU performance benchmark
├── benchmark_reranker.py       # Cross-encoder reranker benchmark
├── validate_quality.py         # FP32 vs FP16 embedding fidelity
├── validate_reranker.py        # Reranker quality (NDCG, ground truth)
├── validate_v2_1.py            # V2.1 orchestrator routing validation
├── eval_llm.py                 # LLM quality evaluation (10-query suite)
│
├── BENCHMARK_RESULTS.md        # Bi-encoder performance report
├── RERANKER_BENCHMARK.md       # Reranker performance report
├── QUALITY_VALIDATION.md       # FP16 fidelity report
├── RERANKER_QUALITY.md         # Reranker quality report
├── LLM_EVAL.md                 # LLM evaluation (Groq/llama-3.3-70b)
├── ROADMAP_V2.md               # Evidence-gated roadmap
├── ARCHITECTURE.md             # This file
└── data/
    └── chroma/                 # Persistent vector DB
```
