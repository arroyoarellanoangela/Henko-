# Kaizen v1 — GPU-Optimized RAG Engine

Production-ready RAG engine built on two principles:
1. **GPU-first**: Every bottleneck (embedding, reranking, batching) runs on CUDA with FP16
2. **Domain factory**: One engine serves unlimited isolated knowledge domains, each with its own vector index and system prompt

## Benchmarks

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.909 (209-query ground-truth suite) |
| Embedding throughput | 2,960 chunks/s (38x CPU baseline) |
| FP16 VRAM savings | -48% vs FP32 (671 MB vs 1,290 MB) |
| FP16 retrieval fidelity | 99.3% Recall@10 vs FP32 |
| Reranker latency | 8.8 ms/query (FP16 on GPU) |
| Test suite | 209/209 passing |

**Hardware target:** NVIDIA RTX 5070 · CUDA 12.8 · PyTorch 2.12

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115 + SSE streaming + Uvicorn |
| Embeddings | sentence-transformers 5.0+ / BAAI/bge-m3 (568M params, 1024-dim, multilingual, FP16) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (FP16 on GPU) |
| Vector Store | ChromaDB 1.5+ (persistent, cosine HNSW, multi-collection) |
| Hybrid Search | rank-bm25 (BM25Okapi) + Reciprocal Rank Fusion (RRF) |
| LLM | Any OpenAI-compatible endpoint (default: Groq llama-3.3-70b-versatile) or Ollama local |
| GPU Monitoring | pynvml 13.0+ |
| Infrastructure | Docker (CPU + GPU variants) |
| Python | 3.12 · PyTorch 2.12+ · CUDA 12.8 |

## Architecture

```
kaizen-v1/
├── api.py                    # FastAPI + SSE streaming
├── app.py                    # Streamlit UI + GPU dashboard
├── ingest.py                 # CLI ingestion (3-phase pipeline)
├── query.py                  # CLI query tool
│
├── rag/                      # Core RAG pipeline (21 modules)
│   ├── config.py             # Centralized config (env vars + Docker secrets)
│   ├── llm.py                # LLM abstraction (Ollama local + OpenAI-compatible cloud)
│   ├── agents.py             # 4-agent pipeline (Router, Retriever, Generator, Evaluator)
│   ├── orchestrator.py       # RoutePlan + hybrid search (BM25 + dense) + RRF fusion
│   ├── model_registry.py     # Embed/reranker model singleton registry
│   ├── index_registry.py     # ChromaDB collection registry (static + dynamic)
│   ├── domain_registry.py    # Domain CRUD + isolation (multi-domain)
│   ├── store.py              # Embedding + ChromaDB storage (FP16 GPU)
│   ├── pipeline.py           # Shared read+chunk pipeline
│   ├── chunker.py            # Character + paragraph + sentence chunking
│   ├── loader.py             # Multi-format reader (MD, TXT, PDF, PY, JSONL)
│   ├── eval.py               # Auto-evaluation + quality flagging + query log
│   ├── gap_tracker.py        # Knowledge gap analysis from query logs
│   ├── monitoring.py         # GPU telemetry via pynvml
│   ├── observability.py      # Structured logging + metrics + request tracing
│   ├── security.py           # Auth, CORS, rate limiting, input validation
│   ├── self_improve.py       # Auto-improvement from eval data
│   └── vector_store.py       # Vector store abstraction
│
├── finetune/                 # Embedding fine-tuning (LoRA, A/B testing)
├── tests/                    # 209-test pytest suite
├── benchmarks/               # Performance & quality benchmarks
├── docs/                     # Architecture, roadmap, benchmark reports
├── frontend/                 # React + Vite frontend
├── loadtest/                 # Locust load testing
├── scripts/                  # Deployment scripts
│
├── data/
│   ├── chroma/               # ChromaDB persistent storage (HNSW)
│   ├── domains/              # Domain configs + isolated indexes
│   ├── eval/                 # Query evaluation logs (JSONL)
│   └── knowledge/            # Source documents for ingestion
│
├── docker-compose.yml        # CPU deployment
├── docker-compose.gpu.yml    # GPU (NVIDIA) deployment
├── Dockerfile / Dockerfile.gpu
├── requirements.txt
└── .env.example
```

## Core Modules

### rag/llm.py — LLM Abstraction
Supports **Ollama** (local) and **any OpenAI-compatible API** (Groq, DeepSeek, Together AI, etc.).

- `quick_complete(prompt, model, provider, api_url, api_key, max_tokens, timeout)` — non-streaming completion for query expansion
- `stream_chat(query, context, system_prompt, provider, api_url, api_key)` — token-by-token streaming for RAG responses
- Detects provider at runtime: `POST /api/chat` for Ollama, `POST /chat/completions` for OpenAI-compatible

### rag/store.py — GPU Embedding & ChromaDB Storage
- `get_embed_model()` — lazy-loads BAAI/bge-m3, applies FP16 + CUDA (+109% throughput, -48% VRAM)
- `embed_batch(texts)` — batch embedding (batch_size=256, optimal from sweep)
- `add_chunks(collection, path, chunks, knowledge_dir)` — batch insert with deduplication by MD5
- Custom `STEmbedFn` bridges sentence-transformers to ChromaDB

### rag/orchestrator.py — Query Router & Hybrid Search
**RoutePlan** decides index, models, and retrieval mode (answer/summary/code).

**Search pipeline (5 stages):**
1. Dense (bi-encoder) search — BAAI/bge-m3 semantic similarity
2. BM25 keyword search — catches exact names, acronyms
3. Hybrid merge via RRF — industry-standard rank fusion
4. Cross-encoder reranking — ms-marco-MiniLM (+7.0% Precision@5)
5. Source diversity cap — max 3 chunks per source

**Advanced features:**
- `expand_query()` — LLM-driven query reformulation (~200ms on Groq) + late fusion via RRF
- `_fetch_adjacent_chunks()` — A-RAG-inspired: fetch neighboring chunks for fuller context
- Overfetch factor 6 (tuned; 4 too conservative, 8+ hurt NDCG)

### rag/agents.py — 4-Agent Pipeline
Sequential execution with retry escalation:

1. **RouterAgent** — classifies complexity (simple/moderate/complex), picks strategy (dense/hybrid/category-filtered)
2. **RetrieverAgent / ReACTRetrieverAgent** — multi-tool reasoning (heuristic, no LLM cost):
   - Tool 1: semantic_search (dense)
   - Tool 2: entity_search (if quality weak)
   - Tool 3: sub_query (decompose complex queries)
   - Tool 4: chunk_read (expand with adjacent chunks)
3. **GeneratorAgent** — streams tokens, adapts prompt to retrieval quality (good/weak/failed)
4. **EvaluatorAgent** — flags issues, decides retry with escalated strategy

**AgentContext** carries state: query, route, results, context_text, reranker_scores, response, flags, timing, trace.

### rag/pipeline.py + rag/chunker.py + rag/loader.py — Ingestion
- `read_and_chunk(path)` — atomic for ThreadPoolExecutor
- Character-based chunking (600 chars, 80 overlap) with paragraph + sentence boundaries
- MD5 hashing for deduplication
- Supports MD, TXT, PDF (PyMuPDF), PY, JSONL

### rag/model_registry.py — Model Singleton Registry
- Caches loaded models globally (embed + reranker)
- `get_embed_model(name)` / `get_reranker(name)` — lazy-load with FP16 + CUDA
- `register_embed_model(name, model_id, precision)` — supports domain fine-tuning

### rag/index_registry.py — ChromaDB Collection Registry
- Static + dynamic indexes with lazy loading
- `get_index(name)` / `register_index()` / `reset_index()`
- `route_to_index(query, hint)` — picks index by domain hint
- Custom `RegistryEmbedFn` enables dynamic model swaps

### rag/domain_registry.py — Multi-Domain Isolation
- `create_domain(name, description, system_prompt, categories)` — isolated ChromaDB collection + config
- `update_domain()` / `delete_domain()` / `get_domain(slug)` / `list_domains()`
- Config persisted to `data/domains/{slug}/config.json`
- Auto-generated system prompts, domain-specific eval logs

### rag/eval.py — Auto-Evaluation
**QueryEvalRecord** captures: query, top_k, route_mode, reranker_scores, bi_encoder_scores, latencies.

**Thresholds:**
- `RERANKER_FLOOR = -2.0` (retrieval failure)
- `RERANKER_WEAK_MEAN = -0.5` (weak retrieval)
- `LATENCY_SPIKE_S = 10.0`
- `CONTAMINATION_CATS = 3` (source diversity)

**Flags:** empty_retrieval, retrieval_failure, weak_retrieval, corpus_gap, category_contamination, latency_spike.

### rag/security.py — Auth & Rate Limiting
- Optional API key validation (AUTH_ENABLED)
- CORS configuration (CORS_ORIGINS)
- Rate limiting (RATE_LIMIT_RPM, RATE_LIMIT_BURST)
- Input sanitization (query length, top_k bounds, path traversal prevention)

### rag/monitoring.py — GPU Telemetry
- `gpu_metrics()` — GPU name, temp, utilization, VRAM used/total via pynvml
- Used by `/api/health` and Streamlit dashboard

### rag/gap_tracker.py — Knowledge Gap Analysis
- `analyze_gaps(entries, top_n)` — mines query log for recurring failures
- Exposed via `/api/gaps` endpoint

## RAG Pipeline

### Ingestion (3 phases)
1. **Parallel read + chunk** — ThreadPoolExecutor (8 workers), `iter_files()` discovers all supported files
2. **Pre-load embedding model** — BAAI/bge-m3 to GPU with FP16
3. **Batch embed + index** — batch_size=256, ChromaDB insert (500 chunks/call), dedup by MD5

### Query (5 stages)
1. **Planning** — deterministic routing (keyword-based mode detection, no LLM calls)
2. **Dense retrieval** — bi-encoder via bge-m3 (overfetch 6x)
3. **BM25 retrieval** — keyword matching
4. **Hybrid merge** — RRF fusion + source diversity cap (max 3/source)
5. **Cross-encoder reranking** — ms-marco-MiniLM, take top_k

Optional: late fusion with LLM query expansion (~200ms on Groq).

### Generation & Evaluation
- Context formatting with source references
- LLM streaming with quality-adapted system prompts
- Auto-flagging + JSONL logging (fail-silent, zero latency impact)
- Retry with escalated strategy (dense → hybrid → no-category dense)

## Configuration

```env
# Embedding
EMBED_MODEL=BAAI/bge-m3          # 568M params, 1024-dim, multilingual
EMBED_BATCH=256                  # Optimal from sweep

# Chunking
CHUNK_SIZE=600
CHUNK_OVERLAP=80

# Retrieval
TOP_K=5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
OVERFETCH_FACTOR=6               # Tested 4,6,8,10 — 6 optimal
RERANKER_BATCH_SIZE=32

# LLM (OpenAI-compatible)
LLM_PROVIDER=openai              # "openai" or "ollama"
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<key>

# Fallback LLM
FALLBACK_PROVIDER=openai
FALLBACK_MODEL=gemini-2.5-flash
FALLBACK_API_URL=https://generativelanguage.googleapis.com/v1beta/openai
FALLBACK_API_KEY=<key>

# Ollama (local)
OLLAMA_URL=http://localhost:11434

# Security
AUTH_ENABLED=false
API_KEYS=key1,key2
RATE_LIMIT_RPM=60
RATE_LIMIT_BURST=10
MAX_QUERY_LENGTH=2000
MAX_TOP_K=20
CORS_ORIGINS=http://localhost:5173

# Concurrency
WORKERS=8

# Observability
LOG_FORMAT=text                  # "json" for prod
```

Docker secrets (`/run/secrets/<NAME>`) override env vars.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| FP16 embeddings | +109% throughput, -48% VRAM, 99.3% Recall@10 parity |
| Overfetch x6 | Tested 4, 6, 8, 10 — 6 optimal; 8+ hurt NDCG |
| Source cap: 3 chunks | Prevents one doc dominating; 2 too aggressive |
| ms-marco reranker | Better NDCG on this corpus than bge-reranker-v2 |
| RRF over averaging | Rank-based, robust to score scale differences |
| Groq for LLM | 70B quality; GPU VRAM reserved for embed+reranker |
| Domain isolation | Specialized indexes prevent cross-domain contamination |
| ReACT retriever | Heuristic reasoning without LLM cost; catches entity queries |
| No fine-tuning yet | Evidence-gated roadmap — only when plateau on real queries |

## Key API

```python
# Orchestrator
plan(query, category, top_k) → RoutePlan
execute_search(query, route, category, use_expansion) → [results]
format_context(results) → str

# Store & Embedding
get_embed_model() → SentenceTransformer
embed_batch(texts) → [[float], ...]
add_chunks(col, path, chunks, knowledge_dir) → (added, skipped)

# Registries
get_index(name) → chromadb.Collection
get_embed_model(name) → SentenceTransformer
get_reranker(name) → CrossEncoder

# Domains
create_domain(name, description, ...) → DomainConfig
get_domain(slug) → DomainConfig
list_domains() → [DomainConfig]

# Eval
compute_flags(record) → [flags]
log_eval(record) → None
analyze_gaps(entries, top_n) → GapReport

# LLM
quick_complete(prompt, ...) → str
stream_chat(query, context, system_prompt, ...) → Generator[str]
```
