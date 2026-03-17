# LLM Evaluation — Groq / llama-3.3-70b-versatile

> Evaluation date: 2026-03-11
> Previous LLM: Ollama / qwen3:14b (replaced due to poor generation quality)
> Current LLM: Groq cloud / llama-3.3-70b-versatile
> Corpus: 29,541 chunks in ChromaDB

---

## Why the switch

qwen3:14b (local Ollama) had persistent issues:
- Mixed models and tools in the same list
- Marketing language in answers ("best-in-class", "cutting-edge")
- Poor category separation
- Inconsistent source citation

llama-3.3-70b via Groq was evaluated as replacement.

---

## Evaluation setup

- **Pipeline**: full RAG (bi-encoder retrieval + cross-encoder reranking + LLM generation)
- **Queries**: 10 real-world test queries covering answer, summary, and edge cases
- **Checks**: category separation, marketing language, hallucination guard, latency
- **Script**: `eval_llm.py`

---

## Results

### Performance

| Metric | Value |
|---|---|
| Avg latency | 3.17s |
| Avg TTFB | 2.77s |
| Min latency | 2.78s |
| Max latency | 3.79s |

### Quality

| Check | Result |
|---|---|
| Marketing words detected | **0** across 10 queries |
| Model/tool separation | Correct — uses `## Models`, `## Tools`, `## Libraries` headers |
| Out-of-context rejection | Correct — "meaning of life" answered with "context insufficient" |
| Source citation | Present in answers |
| Time-awareness gaps | Correctly admits when corpus lacks data |

### Per-query breakdown

| Query | Mode | Sources | Latency | Tokens | Marketing | Notes |
|---|---|---|---|---|---|---|
| best AI model for coding | answer | 5 | 3.10s | 84 | 0 | Lists models only, not tools |
| Claude Code vs Codestral | answer | 5 | 3.11s | 87 | 0 | Admits missing context for Codestral |
| best coding tool | answer | 5 | 2.92s | 55 | 0 | Lists tools only, not models |
| what is Claude? | answer | 5 | 3.12s | 106 | 0 | Separates model/tool/library correctly |
| compare GPT-4 and Gemini | summary | 5 | 3.50s | 134 | 0 | Factual, no hype |
| summarize this architecture | summary | 5 | 3.79s | 176 | 0 | Coherent summary |
| what is RAG and how does it work? | answer | 5 | 3.00s | 70 | 0 | Clear explanation |
| how to use Ollama with Python | answer | 5 | 3.27s | 108 | 0 | Includes code example |
| what is the meaning of life? | answer | 5 | 2.94s | 49 | 0 | Correctly rejects — context insufficient |
| best AI models 2025 | answer | 5 | 2.94s | 52 | 0 | Admits corpus gap |

---

## Decision

**llama-3.3-70b-versatile via Groq is validated as the production LLM.**

Trade-offs:
- (+) 70B model — significantly better generation quality than local 14B
- (+) Zero marketing language with current SYSTEM_PROMPT
- (+) Correct category separation (models vs tools vs libraries)
- (+) Fast inference via Groq cloud (avg 3.17s including retrieval)
- (-) Requires internet + API key (Groq)
- (-) Not local — depends on external service availability
- (-) Rate limits apply (Groq free tier)

The quality improvement over qwen3:14b justifies the cloud dependency for now.
