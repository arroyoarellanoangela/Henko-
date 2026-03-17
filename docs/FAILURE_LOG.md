# Failure Log — Real Usage Observations

> Fill this as you use Kaizen with real queries.
> Each entry helps decide if V2.2 is needed and what the real fix should be.

## How to log

For each failure, note:
- **Query**: what you asked
- **Expected**: what a good answer would look like
- **Got**: what actually happened
- **Type**: retrieval | routing | generation | corpus
- **Severity**: minor | moderate | critical

---

## Retrieval failures
Chunks returned are irrelevant or missing the right source.

<!-- Example:
### 2026-03-12
- Query: "how does vLLM compare to TGI for inference"
- Expected: side-by-side comparison from inference docs
- Got: returned chunks about Ollama instead
- Type: retrieval
- Severity: moderate
- Notes: corpus may lack vLLM-specific content
-->

## Routing failures
Orchestrator picks wrong mode (answer/summary/code).

## Generation failures
LLM gives bad answer despite good retrieved chunks.

## Corpus gaps
Query is valid but knowledge base simply doesn't have the content.

---

## Signals to watch for V2.2 trigger

- [ ] PDFs longform consistently returning low-quality chunks
- [ ] Cross-domain contamination (AI chunks returned for data-eng queries) > 20%
- [ ] NDCG@5 < 0.75 on an identifiable subset
- [ ] New corpus with different domain semantics needs ingestion
