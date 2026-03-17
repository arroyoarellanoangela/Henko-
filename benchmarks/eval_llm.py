"""LLM Quality Evaluation — sends queries through the full RAG pipeline.

Tests:
  - Category separation (models vs tools)
  - Marketing language avoidance
  - Source citation quality
  - Latency per query

Requires: API running on localhost:8000

Usage:
    python eval_llm.py
"""

import json
import os
import sys
import time

# Fix Windows cp1252 console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

API = "http://localhost:8000/api"

# ---------------------------------------------------------------------------
# Eval queries — designed to expose common failure modes
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    # Model/tool confusion
    {"query": "best AI model for coding", "checks": ["should list MODELS not tools"]},
    {"query": "Claude Code vs Codestral", "checks": ["Claude Code = tool, Codestral = model, should separate them"]},
    {"query": "best coding tool", "checks": ["should list TOOLS not models"]},

    # Marketing language
    {"query": "what is Claude?", "checks": ["no marketing language (best-in-class, game-changing, etc.)"]},
    {"query": "compare GPT-4 and Gemini", "checks": ["factual comparison, benchmarks, not hype"]},

    # Synthesis / longform
    {"query": "summarize this architecture", "checks": ["coherent summary, not a list dump"]},
    {"query": "what is RAG and how does it work?", "checks": ["clear explanation, not copied text"]},

    # Code-adjacent
    {"query": "how to use Ollama with Python", "checks": ["practical code guidance"]},

    # Edge cases
    {"query": "what is the meaning of life?", "checks": ["should say context insufficient, not hallucinate"]},
    {"query": "best AI models 2025", "checks": ["time-aware answer if corpus supports it, else admit gap"]},
]

# ---------------------------------------------------------------------------
# Marketing words to flag
# ---------------------------------------------------------------------------

MARKETING_WORDS = [
    "best-in-class", "enterprise-grade", "industry-leading", "cutting-edge",
    "game-changing", "revolutionary", "groundbreaking", "unparalleled",
    "state-of-the-art", "next-generation", "world-class",
]


def collect_response(query: str, top_k: int = 5) -> dict:
    """Send a query to the API and collect the full streamed response."""
    start = time.time()
    resp = requests.post(
        f"{API}/query",
        json={"query": query, "top_k": top_k},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    tokens = []
    sources = []
    route = None
    ttfb = None  # time to first byte

    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data: "):
            continue
        data = json.loads(text[6:])

        if data["type"] == "sources":
            sources = data.get("sources", [])
            route = data.get("route")
        elif data["type"] == "token":
            if ttfb is None:
                ttfb = time.time() - start
            tokens.append(data["content"])

    elapsed = time.time() - start
    answer = "".join(tokens)

    return {
        "answer": answer,
        "sources": sources,
        "route": route,
        "latency_s": round(elapsed, 2),
        "ttfb_s": round(ttfb, 2) if ttfb else None,
        "token_count": len(answer.split()),
    }


def check_marketing(answer: str) -> list[str]:
    """Return any marketing words found in the answer."""
    lower = answer.lower()
    return [w for w in MARKETING_WORDS if w in lower]


def main():
    # Check API is up
    try:
        r = requests.get(f"{API}/status", timeout=5)
        r.raise_for_status()
        status = r.json()
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API} — {e}")
        print("Start the backend first: python api.py")
        return 1

    print("=" * 72)
    print("  LLM QUALITY EVALUATION")
    print(f"  Model:    {status.get('llm_model', '?')}")
    print(f"  Provider: {status.get('provider_label', status.get('llm_provider', '?'))}")
    print(f"  Chunks:   {status.get('chunks', '?')}")
    print("=" * 72)
    print()

    results = []
    for i, tc in enumerate(EVAL_QUERIES, 1):
        query = tc["query"]
        print(f"[{i}/{len(EVAL_QUERIES)}] {query}")

        try:
            resp = collect_response(query)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": query, "error": str(e)})
            continue

        marketing_found = check_marketing(resp["answer"])

        results.append({
            "query": query,
            "checks": tc["checks"],
            "answer": resp["answer"],
            "sources_count": len(resp["sources"]),
            "route": resp["route"],
            "latency_s": resp["latency_s"],
            "ttfb_s": resp["ttfb_s"],
            "token_count": resp["token_count"],
            "marketing_words": marketing_found,
        })

        mode_str = resp["route"]["mode"] if resp["route"] else "?"
        print(f"  mode={mode_str}  sources={len(resp['sources'])}  "
              f"latency={resp['latency_s']}s  ttfb={resp['ttfb_s']}s  "
              f"tokens~{resp['token_count']}")
        if marketing_found:
            print(f"  WARNING: marketing words found: {marketing_found}")
        print()

    # Summary
    print("=" * 72)
    print("  SUMMARY")
    print("-" * 72)

    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    ttfbs = [r["ttfb_s"] for r in results if r.get("ttfb_s") is not None]
    marketing_total = sum(len(r.get("marketing_words", [])) for r in results)

    if latencies:
        print(f"  Avg latency:     {sum(latencies)/len(latencies):.2f}s")
        print(f"  Avg TTFB:        {sum(ttfbs)/len(ttfbs):.2f}s" if ttfbs else "  Avg TTFB:        N/A")
        print(f"  Min/Max latency: {min(latencies):.2f}s / {max(latencies):.2f}s")
    print(f"  Marketing words: {marketing_total} total across {len(results)} queries")
    print("=" * 72)

    # Write detailed results for review
    print()
    print("DETAILED RESPONSES:")
    print("=" * 72)
    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"   Checks: {r.get('checks', [])}")
        if "error" in r:
            print(f"   ERROR: {r['error']}")
        else:
            print(f"   Route: {r.get('route')}")
            print(f"   Sources: {r['sources_count']}")
            print(f"   Latency: {r['latency_s']}s  TTFB: {r['ttfb_s']}s")
            if r["marketing_words"]:
                print(f"   MARKETING: {r['marketing_words']}")
            print(f"   Answer ({r['token_count']} words):")
            # Print first 500 chars of answer
            ans = r["answer"]
            if len(ans) > 500:
                print(f"   {ans[:500]}...")
            else:
                print(f"   {ans}")
        print("-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
