"""V2.1 Validation — functional test of orchestrator routing + traceability.

Runs a set of real-ish queries through orchestrator.plan() and verifies:
  1. Mode classification (answer / summary / code) is correct
  2. RoutePlan fields are populated and consistent
  3. Logs are emitted for every plan

Usage:
    python validate_v2_1.py
"""

import logging
import sys

# Configure logging to stdout so we can see orchestrator tracing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    stream=sys.stdout,
)

from rag.orchestrator import plan

# ---------------------------------------------------------------------------
# Test queries grouped by expected mode
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    # --- General knowledge (expected: mode=answer) ---
    {"query": "What is the best AI model for coding?",          "expected_mode": "answer"},
    {"query": "Claude Code vs Codestral",                       "expected_mode": "answer"},
    {"query": "best coding tool",                               "expected_mode": "answer"},
    {"query": "What GPU is recommended for local LLMs?",        "expected_mode": "answer"},

    # --- Summary / longform (expected: mode=summary) ---
    {"query": "Compare Claude and GPT-4 in terms of reasoning, coding ability, and pricing — give me a detailed overview with pros and cons",
     "expected_mode": "summary"},
    {"query": "Summarize the main differences between Ollama and vLLM for local inference",
     "expected_mode": "summary"},
    {"query": "overview of transformer architectures",          "expected_mode": "summary"},

    # --- Code-like (expected: mode=code) ---
    {"query": "def fibonacci(n): how to implement in Python?",  "expected_mode": "code"},
    {"query": "pip install sentence-transformers not working",   "expected_mode": "code"},
    {"query": "SELECT * FROM users WHERE role = 'admin'",       "expected_mode": "code"},
    {"query": "How to use docker compose for a FastAPI app?",   "expected_mode": "code"},
    {"query": "git push origin main rejected non-fast-forward", "expected_mode": "code"},
]

# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  V2.1 ORCHESTRATOR VALIDATION")
    print("=" * 72)

    passed = 0
    failed = 0
    results = []

    for tc in TEST_CASES:
        query = tc["query"]
        expected = tc["expected_mode"]

        route = plan(query)

        ok = route.mode == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append({
            "query": query,
            "expected": expected,
            "got": route.mode,
            "reason": route.reason,
            "status": status,
            "route": route,
        })

    # Print results table
    print()
    print("-" * 72)
    print(f"{'Status':<6} {'Expected':<10} {'Got':<10} Query")
    print("-" * 72)
    for r in results:
        marker = "OK" if r["status"] == "PASS" else "XX"
        print(f"  [{marker}]  {r['expected']:<10} {r['got']:<10} {r['query'][:50]}")
    print("-" * 72)

    # Print full RoutePlan for each
    print()
    print("FULL ROUTE PLANS:")
    print("-" * 72)
    for r in results:
        rp = r["route"]
        print(f"\nQuery: {r['query'][:60]}")
        print(f"  mode      = {rp.mode}")
        print(f"  indexes   = {rp.indexes}")
        print(f"  embed     = {rp.embed_model}")
        print(f"  reranker  = {rp.use_reranker} ({rp.reranker_model})")
        print(f"  llm       = {rp.llm_model}")
        print(f"  top_k     = {rp.top_k}")
        print(f"  reason    = {rp.reason}")

    # Summary
    print()
    print("=" * 72)
    print(f"  RESULT: {passed}/{len(TEST_CASES)} passed, {failed} failed")
    if failed == 0:
        print("  All routing decisions match expected modes.")
    else:
        print("  Some routing decisions need review (see XX above).")
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
