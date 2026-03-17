"""eval_report.py — Read data/eval/query_log.jsonl and print a summary.

Usage:
    python eval_report.py
    python eval_report.py --last 50      # only last N queries
    python eval_report.py --flags-only   # only show flagged queries
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean as _mean, median as _median

LOG_FILE = Path(__file__).parent / "data" / "eval" / "query_log.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path, last_n: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if last_n:
        rows = rows[-last_n:]
    return rows


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "0%"


def _percentile(vals: list[float], p: int) -> float:
    """Simple percentile (nearest-rank)."""
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(len(vals) * p / 100)))
    return vals[k]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(rows: list[dict], flags_only: bool = False) -> None:
    if not rows:
        print("No eval records found.")
        print(f"  Expected log at: {LOG_FILE}")
        return

    total = len(rows)
    dates = [r["timestamp"][:10] for r in rows]
    print(f"\n{'='*60}")
    print(f"  Suyven Auto-Eval Report")
    print(f"{'='*60}")
    print(f"  Queries:    {total}")
    print(f"  Date range: {min(dates)} .. {max(dates)}")
    print()

    # --- Flag distribution ---
    flag_counter: Counter = Counter()
    flagged_rows = []
    for r in rows:
        flags = r.get("flags", [])
        if flags:
            flagged_rows.append(r)
            for f in flags:
                flag_counter[f] += 1

    flagged_total = len(flagged_rows)
    print(f"  Flagged queries: {flagged_total} / {total} ({_pct(flagged_total, total)})")
    print()

    if flag_counter:
        print("  Flag distribution:")
        for flag, count in flag_counter.most_common():
            print(f"    {flag:<30s} {count:>4d}  ({_pct(count, total)})")
        print()

    # --- Latency stats ---
    latencies = [r["latency_total_s"] for r in rows if r.get("latency_total_s") is not None]
    retrieval_latencies = [r["latency_retrieval_s"] for r in rows if r.get("latency_retrieval_s") is not None]
    llm_latencies = [r["latency_llm_s"] for r in rows if r.get("latency_llm_s") is not None]

    if latencies:
        print("  Latency (total):")
        print(f"    Mean:  {_mean(latencies):.2f}s")
        print(f"    P50:   {_median(latencies):.2f}s")
        print(f"    P95:   {_percentile(latencies, 95):.2f}s")
        print(f"    Max:   {max(latencies):.2f}s")
        print()

    if retrieval_latencies:
        print("  Latency (retrieval):")
        print(f"    Mean:  {_mean(retrieval_latencies):.2f}s")
        print(f"    P50:   {_median(retrieval_latencies):.2f}s")
        print()

    if llm_latencies:
        print("  Latency (LLM):")
        print(f"    Mean:  {_mean(llm_latencies):.2f}s")
        print(f"    P50:   {_median(llm_latencies):.2f}s")
        print()

    # --- Route distribution ---
    route_counter: Counter = Counter(r.get("route_mode", "unknown") for r in rows)
    print("  Route distribution:")
    for mode, count in route_counter.most_common():
        print(f"    {mode:<15s} {count:>4d}  ({_pct(count, total)})")
    print()

    # --- V2.2 trigger dashboard ---
    print(f"  {'='*56}")
    print(f"  V2.2 Trigger Dashboard")
    print(f"  {'='*56}")

    corpus_gap_count = flag_counter.get("corpus_gap", 0)
    contamination_count = flag_counter.get("category_contamination", 0)
    retrieval_fail_count = flag_counter.get("retrieval_failure", 0)
    weak_count = flag_counter.get("weak_retrieval", 0)

    print(f"    Corpus gap rate:        {corpus_gap_count}/{total} ({_pct(corpus_gap_count, total)})")
    print(f"    Contamination rate:     {contamination_count}/{total} ({_pct(contamination_count, total)})")
    print(f"    Retrieval failure rate: {retrieval_fail_count}/{total} ({_pct(retrieval_fail_count, total)})")
    print(f"    Weak retrieval rate:    {weak_count}/{total} ({_pct(weak_count, total)})")
    print()

    # --- Flagged query details ---
    if flagged_rows:
        print(f"  {'='*56}")
        print(f"  Flagged Queries")
        print(f"  {'='*56}")
        for r in flagged_rows:
            flags = r.get("flags", [])
            query = r.get("query", "")[:80]
            mode = r.get("route_mode", "?")
            latency = r.get("latency_total_s", 0)
            n_results = r.get("num_results", 0)
            mean_score = r.get("mean_reranker_score")
            score_str = f"{mean_score:.2f}" if mean_score is not None else "n/a"
            print(f"    [{', '.join(flags)}]")
            print(f"      query:   {query}")
            print(f"      mode:    {mode}  |  results: {n_results}  |  mean_score: {score_str}  |  latency: {latency:.2f}s")
            print()
    elif not flags_only:
        print("  No flagged queries -- all healthy!")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suyven auto-eval report")
    parser.add_argument("--last", type=int, default=None, help="Only show last N queries")
    parser.add_argument("--flags-only", action="store_true", help="Only show flagged queries")
    args = parser.parse_args()

    rows = _load(LOG_FILE, last_n=args.last)
    report(rows, flags_only=args.flags_only)
