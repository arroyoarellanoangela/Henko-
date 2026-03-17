"""bench.py — Quantitative benchmark harness for Suyven RAG.

Usage:
    python bench.py                                        # retrieval-only
    python bench.py --with-generation                      # + LLM generation
    python bench.py --label "top10_exp" --top-k 10         # override config
    python bench.py --compare report_a.json report_b.json  # A/B delta table
    python bench.py --inspect-query "what is RAG"          # helper to populate GT
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean as _mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.bench_metrics import (
    binary_relevance,
    contamination_check,
    faithfulness_embedding,
    keyword_coverage,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.bench_types import (
    BenchmarkReport,
    GenerationResult,
    GroundTruthEntry,
    QueryMetrics,
    RetrievalResult,
)

GT_PATH = Path(__file__).parent / "data" / "eval" / "ground_truth.jsonl"
REPORT_DIR = Path(__file__).parent / "data" / "eval"


# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> list[GroundTruthEntry]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            entries.append(GroundTruthEntry(**d))
    print(f"  Loaded {len(entries)} ground truth entries from {path.name}")
    return entries


# ---------------------------------------------------------------------------
# Run retrieval (no API server needed)
# ---------------------------------------------------------------------------


def run_retrieval(
    entries: list[GroundTruthEntry],
    top_k_override: int | None = None,
    use_agents: bool = False,
    use_react: bool = False,
    use_expansion: bool = False,
) -> list[RetrievalResult]:
    if use_agents or use_react:
        return _run_retrieval_agents(entries, top_k_override, use_react=use_react)
    return _run_retrieval_direct(entries, top_k_override, use_expansion=use_expansion)


def _run_retrieval_direct(
    entries: list[GroundTruthEntry],
    top_k_override: int | None = None,
    use_expansion: bool = False,
) -> list[RetrievalResult]:
    from rag.orchestrator import execute_search, plan

    results = []
    for i, gt in enumerate(entries, 1):
        tk = top_k_override or gt.top_k
        t0 = time.time()
        route = plan(gt.query, category=gt.category, top_k=tk)
        candidates = execute_search(gt.query, route, category=gt.category, use_expansion=use_expansion)
        latency = time.time() - t0

        results.append(RetrievalResult(
            query_id=gt.query_id,
            retrieved_sources=[c["source"] for c in candidates],
            retrieved_texts=[c["text"] for c in candidates],
            reranker_scores=[c["score"] for c in candidates],
            bi_encoder_scores=[c.get("bi_score", 0.0) for c in candidates],
            actual_mode=route.mode,
            latency_retrieval_s=round(latency, 4),
        ))
        status = "ok" if candidates else "EMPTY"
        print(f"  [{i}/{len(entries)}] {gt.query_id}: {status} ({len(candidates)} results, {latency:.2f}s)")

    return results


def _run_retrieval_agents(
    entries: list[GroundTruthEntry],
    top_k_override: int | None = None,
    use_react: bool = False,
) -> list[RetrievalResult]:
    from rag.agents import run_agent_pipeline

    results = []
    for i, gt in enumerate(entries, 1):
        tk = top_k_override or gt.top_k
        ctx = run_agent_pipeline(
            gt.query, category=gt.category, top_k=tk,
            skip_generation=True, use_react=use_react,
        )

        results.append(RetrievalResult(
            query_id=gt.query_id,
            retrieved_sources=[c["source"] for c in ctx.results],
            retrieved_texts=[c["text"] for c in ctx.results],
            reranker_scores=ctx.reranker_scores,
            bi_encoder_scores=ctx.bi_encoder_scores,
            actual_mode=ctx.route.mode if ctx.route else "answer",
            latency_retrieval_s=round(ctx.t_retrieval, 4),
        ))
        retries = ctx.attempt - 1
        retry_tag = f" (retried {retries}x)" if retries else ""
        status = "ok" if ctx.results else "EMPTY"
        print(f"  [{i}/{len(entries)}] {gt.query_id}: {status} ({len(ctx.results)} results, {ctx.t_total:.2f}s){retry_tag}")

    return results


# ---------------------------------------------------------------------------
# Run generation (optional)
# ---------------------------------------------------------------------------


def run_generation(
    entries: list[GroundTruthEntry],
    retrieval: list[RetrievalResult],
) -> list[GenerationResult]:
    from rag.llm import stream_chat
    from rag.orchestrator import format_context

    results = []
    for i, (gt, ret) in enumerate(zip(entries, retrieval), 1):
        # Rebuild candidates dict for format_context
        candidates = []
        for src, txt, score in zip(ret.retrieved_sources, ret.retrieved_texts, ret.reranker_scores):
            candidates.append({"category": "", "subcategory": "", "source": src, "text": txt, "score": score})

        context = format_context(candidates)
        t0 = time.time()
        tokens = []
        for token in stream_chat(gt.query, context):
            tokens.append(token)
        latency = time.time() - t0
        answer = "".join(tokens)

        results.append(GenerationResult(
            query_id=gt.query_id,
            answer_text=answer,
            context_text=context,
            latency_llm_s=round(latency, 4),
        ))
        print(f"  [{i}/{len(entries)}] {gt.query_id}: {len(answer)} chars, {latency:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(
    entries: list[GroundTruthEntry],
    retrieval: list[RetrievalResult],
    generation: list[GenerationResult] | None = None,
    embed_fn=None,
) -> list[QueryMetrics]:
    metrics = []
    gen_map = {}
    if generation:
        gen_map = {g.query_id: g for g in generation}

    for gt, ret in zip(entries, retrieval):
        k = gt.top_k
        rel = binary_relevance(ret.retrieved_sources, gt.relevant_sources)
        total_relevant = len(gt.relevant_sources)

        faith = None
        kw_cov = None
        gen = gen_map.get(gt.query_id)
        if gen:
            kw_cov = keyword_coverage(gen.answer_text, gt.expected_answer_keywords)
            if embed_fn and gen.answer_text.strip() and gen.context_text.strip():
                faith = faithfulness_embedding(gen.answer_text, gen.context_text, embed_fn)

        metrics.append(QueryMetrics(
            query_id=gt.query_id,
            difficulty=gt.difficulty,
            ndcg_at_k=ndcg_at_k(rel, k),
            mrr_at_k=mrr_at_k(rel, k),
            recall_at_k=recall_at_k(rel, k, total_relevant),
            precision_at_k=precision_at_k(rel, k),
            route_correct=(ret.actual_mode == gt.expected_mode),
            faithfulness_score=faith,
            keyword_coverage=kw_cov,
            contamination=contamination_check(ret.retrieved_sources, gt.irrelevant_sources),
        ))

    return metrics


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------


def _percentile(vals: list[float], p: int) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(len(vals) * p / 100)))
    return vals[k]


def build_report(
    label: str,
    metrics: list[QueryMetrics],
    retrieval: list[RetrievalResult],
    generation: list[GenerationResult] | None = None,
) -> BenchmarkReport:
    n = len(metrics)

    # Aggregate IR (exclude out_of_corpus queries — they have no relevant sources)
    ir_metrics = [m for m in metrics if m.difficulty != "out_of_corpus"]
    mean_ndcg = _mean([m.ndcg_at_k for m in ir_metrics]) if ir_metrics else 0.0
    mean_mrr = _mean([m.mrr_at_k for m in ir_metrics]) if ir_metrics else 0.0
    mean_recall = _mean([m.recall_at_k for m in ir_metrics]) if ir_metrics else 0.0
    mean_precision = _mean([m.precision_at_k for m in ir_metrics]) if ir_metrics else 0.0

    # Generation
    faith_vals = [m.faithfulness_score for m in metrics if m.faithfulness_score is not None]
    kw_vals = [m.keyword_coverage for m in metrics if m.keyword_coverage is not None]
    mean_faith = _mean(faith_vals) if faith_vals else None
    mean_kw = _mean(kw_vals) if kw_vals else None

    # Routing
    route_acc = sum(1 for m in metrics if m.route_correct) / n if n else 0.0

    # By difficulty
    by_diff: dict[str, dict[str, float]] = {}
    for diff in ("easy", "medium", "hard", "out_of_corpus"):
        subset = [m for m in metrics if m.difficulty == diff]
        if subset:
            by_diff[diff] = {
                "count": len(subset),
                "mean_ndcg": _mean([m.ndcg_at_k for m in subset]),
                "mean_mrr": _mean([m.mrr_at_k for m in subset]),
                "mean_recall": _mean([m.recall_at_k for m in subset]),
                "mean_precision": _mean([m.precision_at_k for m in subset]),
                "route_accuracy": sum(1 for m in subset if m.route_correct) / len(subset),
            }

    # Latency
    ret_lats = [r.latency_retrieval_s for r in retrieval]
    llm_lats = [g.latency_llm_s for g in generation] if generation else []

    return BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_label=label,
        num_queries=n,
        mean_ndcg=round(mean_ndcg, 4),
        mean_mrr=round(mean_mrr, 4),
        mean_recall=round(mean_recall, 4),
        mean_precision=round(mean_precision, 4),
        mean_faithfulness=round(mean_faith, 4) if mean_faith is not None else None,
        mean_keyword_coverage=round(mean_kw, 4) if mean_kw is not None else None,
        route_accuracy=round(route_acc, 4),
        metrics_by_difficulty=by_diff,
        mean_retrieval_latency_s=round(_mean(ret_lats), 4) if ret_lats else 0.0,
        mean_llm_latency_s=round(_mean(llm_lats), 4) if llm_lats else None,
        p95_retrieval_latency_s=round(_percentile(ret_lats, 95), 4) if ret_lats else 0.0,
        per_query=metrics,
    )


# ---------------------------------------------------------------------------
# Save / print report
# ---------------------------------------------------------------------------


def save_report(report: BenchmarkReport) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"bench_{report.config_label}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=True)
    return path


def print_report(report: BenchmarkReport) -> None:
    print(f"\n{'=' * 62}")
    print(f"  Suyven Benchmark Report: {report.config_label}")
    print(f"{'=' * 62}")
    print(f"  Queries:    {report.num_queries}")
    print(f"  Timestamp:  {report.timestamp}")
    print()

    print("  IR Metrics (aggregate):")
    print(f"    NDCG@k:      {report.mean_ndcg:.4f}")
    print(f"    MRR@k:       {report.mean_mrr:.4f}")
    print(f"    Recall@k:    {report.mean_recall:.4f}")
    print(f"    Precision@k: {report.mean_precision:.4f}")
    print()

    if report.mean_faithfulness is not None:
        print("  Generation Metrics:")
        print(f"    Faithfulness:     {report.mean_faithfulness:.4f}")
        print(f"    Keyword coverage: {report.mean_keyword_coverage:.4f}")
        print()

    print(f"  Route accuracy: {report.route_accuracy:.1%}")
    print()

    # Contamination
    contam = sum(1 for m in report.per_query if m.contamination if isinstance(m, QueryMetrics)) + \
             sum(1 for m in report.per_query if isinstance(m, dict) and m.get("contamination"))
    # Handle both dict and dataclass
    contam_count = 0
    for m in report.per_query:
        if isinstance(m, QueryMetrics):
            contam_count += 1 if m.contamination else 0
        elif isinstance(m, dict):
            contam_count += 1 if m.get("contamination") else 0
    print(f"  Contamination: {contam_count}/{report.num_queries} queries")
    print()

    print("  Latency:")
    print(f"    Retrieval mean: {report.mean_retrieval_latency_s:.2f}s")
    print(f"    Retrieval P95:  {report.p95_retrieval_latency_s:.2f}s")
    if report.mean_llm_latency_s is not None:
        print(f"    LLM mean:       {report.mean_llm_latency_s:.2f}s")
    print()

    # By difficulty
    if report.metrics_by_difficulty:
        print("  By difficulty:")
        print(f"    {'Diff':<12s} {'N':>3s} {'NDCG':>7s} {'MRR':>7s} {'Recall':>7s} {'Prec':>7s} {'Route':>7s}")
        print(f"    {'-'*12} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for diff in ("easy", "medium", "hard", "out_of_corpus"):
            d = report.metrics_by_difficulty.get(diff)
            if d:
                print(f"    {diff:<12s} {int(d['count']):>3d} {d['mean_ndcg']:>7.4f} {d['mean_mrr']:>7.4f} "
                      f"{d['mean_recall']:>7.4f} {d['mean_precision']:>7.4f} {d['route_accuracy']:>6.0%}")
        print()

    # Per-query details for failures
    failures = []
    for m in report.per_query:
        if isinstance(m, QueryMetrics):
            if m.ndcg_at_k == 0.0 or not m.route_correct or m.contamination:
                failures.append(m)
        elif isinstance(m, dict):
            if m.get("ndcg_at_k") == 0.0 or not m.get("route_correct") or m.get("contamination"):
                failures.append(m)

    if failures:
        print(f"  {'=' * 58}")
        print(f"  Queries needing attention ({len(failures)}):")
        print(f"  {'=' * 58}")
        for m in failures:
            qid = m.query_id if isinstance(m, QueryMetrics) else m.get("query_id", "?")
            ndcg = m.ndcg_at_k if isinstance(m, QueryMetrics) else m.get("ndcg_at_k", 0)
            route_ok = m.route_correct if isinstance(m, QueryMetrics) else m.get("route_correct", False)
            contam = m.contamination if isinstance(m, QueryMetrics) else m.get("contamination", False)
            issues = []
            if ndcg == 0.0:
                issues.append("NDCG=0")
            if not route_ok:
                issues.append("wrong_route")
            if contam:
                issues.append("contaminated")
            print(f"    {qid}: {', '.join(issues)}")
        print()


# ---------------------------------------------------------------------------
# Compare two reports
# ---------------------------------------------------------------------------


def compare_reports(path_a: Path, path_b: Path) -> None:
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)

    print(f"\n{'=' * 62}")
    print(f"  A/B Comparison")
    print(f"{'=' * 62}")
    print(f"  A: {a['config_label']}  ({a['num_queries']} queries)")
    print(f"  B: {b['config_label']}  ({b['num_queries']} queries)")
    print()

    metrics = [
        ("NDCG@k", "mean_ndcg"),
        ("MRR@k", "mean_mrr"),
        ("Recall@k", "mean_recall"),
        ("Precision@k", "mean_precision"),
        ("Route accuracy", "route_accuracy"),
        ("Retrieval latency", "mean_retrieval_latency_s"),
        ("Retrieval P95", "p95_retrieval_latency_s"),
    ]

    if a.get("mean_faithfulness") is not None and b.get("mean_faithfulness") is not None:
        metrics.append(("Faithfulness", "mean_faithfulness"))
        metrics.append(("Keyword cov.", "mean_keyword_coverage"))

    if a.get("mean_llm_latency_s") is not None and b.get("mean_llm_latency_s") is not None:
        metrics.append(("LLM latency", "mean_llm_latency_s"))

    print(f"  {'Metric':<22s} {'A':>8s} {'B':>8s} {'Delta':>8s} {'Better':>7s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    # For latency metrics, lower is better
    latency_keys = {"mean_retrieval_latency_s", "p95_retrieval_latency_s", "mean_llm_latency_s"}

    for name, key in metrics:
        va = a.get(key, 0) or 0
        vb = b.get(key, 0) or 0
        delta = vb - va
        if key in latency_keys:
            better = "B" if delta < 0 else ("A" if delta > 0 else "-")
        else:
            better = "B" if delta > 0 else ("A" if delta < 0 else "-")
        sign = "+" if delta > 0 else ""
        print(f"  {name:<22s} {va:>8.4f} {vb:>8.4f} {sign}{delta:>7.4f} {better:>7s}")

    print()


# ---------------------------------------------------------------------------
# Inspect query (helper to populate ground truth)
# ---------------------------------------------------------------------------


def inspect_query(query: str, top_k: int = 10) -> None:
    from rag.orchestrator import execute_search, plan

    print(f"\n  Inspecting: \"{query}\"")
    print(f"  top_k: {top_k}")
    print()

    route = plan(query, top_k=top_k)
    print(f"  Route: mode={route.mode}, reason={route.reason}")
    print()

    results = execute_search(query, route)
    if not results:
        print("  No results.")
        return

    print(f"  {'Rank':<5s} {'Score':>8s} {'BiScore':>8s} {'Category':<12s} {'Source'}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*12} {'-'*30}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<5d} {r['score']:>8.4f} {r.get('bi_score', 0):>8.4f} {r['category']:<12s} {r['source']}")

    print()
    print("  Unique sources:", sorted(set(r["source"] for r in results)))
    print("  Unique categories:", sorted(set(r["category"] for r in results)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Suyven RAG benchmark")
    parser.add_argument("--label", default="baseline", help="Config label for this run")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth JSONL")
    parser.add_argument("--with-generation", action="store_true", help="Also run LLM generation")
    parser.add_argument("--agents", action="store_true", help="Use multi-agent pipeline instead of direct orchestrator")
    parser.add_argument("--react", action="store_true", help="Use ReACT multi-tool retriever (implies --agents)")
    parser.add_argument("--expand", action="store_true", help="Use LLM query expansion for broader recall")
    parser.add_argument("--compare", nargs=2, metavar="FILE", help="Compare two report JSON files")
    parser.add_argument("--inspect-query", type=str, default=None, help="Inspect a single query")

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_reports(Path(args.compare[0]), Path(args.compare[1]))
        return

    # Inspect mode
    if args.inspect_query:
        inspect_query(args.inspect_query, top_k=args.top_k or 10)
        return

    # Full benchmark
    gt_path = Path(args.gt) if args.gt else GT_PATH
    print(f"\n{'=' * 62}")
    print(f"  Suyven Benchmark: {args.label}")
    print(f"{'=' * 62}")

    print("\n[1/4] Loading ground truth...")
    entries = load_ground_truth(gt_path)

    use_agents = args.agents or args.react
    mode_label = "react" if args.react else ("agents" if args.agents else "direct")
    if args.expand and not use_agents:
        mode_label += "+expand"
    print(f"\n[2/4] Running retrieval ({mode_label})...")
    retrieval = run_retrieval(
        entries, top_k_override=args.top_k,
        use_agents=use_agents, use_react=args.react,
        use_expansion=args.expand and not use_agents,
    )

    generation = None
    embed_fn = None
    if args.with_generation:
        print("\n[3/4] Running generation...")
        generation = run_generation(entries, retrieval)

        # Load embed model for faithfulness
        print("\n  Loading embed model for faithfulness scoring...")
        from rag.model_registry import get_embed_model
        model = get_embed_model("default_embed")
        embed_fn = lambda texts: model.encode(texts, convert_to_numpy=True).tolist()
    else:
        print("\n[3/4] Skipping generation (use --with-generation)")

    print("\n[4/4] Computing metrics...")
    metrics = compute_all_metrics(entries, retrieval, generation, embed_fn)

    report = build_report(args.label, metrics, retrieval, generation)

    path = save_report(report)
    print(f"\n  Report saved: {path.name}")

    print_report(report)


if __name__ == "__main__":
    main()
