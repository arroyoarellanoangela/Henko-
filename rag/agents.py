"""Multi-agent RAG pipeline — 4 coordinating agents with retry loop.

Agents:
  RouterAgent    — classifies query complexity, picks retrieval strategy
  RetrieverAgent — executes search with strategy, reports quality
  GeneratorAgent — generates response, adapts prompt to retrieval quality
  EvaluatorAgent — flags issues, decides retry, logs eval record

Coordination:
  Router -> Retriever -> Generator -> Evaluator
  If Evaluator says should_retry and attempt < max_attempts: loop back to Router
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from statistics import mean as _mean
from typing import Generator

from .config import SYSTEM_PROMPT, TOP_K
from .eval import (
    QueryEvalRecord,
    compute_flags,
    detect_insufficient,
    log_eval,
    new_query_id,
    RERANKER_FLOOR,
    RERANKER_WEAK_MEAN,
)
from .orchestrator import execute_search, format_context, plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentContext — shared state between agents
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    # Input
    query: str
    category: str | None = None
    top_k: int = TOP_K
    query_id: str = ""

    # Router output
    route: object | None = None  # RoutePlan
    strategy: str = "dense"  # "dense" | "hybrid" | "category_filtered"
    complexity: str = "simple"  # "simple" | "moderate" | "complex"

    # Retriever output
    results: list[dict] = field(default_factory=list)
    context_text: str = ""
    retrieval_quality: str = "unknown"  # "good" | "weak" | "failed"
    reranker_scores: list[float] = field(default_factory=list)
    bi_encoder_scores: list[float] = field(default_factory=list)

    # Generator output
    response_tokens: list[str] = field(default_factory=list)
    full_response: str = ""
    llm_said_insufficient: bool = False

    # Evaluator output
    eval_flags: list[str] = field(default_factory=list)
    should_retry: bool = False
    retry_reason: str = ""
    retry_strategy: str = ""

    # Coordination state
    attempt: int = 1
    max_attempts: int = 3
    agent_trace: list[dict] = field(default_factory=list)

    # Timing
    t_start: float = 0.0
    t_retrieval: float = 0.0
    t_llm: float = 0.0
    t_total: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def classify_complexity(query: str) -> str:
    """Heuristic query complexity classification. No LLM call."""
    words = query.split()
    n_words = len(words)
    n_questions = query.count("?")
    has_conjunction = bool(re.search(r"\b(and|or|but|also|ademas|tambien|y|o)\b", query, re.IGNORECASE))
    has_semicolon = ";" in query
    has_comparison = bool(re.search(r"\b(compare|vs|versus|diferencia|difference)\b", query, re.IGNORECASE))

    score = 0
    if n_words > 20:
        score += 1
    if n_words > 40:
        score += 1
    if n_questions > 1:
        score += 1
    if has_conjunction:
        score += 1
    if has_semicolon:
        score += 1
    if has_comparison:
        score += 1

    if score >= 3:
        return "complex"
    elif score >= 1:
        return "moderate"
    return "simple"


def assess_quality(reranker_scores: list[float]) -> str:
    """Assess retrieval quality from reranker scores."""
    if not reranker_scores:
        return "failed"
    if all(s < RERANKER_FLOOR for s in reranker_scores):
        return "failed"
    if _mean(reranker_scores) < RERANKER_WEAK_MEAN:
        return "weak"
    return "good"


def pick_next_strategy(current: str, has_category: bool) -> str:
    """Escalation: dense -> hybrid -> no-category dense."""
    if current == "dense":
        return "hybrid"
    if current == "hybrid" and has_category:
        return "dense"  # will clear category
    if current == "category_filtered":
        return "hybrid"
    return "dense"


def _merge_and_dedup(primary: list[dict], secondary: list[dict], top_k: int) -> list[dict]:
    """Merge two result lists, deduplicate by text hash, sort by score."""
    seen = set()
    merged = []
    for r in primary + secondary:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            merged.append(r)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class RouterAgent:
    name = "router"
    role = "router"

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        ctx.complexity = classify_complexity(ctx.query)

        # On retry, use Evaluator's recommendation
        if ctx.attempt > 1 and ctx.retry_strategy:
            ctx.strategy = ctx.retry_strategy
            # If evaluator recommended clearing category
            if ctx.strategy == "dense" and ctx.attempt > 1 and ctx.retry_reason and "no_category" in ctx.retry_reason:
                ctx.category = None
        elif ctx.category:
            ctx.strategy = "category_filtered"
        elif ctx.complexity == "complex":
            ctx.strategy = "hybrid"
        else:
            ctx.strategy = "dense"

        # Build route using existing orchestrator
        tk = ctx.top_k * 2 if ctx.strategy == "hybrid" else ctx.top_k
        ctx.route = plan(ctx.query, category=ctx.category, top_k=tk)

        ctx.agent_trace.append({
            "agent": self.name,
            "action": "route",
            "attempt": ctx.attempt,
            "strategy": ctx.strategy,
            "complexity": ctx.complexity,
            "mode": ctx.route.mode,
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })

        logger.info(
            "[%s] attempt=%d strategy=%s complexity=%s mode=%s",
            self.name, ctx.attempt, ctx.strategy, ctx.complexity, ctx.route.mode,
        )
        return ctx


# ---------------------------------------------------------------------------
# RetrieverAgent
# ---------------------------------------------------------------------------


class RetrieverAgent:
    name = "retriever"
    role = "retriever"

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        results = execute_search(ctx.query, ctx.route, category=ctx.category)

        # Hybrid: if category-filtered returned few results, try without
        if ctx.strategy == "hybrid" and ctx.category and len(results) < ctx.top_k:
            more = execute_search(ctx.query, ctx.route, category=None)
            results = _merge_and_dedup(results, more, ctx.top_k)
        elif ctx.strategy == "hybrid" and len(results) > ctx.top_k:
            results = results[:ctx.top_k]

        ctx.results = results
        ctx.context_text = format_context(results)
        ctx.reranker_scores = [r["score"] for r in results]
        ctx.bi_encoder_scores = [r.get("bi_score", 0.0) for r in results]
        ctx.retrieval_quality = assess_quality(ctx.reranker_scores)
        ctx.t_retrieval = time.time() - ctx.t_start

        ctx.agent_trace.append({
            "agent": self.name,
            "action": "retrieve",
            "attempt": ctx.attempt,
            "strategy": ctx.strategy,
            "num_results": len(results),
            "quality": ctx.retrieval_quality,
            "mean_score": round(_mean(ctx.reranker_scores), 4) if ctx.reranker_scores else None,
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })

        logger.info(
            "[%s] attempt=%d results=%d quality=%s",
            self.name, ctx.attempt, len(results), ctx.retrieval_quality,
        )
        return ctx


# ---------------------------------------------------------------------------
# GeneratorAgent
# ---------------------------------------------------------------------------


_WEAK_CAVEAT = (
    "\n\nIMPORTANT: The retrieved context may be incomplete or only partially relevant. "
    "If the context does not contain enough information to answer accurately, "
    "explicitly state that the context is insufficient."
)

_FAILED_PROMPT = (
    "You are a technical knowledge assistant. No relevant context was found for this query. "
    "Respond honestly that you could not find relevant information in the knowledge base."
)


class GeneratorAgent:
    name = "generator"
    role = "generator"

    def _build_prompt(self, quality: str) -> str:
        if quality == "failed":
            return _FAILED_PROMPT
        if quality == "weak":
            return SYSTEM_PROMPT + _WEAK_CAVEAT
        return SYSTEM_PROMPT

    def execute(self, ctx: AgentContext) -> AgentContext:
        """Batch mode: collect full response. Used by bench.py."""
        from .llm import stream_chat

        t0 = time.time()
        prompt = self._build_prompt(ctx.retrieval_quality)
        tokens = list(stream_chat(ctx.query, ctx.context_text, system_prompt=prompt))
        ctx.response_tokens = tokens
        ctx.full_response = "".join(tokens)
        ctx.llm_said_insufficient = detect_insufficient(ctx.full_response)
        ctx.t_llm = time.time() - t0

        ctx.agent_trace.append({
            "agent": self.name,
            "action": "generate",
            "attempt": ctx.attempt,
            "quality_prompt": ctx.retrieval_quality,
            "response_length": len(ctx.full_response),
            "insufficient": ctx.llm_said_insufficient,
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })

        logger.info(
            "[%s] attempt=%d chars=%d insufficient=%s",
            self.name, ctx.attempt, len(ctx.full_response), ctx.llm_said_insufficient,
        )
        return ctx

    def stream(self, ctx: AgentContext) -> Generator[str, None, None]:
        """SSE mode: yield tokens one by one. Fills ctx when done."""
        from .llm import stream_chat

        t0 = time.time()
        prompt = self._build_prompt(ctx.retrieval_quality)
        tokens = []
        for token in stream_chat(ctx.query, ctx.context_text, system_prompt=prompt):
            tokens.append(token)
            yield token
        ctx.response_tokens = tokens
        ctx.full_response = "".join(tokens)
        ctx.llm_said_insufficient = detect_insufficient(ctx.full_response)
        ctx.t_llm = time.time() - t0

        ctx.agent_trace.append({
            "agent": self.name,
            "action": "generate",
            "attempt": ctx.attempt,
            "quality_prompt": ctx.retrieval_quality,
            "response_length": len(ctx.full_response),
            "insufficient": ctx.llm_said_insufficient,
            "duration_ms": round(ctx.t_llm * 1000, 1),
        })


# ---------------------------------------------------------------------------
# EvaluatorAgent
# ---------------------------------------------------------------------------


_RETRYABLE_FLAGS = {"empty_retrieval", "retrieval_failure", "weak_retrieval"}


class EvaluatorAgent:
    name = "evaluator"
    role = "evaluator"

    def _build_eval_record(self, ctx: AgentContext) -> QueryEvalRecord:
        from datetime import datetime, timezone

        scores = ctx.reranker_scores
        return QueryEvalRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_id=ctx.query_id,
            query=ctx.query,
            category_filter=ctx.category,
            top_k=ctx.top_k,
            route_mode=ctx.route.mode if ctx.route else "unknown",
            route_reason=ctx.route.reason if ctx.route else "",
            route_indexes=ctx.route.indexes if ctx.route else [],
            num_results=len(ctx.results),
            reranker_scores=scores,
            bi_encoder_scores=ctx.bi_encoder_scores,
            max_reranker_score=max(scores) if scores else None,
            min_reranker_score=min(scores) if scores else None,
            mean_reranker_score=_mean(scores) if scores else None,
            source_categories=list({r.get("category", "") for r in ctx.results}),
            llm_said_insufficient=ctx.llm_said_insufficient,
            response_length=len(ctx.full_response),
            token_count_approx=len(ctx.full_response) // 4,
            latency_total_s=round(time.time() - ctx.t_start, 4),
            latency_retrieval_s=round(ctx.t_retrieval, 4),
            latency_llm_s=round(ctx.t_llm, 4),
        )

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        record = self._build_eval_record(ctx)
        flags = compute_flags(record)

        # Tag with attempt and strategy info
        if ctx.attempt > 1:
            flags.append(f"retry_{ctx.attempt}")
        flags.append(f"strategy_{ctx.strategy}")

        record.flags = flags
        ctx.eval_flags = flags

        # Retry decision
        retryable = _RETRYABLE_FLAGS & set(flags)
        if retryable and ctx.attempt < ctx.max_attempts:
            next_strategy = pick_next_strategy(ctx.strategy, ctx.category is not None)
            # Don't retry with the same strategy
            if next_strategy != ctx.strategy:
                ctx.should_retry = True
                ctx.retry_strategy = next_strategy
                ctx.retry_reason = f"flags={sorted(retryable)}, switch to {next_strategy}"
                if next_strategy == "dense" and ctx.category:
                    ctx.retry_reason += " (no_category)"
            else:
                ctx.should_retry = False
        else:
            ctx.should_retry = False

        # Always log
        log_eval(record)

        ctx.agent_trace.append({
            "agent": self.name,
            "action": "evaluate",
            "attempt": ctx.attempt,
            "flags": flags,
            "should_retry": ctx.should_retry,
            "retry_reason": ctx.retry_reason if ctx.should_retry else "",
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })

        logger.info(
            "[%s] attempt=%d flags=%s retry=%s",
            self.name, ctx.attempt, flags, ctx.should_retry,
        )
        return ctx


# ---------------------------------------------------------------------------
# Coordination loops
# ---------------------------------------------------------------------------


def run_agent_pipeline(
    query: str,
    category: str | None = None,
    top_k: int = TOP_K,
    query_id: str | None = None,
    skip_generation: bool = False,
) -> AgentContext:
    """Run the full multi-agent pipeline (batch mode). Returns completed AgentContext.

    Args:
        skip_generation: If True, skip GeneratorAgent (retrieval-only mode for benchmarks).
    """
    ctx = AgentContext(
        query=query,
        category=category,
        top_k=top_k,
        query_id=query_id or new_query_id(),
        t_start=time.time(),
    )

    router = RouterAgent()
    retriever = RetrieverAgent()
    generator = GeneratorAgent()
    evaluator = EvaluatorAgent()

    while ctx.attempt <= ctx.max_attempts:
        router.execute(ctx)
        retriever.execute(ctx)
        if not skip_generation:
            generator.execute(ctx)
        evaluator.execute(ctx)

        if not ctx.should_retry:
            break

        # Prepare for retry — keep trace, reset outputs
        ctx.attempt += 1
        ctx.results = []
        ctx.context_text = ""
        ctx.response_tokens = []
        ctx.full_response = ""
        ctx.retrieval_quality = "unknown"
        ctx.should_retry = False

    ctx.t_total = time.time() - ctx.t_start
    return ctx


def prepare_agent_context(
    query: str,
    category: str | None = None,
    top_k: int = TOP_K,
    query_id: str | None = None,
) -> tuple[AgentContext, RouterAgent, RetrieverAgent, GeneratorAgent, EvaluatorAgent]:
    """Create context and agents for SSE streaming (split flow).

    Usage in api.py:
        ctx, router, retriever, generator, evaluator = prepare_agent_context(...)
        router.execute(ctx)
        retriever.execute(ctx)
        # pre-flight retry if needed
        if ctx.retrieval_quality in ("weak", "failed") and ctx.attempt < ctx.max_attempts:
            evaluator.execute(ctx)
            if ctx.should_retry:
                ctx.attempt += 1; ctx.results = []; ...
                router.execute(ctx)
                retriever.execute(ctx)
        # then stream: for token in generator.stream(ctx): yield token
        # then: evaluator.execute(ctx)
    """
    ctx = AgentContext(
        query=query,
        category=category,
        top_k=top_k,
        query_id=query_id or new_query_id(),
        t_start=time.time(),
    )
    return ctx, RouterAgent(), RetrieverAgent(), GeneratorAgent(), EvaluatorAgent()
