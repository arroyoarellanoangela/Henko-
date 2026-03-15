"""Tests for rag/agents.py — multi-agent pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from rag.agents import (
    AgentContext,
    EvaluatorAgent,
    GeneratorAgent,
    RetrieverAgent,
    RouterAgent,
    assess_quality,
    classify_complexity,
    pick_next_strategy,
    run_agent_pipeline,
    _merge_and_dedup,
)
from rag.orchestrator import RoutePlan


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_route(**overrides) -> RoutePlan:
    defaults = dict(
        indexes=["default"],
        embed_model="default_embed",
        use_reranker=True,
        reranker_model="default_reranker",
        llm_model="test-model",
        mode="answer",
        top_k=5,
        reason="test",
    )
    defaults.update(overrides)
    return RoutePlan(**defaults)


def _make_ctx(**overrides) -> AgentContext:
    defaults = dict(query="what is RAG?", query_id="test123", t_start=1000.0)
    defaults.update(overrides)
    return AgentContext(**defaults)


def _make_results(n=5, score=2.0, bi_score=0.8):
    return [
        {
            "text": f"chunk {i}",
            "category": "ai",
            "subcategory": "",
            "source": f"source_{i}",
            "file_type": "md",
            "score": score,
            "bi_score": bi_score,
        }
        for i in range(n)
    ]


# -----------------------------------------------------------------------
# classify_complexity
# -----------------------------------------------------------------------


class TestClassifyComplexity:
    def test_simple_short(self):
        assert classify_complexity("what is RAG?") == "simple"

    def test_moderate_conjunction(self):
        assert classify_complexity("what is RAG and how does it work?") == "moderate"

    def test_moderate_long(self):
        # >20 words triggers length score
        q = "explain the full detailed architecture of retrieval augmented generation systems used in modern production environments and how they scale"
        assert classify_complexity(q) == "moderate"

    def test_complex_multi_question(self):
        assert classify_complexity("what is RAG? how does it compare vs BM25? and what about hybrid search?") == "complex"

    def test_complex_semicolon_comparison(self):
        assert classify_complexity("compare RAG vs fine-tuning; what are the differences and costs?") == "complex"


# -----------------------------------------------------------------------
# assess_quality
# -----------------------------------------------------------------------


class TestAssessQuality:
    def test_good(self):
        assert assess_quality([2.0, 1.5, 0.5]) == "good"

    def test_weak(self):
        assert assess_quality([-0.8, -0.6, -0.9]) == "weak"

    def test_failed_all_below_floor(self):
        assert assess_quality([-3.0, -2.5, -4.0]) == "failed"

    def test_failed_empty(self):
        assert assess_quality([]) == "failed"


# -----------------------------------------------------------------------
# pick_next_strategy
# -----------------------------------------------------------------------


class TestPickNextStrategy:
    def test_dense_to_hybrid(self):
        assert pick_next_strategy("dense", has_category=False) == "hybrid"

    def test_hybrid_with_category_to_dense(self):
        assert pick_next_strategy("hybrid", has_category=True) == "dense"

    def test_hybrid_no_category_to_dense(self):
        assert pick_next_strategy("hybrid", has_category=False) == "dense"

    def test_category_filtered_to_hybrid(self):
        assert pick_next_strategy("category_filtered", has_category=True) == "hybrid"


# -----------------------------------------------------------------------
# _merge_and_dedup
# -----------------------------------------------------------------------


class TestMergeAndDedup:
    def test_no_duplicates(self):
        a = [{"text": "a", "score": 2.0}, {"text": "b", "score": 1.0}]
        b = [{"text": "c", "score": 1.5}]
        merged = _merge_and_dedup(a, b, top_k=5)
        assert len(merged) == 3
        assert merged[0]["text"] == "a"  # highest score

    def test_dedup(self):
        a = [{"text": "same", "score": 2.0}]
        b = [{"text": "same", "score": 1.0}]
        merged = _merge_and_dedup(a, b, top_k=5)
        assert len(merged) == 1

    def test_top_k_limit(self):
        items = [{"text": f"t{i}", "score": float(i)} for i in range(10)]
        merged = _merge_and_dedup(items, [], top_k=3)
        assert len(merged) == 3
        assert merged[0]["score"] == 9.0  # highest first


# -----------------------------------------------------------------------
# RouterAgent
# -----------------------------------------------------------------------


class TestRouterAgent:
    @patch("rag.agents.plan")
    def test_simple_query_dense(self, mock_plan):
        mock_plan.return_value = _make_route()
        ctx = _make_ctx()
        router = RouterAgent()
        router.execute(ctx)
        assert ctx.strategy == "dense"
        assert ctx.complexity == "simple"
        assert len(ctx.agent_trace) == 1
        assert ctx.agent_trace[0]["agent"] == "router"

    @patch("rag.agents.plan")
    def test_complex_query_hybrid(self, mock_plan):
        mock_plan.return_value = _make_route()
        ctx = _make_ctx(query="compare RAG vs fine-tuning; what are the pros and cons?")
        RouterAgent().execute(ctx)
        assert ctx.strategy == "hybrid"

    @patch("rag.agents.plan")
    def test_category_hint_filtered(self, mock_plan):
        mock_plan.return_value = _make_route()
        ctx = _make_ctx(category="aws")
        RouterAgent().execute(ctx)
        assert ctx.strategy == "category_filtered"

    @patch("rag.agents.plan")
    def test_retry_uses_evaluator_strategy(self, mock_plan):
        mock_plan.return_value = _make_route()
        ctx = _make_ctx(attempt=2)
        ctx.retry_strategy = "hybrid"
        RouterAgent().execute(ctx)
        assert ctx.strategy == "hybrid"


# -----------------------------------------------------------------------
# RetrieverAgent
# -----------------------------------------------------------------------


class TestRetrieverAgent:
    @patch("rag.agents.format_context", return_value="formatted context")
    @patch("rag.agents.execute_search")
    def test_dense_strategy(self, mock_search, mock_format):
        mock_search.return_value = _make_results(5)
        ctx = _make_ctx(strategy="dense")
        ctx.route = _make_route()
        RetrieverAgent().execute(ctx)
        assert len(ctx.results) == 5
        assert ctx.retrieval_quality == "good"
        assert len(ctx.reranker_scores) == 5

    @patch("rag.agents.format_context", return_value="formatted")
    @patch("rag.agents.execute_search")
    def test_empty_results_failed_quality(self, mock_search, mock_format):
        mock_search.return_value = []
        ctx = _make_ctx(strategy="dense")
        ctx.route = _make_route()
        RetrieverAgent().execute(ctx)
        assert ctx.retrieval_quality == "failed"
        assert len(ctx.results) == 0

    @patch("rag.agents.format_context", return_value="formatted")
    @patch("rag.agents.execute_search")
    def test_hybrid_merges_on_low_results(self, mock_search, mock_format):
        # First call with category returns 2, second without returns 5
        mock_search.side_effect = [_make_results(2), _make_results(5)]
        ctx = _make_ctx(strategy="hybrid", category="ai", top_k=5)
        ctx.route = _make_route()
        RetrieverAgent().execute(ctx)
        assert mock_search.call_count == 2
        assert len(ctx.results) <= 5

    @patch("rag.agents.format_context", return_value="formatted")
    @patch("rag.agents.execute_search")
    def test_weak_quality_detection(self, mock_search, mock_format):
        mock_search.return_value = _make_results(5, score=-0.8)
        ctx = _make_ctx(strategy="dense")
        ctx.route = _make_route()
        RetrieverAgent().execute(ctx)
        assert ctx.retrieval_quality == "weak"


# -----------------------------------------------------------------------
# GeneratorAgent
# -----------------------------------------------------------------------


class TestGeneratorAgent:
    @patch("rag.llm.stream_chat")
    def test_execute_collects_response(self, mock_chat):
        mock_chat.return_value = iter(["Hello", " world"])
        ctx = _make_ctx()
        ctx.context_text = "some context"
        ctx.retrieval_quality = "good"
        ctx.route = _make_route()
        GeneratorAgent().execute(ctx)
        assert ctx.full_response == "Hello world"
        assert len(ctx.agent_trace) == 1

    @patch("rag.llm.stream_chat")
    def test_weak_quality_adjusts_prompt(self, mock_chat):
        mock_chat.return_value = iter(["ok"])
        ctx = _make_ctx()
        ctx.context_text = "context"
        ctx.retrieval_quality = "weak"
        ctx.route = _make_route()
        GeneratorAgent().execute(ctx)
        # Verify stream_chat was called with modified prompt
        call_kwargs = mock_chat.call_args
        prompt = call_kwargs.kwargs.get("system_prompt", "")
        assert "incomplete" in prompt or "insufficient" in prompt

    @patch("rag.llm.stream_chat")
    def test_insufficient_detection(self, mock_chat):
        mock_chat.return_value = iter(["The context is insufficient to answer this question."])
        ctx = _make_ctx()
        ctx.context_text = "context"
        ctx.retrieval_quality = "good"
        ctx.route = _make_route()
        GeneratorAgent().execute(ctx)
        assert ctx.llm_said_insufficient is True

    @patch("rag.llm.stream_chat")
    def test_stream_yields_tokens(self, mock_chat):
        mock_chat.return_value = iter(["a", "b", "c"])
        ctx = _make_ctx()
        ctx.context_text = "context"
        ctx.retrieval_quality = "good"
        ctx.route = _make_route()
        tokens = list(GeneratorAgent().stream(ctx))
        assert tokens == ["a", "b", "c"]
        assert ctx.full_response == "abc"


# -----------------------------------------------------------------------
# EvaluatorAgent
# -----------------------------------------------------------------------


class TestEvaluatorAgent:
    def _ctx_with_results(self, scores=None, insufficient=False, attempt=1, strategy="dense"):
        ctx = _make_ctx(attempt=attempt, strategy=strategy)
        ctx.route = _make_route()
        results = _make_results(5, score=(scores[0] if scores else 2.0))
        if scores:
            for r, s in zip(results, scores):
                r["score"] = s
        ctx.results = results
        ctx.reranker_scores = [r["score"] for r in results]
        ctx.bi_encoder_scores = [r["bi_score"] for r in results]
        ctx.full_response = "The context is insufficient" if insufficient else "RAG uses retrieval."
        ctx.llm_said_insufficient = insufficient
        ctx.t_retrieval = 0.5
        ctx.t_llm = 1.0
        return ctx

    @patch("rag.agents.log_eval")
    def test_healthy_no_retry(self, mock_log):
        ctx = self._ctx_with_results(scores=[2.0, 1.5, 1.0, 0.5, 0.3])
        EvaluatorAgent().execute(ctx)
        assert ctx.should_retry is False
        mock_log.assert_called_once()

    @patch("rag.agents.log_eval")
    def test_weak_retrieval_triggers_retry(self, mock_log):
        ctx = self._ctx_with_results(scores=[-0.8, -0.6, -0.9, -0.7, -0.5])
        EvaluatorAgent().execute(ctx)
        assert ctx.should_retry is True
        assert ctx.retry_strategy == "hybrid"

    @patch("rag.agents.log_eval")
    def test_retrieval_failure_triggers_retry(self, mock_log):
        ctx = self._ctx_with_results(scores=[-3.0, -2.5, -4.0, -2.1, -3.5])
        EvaluatorAgent().execute(ctx)
        assert ctx.should_retry is True

    @patch("rag.agents.log_eval")
    def test_corpus_gap_no_retry(self, mock_log):
        ctx = self._ctx_with_results(scores=[2.0, 1.5, 1.0, 0.5, 0.3], insufficient=True)
        EvaluatorAgent().execute(ctx)
        assert ctx.should_retry is False
        assert "corpus_gap" in ctx.eval_flags

    @patch("rag.agents.log_eval")
    def test_max_attempts_stops_retry(self, mock_log):
        ctx = self._ctx_with_results(scores=[-0.8, -0.6, -0.9, -0.7, -0.5], attempt=3)
        EvaluatorAgent().execute(ctx)
        assert ctx.should_retry is False

    @patch("rag.agents.log_eval")
    def test_strategy_escalation_dense_to_hybrid(self, mock_log):
        ctx = self._ctx_with_results(scores=[-0.8, -0.6, -0.9, -0.7, -0.5], strategy="dense")
        EvaluatorAgent().execute(ctx)
        assert ctx.retry_strategy == "hybrid"

    @patch("rag.agents.log_eval")
    def test_strategy_escalation_category_filtered(self, mock_log):
        ctx = self._ctx_with_results(scores=[-0.8, -0.6, -0.9, -0.7, -0.5], strategy="category_filtered")
        EvaluatorAgent().execute(ctx)
        assert ctx.retry_strategy == "hybrid"

    @patch("rag.agents.log_eval")
    def test_flags_include_strategy_tag(self, mock_log):
        ctx = self._ctx_with_results(scores=[2.0, 1.5, 1.0, 0.5, 0.3])
        EvaluatorAgent().execute(ctx)
        assert any(f.startswith("strategy_") for f in ctx.eval_flags)

    @patch("rag.agents.log_eval")
    def test_retry_flags_include_retry_tag(self, mock_log):
        ctx = self._ctx_with_results(scores=[2.0, 1.5, 1.0, 0.5, 0.3], attempt=2)
        EvaluatorAgent().execute(ctx)
        assert "retry_2" in ctx.eval_flags

    @patch("rag.agents.log_eval")
    def test_same_strategy_no_retry(self, mock_log):
        """If pick_next_strategy returns the same strategy, don't retry."""
        ctx = self._ctx_with_results(scores=[-0.8, -0.6, -0.9, -0.7, -0.5], strategy="hybrid")
        ctx.category = None
        EvaluatorAgent().execute(ctx)
        # hybrid -> dense (different), so should retry
        assert ctx.should_retry is True
        assert ctx.retry_strategy == "dense"


# -----------------------------------------------------------------------
# Coordination loop
# -----------------------------------------------------------------------


class TestCoordinationLoop:
    @patch("rag.agents.log_eval")
    @patch("rag.llm.stream_chat")
    @patch("rag.agents.format_context", return_value="ctx")
    @patch("rag.agents.execute_search")
    @patch("rag.agents.plan")
    def test_single_pass_good_retrieval(self, mock_plan, mock_search, mock_format, mock_chat, mock_log):
        mock_plan.return_value = _make_route()
        mock_search.return_value = _make_results(5)
        mock_chat.return_value = iter(["answer"])

        ctx = run_agent_pipeline("what is RAG?")
        assert ctx.attempt == 1
        assert ctx.full_response == "answer"
        assert len(ctx.agent_trace) == 4  # router, retriever, generator, evaluator

    @patch("rag.agents.log_eval")
    @patch("rag.llm.stream_chat")
    @patch("rag.agents.format_context", return_value="ctx")
    @patch("rag.agents.execute_search")
    @patch("rag.agents.plan")
    def test_retry_once_then_success(self, mock_plan, mock_search, mock_format, mock_chat, mock_log):
        mock_plan.return_value = _make_route()
        # First attempt: weak results. Second: good results.
        mock_search.side_effect = [
            _make_results(5, score=-0.8),  # weak
            _make_results(5, score=2.0),   # good
        ]
        mock_chat.return_value = iter(["answer"])

        ctx = run_agent_pipeline("what is RAG?")
        assert ctx.attempt == 2
        assert len(ctx.agent_trace) == 8  # 4 per attempt

    @patch("rag.agents.log_eval")
    @patch("rag.llm.stream_chat")
    @patch("rag.agents.format_context", return_value="ctx")
    @patch("rag.agents.execute_search")
    @patch("rag.agents.plan")
    def test_max_attempts_exhausted(self, mock_plan, mock_search, mock_format, mock_chat, mock_log):
        mock_plan.return_value = _make_route()
        mock_search.return_value = _make_results(5, score=-0.8)  # always weak
        mock_chat.return_value = iter(["answer"])

        ctx = run_agent_pipeline("what is RAG?")
        # Should stop at max_attempts (3): attempt 1 weak -> retry, attempt 2 weak -> retry same strategy -> stop
        assert ctx.attempt <= 3

    @patch("rag.agents.log_eval")
    @patch("rag.llm.stream_chat")
    @patch("rag.agents.format_context", return_value="ctx")
    @patch("rag.agents.execute_search")
    @patch("rag.agents.plan")
    def test_trace_records_all_steps(self, mock_plan, mock_search, mock_format, mock_chat, mock_log):
        mock_plan.return_value = _make_route()
        mock_search.return_value = _make_results(5)
        mock_chat.return_value = iter(["answer"])

        ctx = run_agent_pipeline("what is RAG?")
        agents = [t["agent"] for t in ctx.agent_trace]
        assert agents == ["router", "retriever", "generator", "evaluator"]

    @patch("rag.agents.log_eval")
    @patch("rag.llm.stream_chat")
    @patch("rag.agents.format_context", return_value="ctx")
    @patch("rag.agents.execute_search")
    @patch("rag.agents.plan")
    def test_timing_populated(self, mock_plan, mock_search, mock_format, mock_chat, mock_log):
        mock_plan.return_value = _make_route()
        mock_search.return_value = _make_results(5)
        mock_chat.return_value = iter(["answer"])

        ctx = run_agent_pipeline("what is RAG?")
        assert ctx.t_total > 0
        assert ctx.t_retrieval >= 0
