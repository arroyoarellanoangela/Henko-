"""Benchmark data structures — ground truth, results, metrics, reports."""

from dataclasses import dataclass, field


@dataclass
class GroundTruthEntry:
    query_id: str
    query: str
    category: str | None
    top_k: int
    expected_mode: str  # "answer" | "summary" | "code"
    relevant_sources: list[str]
    relevant_keywords: list[str]
    expected_answer_keywords: list[str]
    irrelevant_sources: list[str]
    difficulty: str  # "easy" | "medium" | "hard"


@dataclass
class RetrievalResult:
    query_id: str
    retrieved_sources: list[str]
    retrieved_texts: list[str]
    reranker_scores: list[float]
    bi_encoder_scores: list[float]
    actual_mode: str
    latency_retrieval_s: float


@dataclass
class GenerationResult:
    query_id: str
    answer_text: str
    context_text: str
    latency_llm_s: float


@dataclass
class QueryMetrics:
    query_id: str
    difficulty: str
    # IR metrics
    ndcg_at_k: float
    mrr_at_k: float
    recall_at_k: float
    precision_at_k: float
    # Routing
    route_correct: bool
    # Generation (None if retrieval-only run)
    faithfulness_score: float | None
    keyword_coverage: float | None
    # Contamination
    contamination: bool


@dataclass
class BenchmarkReport:
    timestamp: str
    config_label: str
    num_queries: int
    # Aggregate IR
    mean_ndcg: float
    mean_mrr: float
    mean_recall: float
    mean_precision: float
    # Aggregate generation
    mean_faithfulness: float | None
    mean_keyword_coverage: float | None
    # Routing
    route_accuracy: float
    # Breakdown by difficulty
    metrics_by_difficulty: dict[str, dict[str, float]]
    # Latency
    mean_retrieval_latency_s: float
    mean_llm_latency_s: float | None
    p95_retrieval_latency_s: float
    # Per-query detail
    per_query: list[QueryMetrics] = field(default_factory=list)
