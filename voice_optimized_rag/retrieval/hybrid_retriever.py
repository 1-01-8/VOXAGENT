"""混合检索器 —— 稠密检索 + 倒排召回 + RRF 融合 + 启发式 rerank。"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.retrieval.inverted_index import InvertedIndex, tokenize_text
from voice_optimized_rag.retrieval.vector_store import SearchResult
from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector, Timer

logger = get_logger("hybrid_retriever")


RETRIEVAL_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "产品": ("product", "products", "features", "crm", "module"),
    "商品": ("product", "products", "catalog", "features", "module"),
    "目录": ("catalog", "products", "modules", "product line", "features"),
    "模块": ("module", "modules", "feature", "features", "product"),
    "型号": ("product", "model", "sku", "module"),
    "编号": ("product", "sku", "id", "module"),
    "价格": ("pricing", "price", "prices", "plans", "billing", "cost"),
    "多少钱": ("pricing", "price", "cost", "plans"),
    "报价": ("quote", "pricing", "price"),
    "套餐": ("plan", "plans", "tier", "tiers", "pricing"),
    "费用": ("pricing", "billing", "cost", "fee"),
    "收费": ("pricing", "billing", "cost"),
    "优惠": ("discount", "promotion", "offer"),
    "活动": ("promotion", "campaign", "offer"),
    "促销": ("promotion", "discount", "offer"),
    "库存": ("inventory", "stock", "availability"),
    "现货": ("inventory", "stock", "available"),
    "退款": ("refund", "refund policy", "refund process"),
    "退货": ("refund", "return", "return policy"),
    "发票": ("invoice", "billing", "receipt"),
    "对账": ("billing", "reconciliation", "invoice"),
    "订单": ("order", "purchase", "shipment"),
    "物流": ("shipping", "logistics", "delivery"),
    "集成": ("integration", "integrations", "api"),
    "接口": ("api", "endpoint", "integration"),
}


def expand_query_text(query_text: str) -> str:
    """Expand business Chinese queries with English retrieval hints.

    The knowledge base is currently English-heavy while the CLI and web users ask in
    Chinese. Appending stable English aliases improves both dense and sparse recall
    without depending on another LLM call.
    """
    normalized = query_text.lower()
    expansions: list[str] = []
    for keyword, aliases in RETRIEVAL_QUERY_ALIASES.items():
        if keyword in normalized:
            for alias in aliases:
                if alias not in expansions:
                    expansions.append(alias)

    if not expansions:
        return query_text

    return f"{query_text} {' '.join(expansions)}"


class DenseDocumentStore(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def document_version(self) -> int: ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[SearchResult]: ...

    def list_documents(self) -> list[SearchResult]: ...

    def get_embedding(self, index: int) -> np.ndarray | None: ...


class HybridRetriever:
    """统一检索入口。

    特性：
    - 双路召回：dense 向量检索 + sparse 倒排 BM25
    - 融合：RRF
    - rerank：基于 lexical coverage 和融合分的启发式精排
    """

    def __init__(
        self,
        config: VORConfig,
        vector_store: DenseDocumentStore,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._metrics = metrics
        self._inverted_index = InvertedIndex(
            k1=config.sparse_bm25_k1,
            b=config.sparse_bm25_b,
        )
        self._synced_document_version = -1

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        dense_top_k = max(top_k, self._config.hybrid_dense_top_k)

        if self._config.retrieval_mode == "dense":
            return self._dense_search(query_embedding, dense_top_k, include_embeddings)[:top_k]

        with self._timer("search"):
            dense_results = self._dense_search(query_embedding, dense_top_k, include_embeddings)
            sparse_results = self._sparse_search(query_text, max(top_k, self._config.hybrid_sparse_top_k), include_embeddings)

            if not sparse_results:
                return dense_results[:top_k]
            if not dense_results:
                ranked_sparse = self._rerank(query_text, sparse_results)
                return ranked_sparse[:top_k]

            fused = self._fuse_results(
                dense_results=dense_results,
                sparse_results=sparse_results,
                candidate_pool=max(top_k, self._config.hybrid_candidate_pool),
            )
            ranked = self._rerank(query_text, fused) if self._config.rerank_enabled else fused
            return ranked[:top_k]

    def _dense_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        with self._timer("dense_search"):
            results = self._vector_store.search(
                query_embedding,
                top_k=top_k,
                include_embeddings=include_embeddings,
            )

        for result in results:
            if result.dense_score is None:
                result.dense_score = result.score
            if result.retrieval_source != "hybrid":
                result.retrieval_source = "dense"
        if self._metrics:
            self._metrics.increment("hybrid_dense_hits", len(results))
        return results

    def _sparse_search(
        self,
        query_text: str,
        top_k: int,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        self._ensure_sparse_index_synced()
        with self._timer("sparse_search"):
            results = self._inverted_index.search(query_text, top_k=top_k)

        if include_embeddings:
            for result in results:
                if result.embedding is None:
                    result.embedding = self._safe_get_embedding(result.index)

        if self._metrics:
            self._metrics.increment("hybrid_sparse_hits", len(results))
        return results

    def _ensure_sparse_index_synced(self) -> None:
        current_version = self._vector_store.document_version
        if current_version == self._synced_document_version:
            return

        documents = self._vector_store.list_documents()
        self._inverted_index.rebuild(documents)
        self._synced_document_version = current_version
        logger.debug(f"Sparse index synced: {self._inverted_index.size} docs")

    def _fuse_results(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        candidate_pool: int,
    ) -> list[SearchResult]:
        fused_by_key: dict[str, SearchResult] = {}
        rrf_k = self._config.hybrid_rrf_k

        for rank, result in enumerate(dense_results, start=1):
            key = self._result_key(result)
            fused = fused_by_key.setdefault(key, self._clone_result(result))
            fused.dense_score = result.dense_score if result.dense_score is not None else result.score
            fused.score += 1.0 / (rrf_k + rank)
            fused.retrieval_source = "dense"

        for rank, result in enumerate(sparse_results, start=1):
            key = self._result_key(result)
            fused = fused_by_key.setdefault(key, self._clone_result(result))
            fused.sparse_score = result.sparse_score if result.sparse_score is not None else result.score
            if fused.embedding is None and result.embedding is not None:
                fused.embedding = result.embedding
            fused.score += 1.0 / (rrf_k + rank)
            fused.retrieval_source = (
                "hybrid" if fused.dense_score is not None and fused.sparse_score is not None else "sparse"
            )

        fused_results = sorted(fused_by_key.values(), key=lambda item: item.score, reverse=True)
        return fused_results[:candidate_pool]

    def _rerank(self, query_text: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return candidates

        query_tokens = set(tokenize_text(query_text))
        fused_scores = [candidate.score for candidate in candidates]
        dense_scores = [candidate.dense_score or 0.0 for candidate in candidates]
        sparse_scores = [candidate.sparse_score or 0.0 for candidate in candidates]

        fused_min, fused_max = min(fused_scores), max(fused_scores)
        dense_min, dense_max = min(dense_scores), max(dense_scores)
        sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
        normalized_query = "".join(query_text.lower().split())

        with self._timer("rerank"):
            for candidate in candidates:
                candidate_tokens = set(tokenize_text(candidate.text))
                token_coverage = (
                    len(query_tokens & candidate_tokens) / len(query_tokens)
                    if query_tokens else 0.0
                )
                normalized_text = "".join(candidate.text.lower().split())
                phrase_boost = 1.0 if normalized_query and normalized_query in normalized_text else 0.0

                fused_norm = self._normalize(candidate.score, fused_min, fused_max)
                dense_norm = self._normalize(candidate.dense_score or 0.0, dense_min, dense_max)
                sparse_norm = self._normalize(candidate.sparse_score or 0.0, sparse_min, sparse_max)

                rerank_score = (
                    0.30 * fused_norm
                    + 0.15 * dense_norm
                    + 0.25 * sparse_norm
                    + 0.20 * token_coverage
                    + 0.10 * phrase_boost
                )
                candidate.rerank_score = rerank_score
                candidate.score = rerank_score

        if self._metrics:
            self._metrics.increment("hybrid_rerank_runs")
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _result_key(self, result: SearchResult) -> str:
        source = result.metadata.get("source", "")
        chunk = result.metadata.get("chunk_index", "")
        if source or chunk != "":
            return f"{source}::{chunk}"
        return f"{result.index}::{hash(result.text)}"

    def _clone_result(self, result: SearchResult) -> SearchResult:
        return SearchResult(
            text=result.text,
            metadata=dict(result.metadata),
            score=0.0,
            index=result.index,
            embedding=result.embedding,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            rerank_score=result.rerank_score,
            retrieval_source=result.retrieval_source,
        )

    def _safe_get_embedding(self, index: int) -> np.ndarray | None:
        try:
            return self._vector_store.get_embedding(index)
        except Exception:
            return None

    def _timer(self, operation: str):
        if self._metrics is None:
            return _NullTimer()
        return Timer(self._metrics, "hybrid_retriever", operation)

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value > 0 else 0.0
        return (value - low) / (high - low)


class _NullTimer:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None