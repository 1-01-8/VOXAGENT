"""Tests for retrieval/hybrid_retriever.py — 混合检索模块"""

from __future__ import annotations

import numpy as np
import pytest

from voice_optimized_rag.retrieval.hybrid_retriever import HybridRetriever, expand_query_text
from voice_optimized_rag.retrieval.kb_manager import KBManager
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (matrix / norms).astype(np.float32)


class TestHybridRetriever:
    def test_expand_query_text_adds_english_business_aliases(self):
        expanded = expand_query_text("你们的价格和套餐呢")

        assert expanded.startswith("你们的价格和套餐呢")
        assert "pricing" in expanded
        assert "plans" in expanded

    def test_expand_query_text_adds_catalog_aliases(self):
        expanded = expand_query_text("商品目录")

        assert expanded.startswith("商品目录")
        assert "catalog" in expanded
        assert "products" in expanded

    def test_hybrid_recall_and_rerank_promote_exact_lexical_match(self, config, dim: int):
        store = FAISSVectorStore(dimension=dim)
        retriever = HybridRetriever(config=config, vector_store=store)

        texts = [
            "退款流程需要订单号和退款原因。",
            "物流查询请提供订单号和快递单号。",
        ]
        embeddings = np.zeros((2, dim), dtype=np.float32)
        embeddings[0, 1] = 1.0
        embeddings[1, 0] = 1.0
        store.add_documents(texts, _normalize_rows(embeddings))

        query_embedding = np.zeros(dim, dtype=np.float32)
        query_embedding[0] = 1.0  # 故意让 dense 更偏向物流文档

        results = retriever.search(
            query_text="退款流程",
            query_embedding=query_embedding,
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].text == "退款流程需要订单号和退款原因。"
        assert results[0].sparse_score is not None
        assert results[0].rerank_score is not None
        assert results[0].retrieval_source in {"sparse", "hybrid"}

    @pytest.mark.asyncio
    async def test_sparse_index_resyncs_after_kb_delete(self, config, mock_embeddings, dim: int):
        store = FAISSVectorStore(dimension=dim)
        kb = KBManager(store, mock_embeddings)
        retriever = HybridRetriever(config=config, vector_store=store)

        await kb.add_documents(["退款流程说明"], source="refund.md")
        await kb.add_documents(["物流规则说明"], source="shipping.md")

        query_embedding = await mock_embeddings.embed_single("退款流程")
        before = retriever.search("退款流程", query_embedding, top_k=2)
        assert any(result.text == "退款流程说明" for result in before)

        removed = kb.delete_by_source("refund.md")
        assert removed == 1

        after = retriever.search("退款流程", query_embedding, top_k=2)
        assert all(result.text != "退款流程说明" for result in after)