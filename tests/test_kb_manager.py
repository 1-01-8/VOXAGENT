"""Tests for retrieval/kb_manager.py — 知识库热更新模块

系统组件: Retrieval — KBManager 知识库热更新
源文件:   voice_optimized_rag/retrieval/kb_manager.py
职责:     增量/删除文档，触发 FAISS 索引重建，管理知识库元数据

测试覆盖：
- 增量添加文档并写入向量数据库
- 按来源删除文档（FAISS 重建索引）
- 按类别删除文档
- 空输入处理
- 统计信息
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.retrieval.kb_manager import KBManager
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from tests.conftest import MockEmbedding


@pytest.fixture
def kb(dim: int, mock_embeddings: MockEmbedding) -> KBManager:
    store = FAISSVectorStore(dimension=dim)
    return KBManager(store, mock_embeddings)


class TestKBManager:
    """知识库管理器测试"""

    @pytest.mark.asyncio
    async def test_add_documents(self, kb: KBManager):
        """应成功添加文档块"""
        count = await kb.add_documents(
            texts=["产品A说明", "产品B说明", "售后规则"],
            category="product",
            source="products.md",
        )
        assert count == 3
        stats = kb.get_stats()
        assert stats["total_chunks"] == 3

    @pytest.mark.asyncio
    async def test_add_empty(self, kb: KBManager):
        """空列表应返回 0"""
        count = await kb.add_documents(texts=[])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_by_source(self, kb: KBManager):
        """按来源删除后总数应减少"""
        await kb.add_documents(texts=["A1", "A2"], source="a.md")
        await kb.add_documents(texts=["B1"], source="b.md")
        assert kb.get_stats()["total_chunks"] == 3

        removed = kb.delete_by_source("a.md")
        assert removed == 2
        assert kb.get_stats()["total_chunks"] == 1

    @pytest.mark.asyncio
    async def test_delete_by_category(self, kb: KBManager):
        """按类别删除"""
        await kb.add_documents(texts=["FAQ1", "FAQ2"], category="faq")
        await kb.add_documents(texts=["Policy1"], category="policy")

        removed = kb.delete_by_category("faq")
        assert removed == 2
        stats = kb.get_stats()
        assert stats["total_chunks"] == 1
        assert "faq" not in stats["categories"]

    @pytest.mark.asyncio
    async def test_stats_categories(self, kb: KBManager):
        """统计信息应按类别分组"""
        await kb.add_documents(texts=["P1", "P2"], category="product")
        await kb.add_documents(texts=["F1"], category="faq")

        stats = kb.get_stats()
        assert stats["categories"]["product"] == 2
        assert stats["categories"]["faq"] == 1
        assert stats["sources_count"] >= 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_source(self, kb: KBManager):
        """删除不存在的来源应返回 0"""
        await kb.add_documents(texts=["doc"], source="exists.md")
        removed = kb.delete_by_source("nonexistent.md")
        assert removed == 0
        assert kb.get_stats()["total_chunks"] == 1
