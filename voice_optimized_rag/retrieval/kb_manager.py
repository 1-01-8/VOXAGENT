"""
知识库热更新接口 —— 增量添加/删除文档块

支持：
- 增量 upsert：添加新文档块到向量数据库
- 删除：按来源文件或分块 ID 删除文档块
- 分类标签：按 产品/政策/FAQ/竞品 打标签存入 metadata
- 对 FAISSVectorStore 实现 rebuild，对 Qdrant 实现 upsert/delete

无需重启系统即可更新知识库内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from voice_optimized_rag.retrieval.document_loader import (
    DocumentChunk,
    chunk_text,
    load_directory,
)
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("kb_manager")

# 知识库类别常量
KB_CATEGORIES = {
    "product": "产品知识库",
    "policy": "销售政策库",
    "faq": "FAQ问答库",
    "competitor": "竞品对比库",
    "aftermarket": "售后规则库",
    "case": "案例库",
}


class KBManager:
    """
    知识库热更新管理器

    提供增量更新接口，支持不停机更新知识库内容。
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embedding_provider

    async def add_documents(
        self,
        texts: list[str],
        category: str = "",
        source: str = "",
        metadata_list: Optional[list[dict]] = None,
    ) -> int:
        """
        增量添加文档块

        Args:
            texts: 文本列表（已分块）
            category: 知识库类别标签
            source: 来源文件标识
            metadata_list: 自定义元数据列表

        Returns:
            成功添加的文档块数量
        """
        if not texts:
            return 0

        # 构建元数据
        metadata = []
        for i, text in enumerate(texts):
            meta = (metadata_list[i] if metadata_list and i < len(metadata_list) else {}).copy()
            if category:
                meta["category"] = category
            if source:
                meta["source"] = source
            meta["chunk_index"] = i
            metadata.append(meta)

        # 批量 Embedding
        embeddings = await self._embeddings.embed(texts)

        # 写入向量数据库
        self._vector_store.add_documents(texts, embeddings, metadata)
        logger.info(f"Added {len(texts)} chunks (category={category}, source={source})")
        return len(texts)

    async def add_file(
        self,
        file_path: Path,
        category: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> int:
        """
        添加单个文件（自动分块）

        Args:
            file_path: 文件路径
            category: 知识库类别
            chunk_size: 分块大小
            chunk_overlap: 分块重叠

        Returns:
            成功添加的文档块数量
        """
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        return await self.add_documents(
            texts=chunks,
            category=category,
            source=str(file_path),
        )

    async def add_directory(
        self,
        directory: Path,
        category: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> int:
        """
        添加整个目录（自动分块）
        """
        doc_chunks = load_directory(directory, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not doc_chunks:
            return 0

        texts = [c.text for c in doc_chunks]
        metadata = []
        for c in doc_chunks:
            meta = c.metadata.copy()
            if category:
                meta["category"] = category
            metadata.append(meta)

        embeddings = await self._embeddings.embed(texts)
        self._vector_store.add_documents(texts, embeddings, metadata)
        logger.info(f"Added {len(texts)} chunks from directory {directory}")
        return len(texts)

    def delete_by_source(self, source: str) -> int:
        """
        按来源文件删除文档块

        注意：FAISS 不支持原生删除，需要重建索引。
        对于 Qdrant，使用 payload 过滤删除。

        Args:
            source: 来源文件路径标识

        Returns:
            删除的文档块数量
        """
        if hasattr(self._vector_store, "delete_by_metadata"):
            return self._vector_store.delete_by_metadata("source", source)

        # FAISS fallback: 重建索引（排除匹配 source 的文档）
        indices_to_keep = [
            i for i, meta in enumerate(self._vector_store._metadata)
            if meta.get("source") != source
        ]
        removed = len(self._vector_store._metadata) - len(indices_to_keep)

        if removed > 0:
            self._rebuild_faiss(indices_to_keep)
            logger.info(f"Deleted {removed} chunks from source: {source}")

        return removed

    def delete_by_category(self, category: str) -> int:
        """按类别删除文档块"""
        if hasattr(self._vector_store, "delete_by_metadata"):
            return self._vector_store.delete_by_metadata("category", category)

        indices_to_keep = [
            i for i, meta in enumerate(self._vector_store._metadata)
            if meta.get("category") != category
        ]
        removed = len(self._vector_store._metadata) - len(indices_to_keep)

        if removed > 0:
            self._rebuild_faiss(indices_to_keep)
            logger.info(f"Deleted {removed} chunks in category: {category}")

        return removed

    def _rebuild_faiss(self, keep_indices: list[int]) -> None:
        """重建 FAISS 索引（保留指定索引的文档）"""
        import faiss

        kept_texts = [self._vector_store._texts[i] for i in keep_indices]
        kept_metadata = [self._vector_store._metadata[i] for i in keep_indices]

        # 重建索引
        new_index = faiss.IndexFlatIP(self._vector_store._dimension)
        if keep_indices:
            # 从旧索引中提取保留文档的向量
            kept_embeddings = np.stack([
                self._vector_store._index.reconstruct(i) for i in keep_indices
            ]).astype(np.float32)
            new_index.add(kept_embeddings)

        self._vector_store._index = new_index
        self._vector_store._texts = kept_texts
        self._vector_store._metadata = kept_metadata
        if hasattr(self._vector_store, "bump_version"):
            self._vector_store.bump_version()

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        categories: dict[str, int] = {}
        sources: dict[str, int] = {}

        for meta in self._vector_store._metadata:
            cat = meta.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        return {
            "total_chunks": self._vector_store.size,
            "categories": categories,
            "sources_count": len(sources),
        }
