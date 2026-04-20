"""
语义相似度缓存 —— 亚毫秒级上下文检索

这是 Slow Thinker 和 Fast Talker 之间的核心桥梁：
- Slow Thinker 负责写入（put）：把预取的文档块存进来
- Fast Talker 负责读取（get）：用用户的查询向量找最相关的缓存块

内部使用 FAISS IndexFlatIP（内积索引）做余弦相似度搜索，
相比完整的向量数据库查询（100ms+），缓存命中只需 < 1ms。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import faiss
import numpy as np

from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector

logger = get_logger("semantic_cache")


@dataclass
class CachedContext:
    """
    一条缓存条目，存储预取的文档块及其元信息

    TTL 机制：创建时记录 created_at，超过 ttl 秒后标记为过期。
    LRU 机制：记录 last_accessed，缓存满时淘汰最久未访问的条目。
    """
    text: str                                               # 文档块原文
    metadata: dict                                          # 来源、分块 ID 等元信息
    embedding: np.ndarray                                   # 该文档块的向量（作为缓存键）
    relevance_score: float                                  # 与原始查询的相关性分数（0-1）
    created_at: float = field(default_factory=time.time)   # 创建时间戳
    ttl: float = 300.0                                      # 生存时间（秒），默认 5 分钟
    access_count: int = 0                                   # 被访问次数（统计用）
    last_accessed: float = field(default_factory=time.time) # 最近访问时间（LRU 依据）

    @property
    def is_expired(self) -> bool:
        """判断该条目是否已超过 TTL 过期"""
        return (time.time() - self.created_at) > self.ttl


class SemanticCache:
    """
    基于 FAISS 的内存语义缓存

    核心设计：
    - 用文档块自身的向量（而非查询向量）作为缓存键
      → 不同的查询问法只要语义相近，就能命中同一条缓存
    - 使用 IndexFlatIP（内积）+ L2 归一化 = 余弦相似度搜索
    - asyncio.Lock 保证并发写入安全（Slow Thinker 异步写，Fast Talker 异步读）

    淘汰策略（双重）：
    1. TTL 过期淘汰：条目创建超过 ttl 秒后在下次写入时被清除
    2. LRU 淘汰：缓存容量满时淘汰最久未访问的条目
    """

    def __init__(
        self,
        dimension: int,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        similarity_threshold: float = 0.75,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._dimension = dimension                              # 向量维度（由 embedding 模型决定）
        self._max_size = max_size                                # 最大缓存条目数
        self._default_ttl = default_ttl                         # 默认 TTL（秒）
        self._similarity_threshold = similarity_threshold       # 命中阈值（余弦相似度，0-1）
        self._metrics = metrics or MetricsCollector()
        self._lock = asyncio.Lock()                             # 保证并发安全

        self._entries: list[CachedContext] = []                 # 缓存条目列表（与 FAISS 索引位置对应）
        # IndexFlatIP：内积索引，配合 L2 归一化实现余弦相似度
        # 相比 IndexFlatL2，内积搜索在已归一化向量上等价于余弦相似度且更快
        self._index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        """当前缓存中的条目数量"""
        return len(self._entries)

    async def put(
        self,
        query_embedding: np.ndarray,
        text: str,
        metadata: dict | None = None,
        relevance_score: float = 1.0,
        ttl: float | None = None,
    ) -> None:
        """
        向缓存中添加一条文档块记录

        写入前先查重：若已存在余弦相似度 > 0.95 的近似条目，
        直接更新其内容而非新增（避免重复占用空间）。

        Args:
            query_embedding: 文档块的向量（作为缓存键）
            text: 文档块原文
            metadata: 附加元信息（来源文件、分块 ID 等）
            relevance_score: 与检索查询的相关性分数
            ttl: 该条目的生存时间（None 使用默认值）
        """
        async with self._lock:
            # ── 查重：避免存入近似重复内容 ──
            if self._index.ntotal > 0:
                query = query_embedding.reshape(1, -1).astype(np.float32).copy()
                faiss.normalize_L2(query)                      # 归一化使内积等于余弦相似度
                scores, indices = self._index.search(query, 1) # 找最相似的 1 条
                if scores[0][0] > 0.95 and indices[0][0] != -1:
                    # 相似度 > 95%，视为重复，更新已有条目而非新增
                    idx = int(indices[0][0])
                    if idx < len(self._entries):
                        self._entries[idx].text = text
                        self._entries[idx].relevance_score = relevance_score
                        self._entries[idx].created_at = time.time()  # 刷新创建时间（延长 TTL）
                        return

            # ── 淘汰过期条目（每次写入时顺带清理） ──
            self._evict_expired()

            # ── 容量满时淘汰最久未访问的条目（LRU）──
            if len(self._entries) >= self._max_size:
                self._evict_lru()

            # ── 添加新条目 ──
            embedding = query_embedding.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(embedding)                      # 归一化后写入 FAISS
            self._index.add(embedding)
            self._entries.append(CachedContext(
                text=text,
                metadata=metadata or {},
                embedding=query_embedding,                     # 保存原始（未归一化）向量供后续使用
                relevance_score=relevance_score,
                ttl=ttl or self._default_ttl,
            ))

    async def get(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[CachedContext]:
        """
        从缓存中检索语义相关的文档块

        流程：归一化查询向量 → FAISS 内积搜索 → 过滤过期/低相似度条目 → 更新访问记录

        Args:
            query_embedding: 用户查询的向量
            top_k: 最多返回的结果数
            similarity_threshold: 命中阈值（None 使用构造时的默认值）

        Returns:
            按 relevance_score 降序排列的缓存条目列表，空列表表示未命中
        """
        threshold = similarity_threshold or self._similarity_threshold

        async with self._lock:
            if self._index.ntotal == 0:
                self._metrics.increment("cache_miss")
                return []

            query = query_embedding.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(query)

            # min 确保 top_k 不超过索引中的实际向量数
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query, k)

            results: list[CachedContext] = []
            now = time.time()
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self._entries):
                    continue                        # FAISS 返回 -1 表示该位置无有效结果
                entry = self._entries[idx]
                if entry.is_expired:
                    continue                        # 跳过已过期条目（延迟删除）
                if score >= threshold:             # 余弦相似度超过阈值才算命中
                    entry.access_count += 1
                    entry.last_accessed = now      # 更新 LRU 时间戳
                    results.append(entry)

            if results:
                self._metrics.increment("cache_hit")
                logger.debug(f"Cache hit: {len(results)} results (best score: {scores[0][0]:.3f})")
            else:
                self._metrics.increment("cache_miss")
                logger.debug(f"Cache miss (best score: {scores[0][0]:.3f} < threshold {threshold})")

            # 按 relevance_score 降序排列（优先返回原本就最相关的文档块）
            return sorted(results, key=lambda e: e.relevance_score, reverse=True)

    async def clear(self) -> None:
        """清空所有缓存条目（话题大幅切换时调用）"""
        async with self._lock:
            self._entries.clear()
            self._index = faiss.IndexFlatIP(self._dimension)  # 重建空索引

    async def clear_stale(self, max_age: float | None = None) -> int:
        """
        清除过期条目，返回清除数量

        Args:
            max_age: 超过此秒数的条目视为过期（None 使用各条目自身的 TTL）
        """
        async with self._lock:
            return self._evict_expired(max_age)

    def _evict_expired(self, max_age: float | None = None) -> int:
        """
        清除过期条目（必须在锁内调用）

        由于 FAISS 不支持按位置删除，需要记录要保留的索引，
        然后重建整个 FAISS 索引（_rebuild_index）。
        """
        now = time.time()
        to_keep: list[int] = []
        removed = 0

        for i, entry in enumerate(self._entries):
            expired = entry.is_expired
            if max_age is not None:
                # 额外检查：超过 max_age 秒的也视为过期（用于主动清理）
                expired = expired or (now - entry.created_at) > max_age
            if not expired:
                to_keep.append(i)
            else:
                removed += 1

        if removed > 0:
            self._rebuild_index(to_keep)
            logger.debug(f"Evicted {removed} expired entries")

        return removed

    def _evict_lru(self) -> None:
        """
        淘汰最久未访问的条目（必须在锁内调用）

        找到 last_accessed 最小的条目索引，从保留列表中排除它，
        然后重建 FAISS 索引。
        """
        if not self._entries:
            return

        # argmin of last_accessed → 最久未访问的条目
        lru_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].last_accessed)
        to_keep = [i for i in range(len(self._entries)) if i != lru_idx]
        self._rebuild_index(to_keep)
        logger.debug("Evicted LRU entry")

    def _rebuild_index(self, keep_indices: list[int]) -> None:
        """
        根据保留索引重建 FAISS 索引

        FAISS 的 IndexFlatIP 不支持删除操作，
        唯一的删除方式是重建索引（保留要保留的向量）。

        流程：过滤 _entries 列表 → 创建新的空 FAISS 索引 → 批量添加保留的向量
        """
        kept_entries = [self._entries[i] for i in keep_indices]
        self._entries = kept_entries

        # 重建空索引
        self._index = faiss.IndexFlatIP(self._dimension)
        if kept_entries:
            # 批量归一化并重新添加所有保留的向量
            embeddings = np.stack([e.embedding for e in kept_entries]).astype(np.float32)
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
