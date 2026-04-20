"""
Fast Talker（快速响应者）—— 前台低延迟响应 Agent

核心职责：以最低延迟响应用户查询。
优先从 Slow Thinker 预填充好的语义缓存中获取上下文（< 1ms），
缓存未命中时降级到直接向量数据库检索（正常 RAG 延迟），
最终将上下文交给 LLM 生成回答。

架构角色：双 Agent 架构中的"前台响应者"
- 与用户交互的主路径，对延迟极为敏感
- 不做任何耗时的后台工作，专注于"快速取用已准备好的数据"
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector, Timer

logger = get_logger("fast_talker")


class FastTalker:
    """
    前台响应 Agent，专注最小化响应延迟

    响应流程：
      1. 接收用户查询
      2. 异步向量化查询（embed）
      3. 查询语义缓存（sub-ms，缓存命中率 75%+）
         ├── 命中 → 用缓存上下文直接生成回答
         └── 未命中 → 降级到 FAISS 直接检索（正常 RAG 速度）
                      → 将检索结果写入缓存（热身，下次命中）
                      → 发布 PRIORITY_RETRIEVAL 事件通知 Slow Thinker 加速预取
      4. 流式输出 LLM 回答
    """

    def __init__(
        self,
        config: VORConfig,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        vector_store: FAISSVectorStore,
        cache: SemanticCache,
        stream: ConversationStream,
        metrics: MetricsCollector,
    ) -> None:
        self._config = config
        self._llm = llm
        self._embeddings = embedding_provider
        self._vector_store = vector_store
        self._cache = cache
        self._stream = stream
        self._metrics = metrics

    async def respond(self, query: str) -> str:
        """
        生成完整回答（非流式）

        Args:
            query: 用户的问题/语音转文字结果

        Returns:
            完整的回答文本字符串
        """
        with Timer(self._metrics, "fast_talker", "total_response") as timer:
            context = await self._get_context(query)   # 获取上下文（缓存或检索）
            response = await self._llm.generate(query, context=context)

        logger.info(f"Response generated in {timer.elapsed_ms:.1f}ms (context: {len(context)} chars)")
        return response

    async def respond_stream(self, query: str) -> AsyncIterator[str]:
        """
        流式生成回答（逐 token 输出，适合语音场景）

        语音 Agent 的 TTS 可以在收到第一个 token 时就开始合成，
        而不必等待完整回答，进一步降低端到端延迟。

        Args:
            query: 用户的问题

        Yields:
            LLM 生成的文本块（每次 yield 一小段）
        """
        with Timer(self._metrics, "fast_talker", "total_response"):
            context = await self._get_context(query)
            first_token = True
            async for chunk in self._llm.stream(query, context=context):
                if first_token:
                    # 记录首 token 延迟（TTFT：Time To First Token）
                    # 这是语音对话中最关键的延迟指标
                    self._metrics.record_latency("fast_talker", "time_to_first_token", 0)
                    first_token = False
                yield chunk

    async def _get_context(self, query: str) -> str:
        """
        双路径上下文获取策略（缓存优先，向量数据库兜底）

        ┌─────────────────────────────────────────────────────────┐
        │  快路径（缓存命中，~0.35ms）                              │
        │  embed(query) → 缓存语义搜索 → 命中 → 返回上下文        │
        ├─────────────────────────────────────────────────────────┤
        │  慢路径（缓存未命中，~110ms）                             │
        │  → FAISS 直接检索 → 结果写入缓存 → 发送紧急预取信号      │
        ├─────────────────────────────────────────────────────────┤
        │  无路径（向量库也无结果）                                  │
        │  → 发送紧急预取信号 → 返回空字符串（LLM 凭参数记忆回答）  │
        └─────────────────────────────────────────────────────────┘

        为什么能实现高缓存命中率？
        Slow Thinker 用文档向量（而非查询向量）作为缓存键，
        因此用户任何语义相近的问法都能命中同一条缓存内容。
        """
        # ── Step 1: 向量化查询 ──
        with Timer(self._metrics, "fast_talker", "embedding"):
            query_embedding = await self._embeddings.embed_single(query)

        # ── Step 2: 尝试语义缓存（快路径）──
        with Timer(self._metrics, "fast_talker", "cache_lookup") as cache_timer:
            cached = await self._cache.get(
                query_embedding,
                top_k=self._config.fast_talker_max_context_chunks,
            )

        if cached:
            # 缓存命中 → 直接格式化上下文返回，无需访问向量数据库
            logger.debug(f"Cache HIT: {len(cached)} chunks in {cache_timer.elapsed_ms:.2f}ms")
            context_chunks = [entry.text for entry in cached]
            return self._format_context(context_chunks)

        # ── Step 3: 缓存未命中 → 降级到 FAISS 直接检索（慢路径）──
        logger.debug("Cache MISS — falling back to retrieval")

        if self._config.fast_talker_fallback_to_retrieval:
            with Timer(self._metrics, "fast_talker", "fallback_retrieval"):
                results = self._vector_store.search(
                    query_embedding,
                    top_k=self._config.fast_talker_max_context_chunks,
                    include_embeddings=True,
                )
                if results:
                    context_chunks = [r.text for r in results]

                    # 将检索结果写入缓存（以文档向量为键），热身缓存
                    # 下一个相似问题就能命中缓存，不再走慢路径
                    for r in results:
                        cache_key = r.embedding if r.embedding is not None else query_embedding
                        await self._cache.put(
                            query_embedding=cache_key,
                            text=r.text,
                            metadata=r.metadata,
                            relevance_score=r.score,
                        )

                    # 通知 Slow Thinker 当前话题需要紧急预取更多内容
                    await self._stream.publish(StreamEvent(
                        event_type=EventType.PRIORITY_RETRIEVAL,
                        text=query,
                    ))
                    return self._format_context(context_chunks)

        # ── Step 4: 向量库也无结果 → 纯参数记忆回答 ──
        # 通知 Slow Thinker 尽快预取，为后续相关问题准备好上下文
        await self._stream.publish(StreamEvent(
            event_type=EventType.PRIORITY_RETRIEVAL,
            text=query,
        ))
        return ""  # 空上下文，LLM 依靠自身训练知识回答（可能不准确）

    def _format_context(self, chunks: list[str]) -> str:
        """
        将多个文档块格式化为单一上下文字符串

        格式：
          [1] 第一个文档块内容...

          [2] 第二个文档块内容...

        编号让 LLM 能在回答中引用来源（"根据[1]..."）。
        """
        if not chunks:
            return ""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[{i}] {chunk}")
        return "\n\n".join(formatted)
