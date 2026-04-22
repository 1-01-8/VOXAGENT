"""
Memory Router（记忆路由器）—— 双 Agent 架构的中央协调器

这是整个 VoiceAgentRAG 系统的主入口和总调度器，负责：
1. 初始化并管理所有子组件（LLM、Embedding、向量数据库、缓存、对话流、两个 Agent）
2. 对外暴露简洁的 query / query_stream / ingest 接口
3. 协调 Slow Thinker（后台预取）和 Fast Talker（前台响应）的协作

典型使用流程：
    config = VORConfig(llm_provider="openai", llm_api_key="sk-...")
    router = MemoryRouter(config)
    await router.start()              # 启动 Slow Thinker 后台任务
    await router.ingest_directory(Path("docs/"))  # 导入知识库
    response = await router.query("What is the pricing?")  # 问答
    await router.stop()              # 停止后台任务
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.fast_talker import FastTalker
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.core.slow_thinker import SlowThinker
from voice_optimized_rag.llm.base import LLMProvider, create_llm
from voice_optimized_rag.retrieval.document_loader import DocumentChunk, load_directory
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider, create_embedding_provider
from voice_optimized_rag.retrieval.hybrid_retriever import HybridRetriever
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.logging import get_logger, setup_logging
from voice_optimized_rag.utils.metrics import MetricsCollector

logger = get_logger("memory_router")


class MemoryRouter:
    """
    VoiceAgentRAG 双 Agent 系统的中央协调器

    组件关系图：
    ┌─────────────────────────────────────────────────┐
    │                 MemoryRouter                     │
    │                                                  │
    │  ┌──────────────┐      ┌──────────────────────┐ │
    │  │  SlowThinker │      │     FastTalker        │ │
    │  │  (后台预取)  │      │  (前台响应, <200ms)   │ │
    │  └──────┬───────┘      └──────────┬────────────┘ │
    │         │ 写入                     │ 读取          │
    │         ▼                          ▼              │
    │  ┌─────────────────────────────────────────────┐ │
    │  │           SemanticCache (语义缓存)           │ │
    │  │     FAISS IndexFlatIP，sub-ms 检索           │ │
    │  └─────────────────────────┬───────────────────┘ │
    │                            │ 未命中时降级          │
    │                            ▼                      │
    │  ┌─────────────────────────────────────────────┐ │
    │  │    VectorStore (FAISS / Qdrant)              │ │
    │  │    完整知识库，~110ms 检索                    │ │
    │  └─────────────────────────────────────────────┘ │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │    ConversationStream (事件总线)             │ │
    │  │    SlowThinker 订阅 / MemoryRouter 发布      │ │
    │  └─────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: VORConfig | None = None,
        llm: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config or VORConfig()        # 未传入时从环境变量自动加载
        self._metrics = MetricsCollector()

        # ── 初始化 LLM 和 Embedding 提供商 ──
        # 支持依赖注入（测试时传入 mock），生产时由工厂函数根据 config 创建
        self._llm = llm or create_llm(self._config)
        self._embeddings = embedding_provider or create_embedding_provider(self._config)

        # ── 初始化向量数据库（FAISS 本地 或 Qdrant 云端）──
        if self._config.vector_store_provider == "qdrant":
            # Qdrant Cloud：持久化存储，适合生产环境和大规模知识库
            from voice_optimized_rag.retrieval.qdrant_store import QdrantVectorStore
            self._vector_store = QdrantVectorStore(
                dimension=self._embeddings.dimension,
                url=self._config.qdrant_url,
                api_key=self._config.qdrant_api_key or None,
                collection_name=self._config.qdrant_collection,
            )
        else:
            # FAISS 本地：内存存储，重启后需重新导入（适合开发和小规模场景）
            index_path = self._config.faiss_index_path
            # 如果磁盘上已有保存的索引，直接加载（避免重新 embed 所有文档）
            load_path = index_path if (index_path.exists() and (index_path / "index.faiss").exists()) else None
            self._vector_store = FAISSVectorStore(
                dimension=self._embeddings.dimension,
                index_path=load_path,
                simulated_latency_ms=self._config.retrieval_latency_ms,
            )

        self._retriever = HybridRetriever(
            config=self._config,
            vector_store=self._vector_store,
            metrics=self._metrics,
        )

        # ── 初始化语义缓存（Slow Thinker 写、Fast Talker 读）──
        self._cache = SemanticCache(
            dimension=self._embeddings.dimension,
            max_size=self._config.cache_max_size,
            default_ttl=self._config.cache_ttl_seconds,
            similarity_threshold=self._config.cache_similarity_threshold,
            metrics=self._metrics,
        )

        # ── 初始化对话事件流（发布-订阅总线）──
        self._stream = ConversationStream(
            window_size=self._config.conversation_window_size,
        )

        # ── 初始化双 Agent ──
        # Slow Thinker：后台异步任务，订阅对话流，预取并填充缓存
        self._slow_thinker = SlowThinker(
            config=self._config,
            llm=self._llm,
            embedding_provider=self._embeddings,
            vector_store=self._vector_store,
            cache=self._cache,
            stream=self._stream,
            metrics=self._metrics,
            retriever=self._retriever,
        )
        # Fast Talker：前台同步响应，先查缓存，缓存未命中时降级到向量检索
        self._fast_talker = FastTalker(
            config=self._config,
            llm=self._llm,
            embedding_provider=self._embeddings,
            vector_store=self._vector_store,
            cache=self._cache,
            stream=self._stream,
            metrics=self._metrics,
            retriever=self._retriever,
        )
        self._running = False

    async def start(self, log_level: str = "INFO") -> None:
        """
        启动 Memory Router 和所有后台 Agent

        必须在 query/query_stream 之前调用，
        否则 Slow Thinker 未运行，缓存不会被预热。
        """
        setup_logging(log_level)
        await self._slow_thinker.start()  # 启动 Slow Thinker 后台 asyncio 任务
        self._running = True
        logger.info("Memory Router started")

    async def stop(self) -> None:
        """停止 Memory Router 和所有后台 Agent（取消 Slow Thinker 任务）"""
        self._running = False
        await self._slow_thinker.stop()
        logger.info("Memory Router stopped")

    async def query(self, text: str) -> str:
        """
        处理用户查询，返回完整回答（非流式）

        流程：
        1. 发布 USER_UTTERANCE 事件 → Slow Thinker 收到后开始预取下一步话题
        2. Fast Talker 处理当前问题（查缓存/检索/生成）
        3. 发布 AGENT_RESPONSE 事件 → 更新对话历史

        Args:
            text: 用户的问题或语音识别结果

        Returns:
            LLM 生成的完整回答文本
        """
        response, _ = await self.query_with_trace(text)
        return response

    async def query_with_trace(self, text: str) -> tuple[str, dict[str, object]]:
        """Query business knowledge and return a lightweight decision trace."""
        await self._stream.publish(StreamEvent(
            event_type=EventType.USER_UTTERANCE,
            text=text,
        ))

        response = await self._fast_talker.respond(text)
        trace = dict(self._fast_talker.last_trace)

        await self._stream.publish(StreamEvent(
            event_type=EventType.AGENT_RESPONSE,
            text=response,
        ))

        return response, trace

    async def query_stream(self, text: str) -> AsyncIterator[str]:
        """
        处理用户查询，流式返回回答（适合语音 TTS 场景）

        与 query() 相同，但通过 AsyncIterator 逐 chunk 返回 LLM 输出，
        TTS 引擎可以在收到第一个 chunk 时就开始合成语音，降低感知延迟。

        Args:
            text: 用户的问题

        Yields:
            回答文本块（每次 yield 一小段，直到回答完成）
        """
        await self._stream.publish(StreamEvent(
            event_type=EventType.USER_UTTERANCE,
            text=text,
        ))

        # 收集完整回答用于写入对话历史（流式输出时需要拼接）
        full_response: list[str] = []
        async for chunk in self._fast_talker.respond_stream(text):
            full_response.append(chunk)
            yield chunk  # 立即 yield 给调用方（TTS 可以同步处理）

        # 流式回答结束后，将完整回答写入对话历史
        await self._stream.publish(StreamEvent(
            event_type=EventType.AGENT_RESPONSE,
            text="".join(full_response),
        ))

    async def ingest_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
    ) -> int:
        """
        将目录下的所有文档导入向量数据库

        流程：加载文件 → 分块 → 批量 Embedding → 写入向量数据库

        Args:
            directory: 文档目录路径
            extensions: 要包含的文件扩展名集合（None 使用默认支持格式）

        Returns:
            成功导入的文档块数量
        """
        chunks = load_directory(
            directory,
            extensions=extensions,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        return await self._ingest_chunks(chunks)

    async def ingest_texts(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        """
        将原始文本字符串列表导入向量数据库

        适用于从数据库或 API 动态获取内容的场景。

        Args:
            texts: 文本字符串列表
            metadata: 每条文本对应的元数据（None 时全部使用空字典）

        Returns:
            成功导入的文档块数量
        """
        chunks = [
            DocumentChunk(text=t, metadata=m)
            for t, m in zip(texts, metadata or [{} for _ in texts])
        ]
        return await self._ingest_chunks(chunks)

    async def _ingest_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        内部方法：批量 Embedding 并写入向量数据库

        分批处理（每批 100 条）避免单次 Embedding API 调用超时或内存溢出。
        np.vstack 将多批结果合并为单一矩阵后一次性写入向量数据库。
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadata = [c.metadata for c in chunks]
        size_before = self._vector_store.size

        # ── 分批 Embedding（避免单次请求过大）──
        batch_size = 100
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._embeddings.embed(batch)
            all_embeddings.append(embeddings)

        # 将所有批次的向量垂直拼接为 (total_chunks, dim) 矩阵
        embeddings_array = np.vstack(all_embeddings)
        self._vector_store.add_documents(texts, embeddings_array, metadata)
        added_count = self._vector_store.size - size_before
        logger.info(f"Ingested {added_count} new chunks")
        return added_count

    def save_index(self, path: Path | None = None) -> None:
        """
        将 FAISS 索引保存到磁盘（Qdrant 自动持久化，此方法对其无效）

        建议在导入文档后调用，下次启动时可直接加载，无需重新 Embedding。
        """
        if hasattr(self._vector_store, "save"):
            self._vector_store.save(path or self._config.faiss_index_path)

    @property
    def document_count(self) -> int:
        """Number of indexed document chunks currently available."""
        return self._vector_store.size

    @property
    def metrics(self) -> MetricsCollector:
        """访问性能指标收集器（缓存命中率、延迟等）"""
        return self._metrics

    @property
    def cache(self) -> SemanticCache:
        """直接访问语义缓存（用于调试和监控）"""
        return self._cache

    @property
    def stream(self) -> ConversationStream:
        """直接访问对话事件流（用于发布自定义事件）"""
        return self._stream
