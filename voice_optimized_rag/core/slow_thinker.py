"""
Slow Thinker（慢思考者）—— 后台预取 Agent

核心职责：在后台持续监听对话流，利用 LLM 预测用户下一步可能问什么，
提前从向量数据库检索相关文档块并填充到语义缓存，
使 Fast Talker 在用户真正发问时能直接命中缓存（< 1ms），而无需实时检索（100ms+）。

架构角色：双 Agent 架构中的"后台思考者"
- 异步后台任务，不阻塞主对话流程
- 订阅对话事件流，对不同事件执行不同的预取策略
"""

from __future__ import annotations

import asyncio
import time
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

logger = get_logger("slow_thinker")

# ── Prompt 模板 ──
# 关键设计：要求 LLM 输出"文档描述式短语"而非问句
# 原因：知识库内容是文档形式，用描述性关键词检索比用问句效果更好
PREDICTION_PROMPT = """Based on the conversation so far, predict the {n} most likely topics the user will ask about next. For each topic, write a short document-style description (not a question) that would match relevant knowledge base content.

Conversation:
{conversation}

Latest user message: {latest}

Return ONLY a numbered list, one topic per line. Write each as a descriptive phrase that matches documentation content. Example format:
1. Enterprise plan pricing features unlimited contacts dedicated support custom SLA
2. API authentication using Bearer token API key generation permissions
3. Slack integration deal notifications slash commands setup"""

# 关键词提取 Prompt（作为 LLM 预测的降级回退方案）
KEYWORD_EXTRACTION_PROMPT = """Extract the key topics and entities from this text. Return only a comma-separated list of keywords.

Text: {text}

Keywords:"""


class SlowThinker:
    """
    后台预取 Agent，持续将上下文预填充到语义缓存

    事件处理策略：
    ┌─────────────────────────┬──────────────────────────────────────────────┐
    │ 事件类型                 │ 处理策略                                      │
    ├─────────────────────────┼──────────────────────────────────────────────┤
    │ USER_UTTERANCE          │ 检索当前问题 + LLM预测下一步话题并并行预取     │
    │ SILENCE_DETECTED        │ 从近期对话提取关键词，预取相关内容             │
    │ TOPIC_SHIFT             │ 清除旧缓存，为新话题预取内容                   │
    │ PRIORITY_RETRIEVAL      │ 紧急检索（Fast Talker 缓存未命中时触发）       │
    └─────────────────────────┴──────────────────────────────────────────────┘
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
        self._task: asyncio.Task | None = None   # 后台 asyncio 任务句柄
        self._running = False
        self._last_prediction_time: float = 0    # 上次 LLM 预测的时间（用于限速）

    async def start(self) -> None:
        """启动后台处理循环（创建 asyncio 后台任务）"""
        self._running = True
        # create_task 使 _run() 在事件循环中并发执行，不阻塞调用方
        self._task = asyncio.create_task(self._run())
        logger.info("Slow Thinker started")

    async def stop(self) -> None:
        """停止后台处理循环（取消任务并等待其完成）"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task   # 等待任务真正结束（处理 CancelledError）
            except asyncio.CancelledError:
                pass
        logger.info("Slow Thinker stopped")

    async def _run(self) -> None:
        """
        主处理循环 —— 订阅对话流并处理每个事件

        使用 async for 持续消费事件队列，直到任务被取消或 _running 为 False。
        单个事件处理异常不会终止整个循环（只记录日志继续运行）。
        """
        subscription = self._stream.subscribe()
        try:
            async for event in subscription:
                if not self._running:
                    break
                try:
                    await self._handle_event(event)
                except Exception as e:
                    logger.error(f"Error handling event: {e}")  # 容错：单次失败不崩溃
        except asyncio.CancelledError:
            pass  # 任务被取消是正常退出，不作为错误处理
        finally:
            if hasattr(subscription, "unsubscribe"):
                await subscription.unsubscribe()  # 退订，从全局订阅者列表移除

    async def _handle_event(self, event: StreamEvent) -> None:
        """将事件路由到对应的处理方法"""
        if event.event_type == EventType.USER_UTTERANCE:
            await self._on_user_utterance(event)
        elif event.event_type == EventType.SILENCE_DETECTED:
            await self._on_silence(event)
        elif event.event_type == EventType.TOPIC_SHIFT:
            await self._on_topic_shift(event)
        elif event.event_type == EventType.PRIORITY_RETRIEVAL:
            await self._on_priority_retrieval(event)

    async def _on_user_utterance(self, event: StreamEvent) -> None:
        """
        处理用户发言事件：检索当前问题 + 预测并预取后续话题

        限速保护：两次预测之间至少间隔 slow_thinker_rate_limit 秒，
        防止用户快速连续发言时 LLM 被反复调用（节省 API 费用）。

        并行预取：所有预测话题同时发起检索（asyncio.gather），
        最大化利用 I/O 等待时间。
        """
        # ── 限速检查 ──
        now = time.time()
        if now - self._last_prediction_time < self._config.slow_thinker_rate_limit:
            return
        self._last_prediction_time = now

        # ── Step 1: 立即检索当前发言的相关内容（保证本轮对话有上下文）──
        await self._retrieve_and_cache(event.text)

        # ── Step 2: 用 LLM 预测用户接下来可能问的 N 个话题 ──
        predictions = await self._predict_followups(event.text)
        self._metrics.increment("predictions_made", len(predictions))

        # ── Step 3: 为每个预测话题并行检索并缓存（提前热身缓存）──
        if predictions:
            tasks = [self._retrieve_and_cache(pred) for pred in predictions]
            await asyncio.gather(*tasks, return_exceptions=True)
            # return_exceptions=True：单个预取失败不影响其他预取任务

    async def _on_silence(self, event: StreamEvent) -> None:
        """
        处理静音事件：趁用户沉默时预取更多相关内容

        策略：从近期对话提取关键词，用关键词做宽泛检索，
        为可能的延伸问题提前准备上下文。
        """
        conversation = self._stream.get_conversation_text(max_turns=4)
        if not conversation:
            return

        keywords = await self._extract_keywords(conversation)
        if keywords:
            await self._retrieve_and_cache(keywords)

    async def _on_topic_shift(self, event: StreamEvent) -> None:
        """
        处理话题切换事件：清理旧缓存，预取新话题内容

        当检测到话题大幅转变时，旧话题的缓存内容价值降低，
        主动清理一半 TTL 的旧条目，为新话题释放缓存空间。
        """
        # 清除超过半个 TTL 的条目（它们更可能与旧话题相关）
        await self._cache.clear_stale(self._config.cache_ttl_seconds / 2)
        if event.text:
            await self._retrieve_and_cache(event.text)

    async def _on_priority_retrieval(self, event: StreamEvent) -> None:
        """
        处理紧急检索事件（Fast Talker 缓存未命中时触发）

        Fast Talker 在缓存未命中时会发布此事件，
        Slow Thinker 以更大的 top_k 立即检索并缓存，
        确保下一个相似问题能命中缓存。
        """
        with Timer(self._metrics, "slow_thinker", "priority_retrieval"):
            # prefetch_top_k × 2：紧急情况下获取更多文档，提高后续命中率
            await self._retrieve_and_cache(event.text, top_k=self._config.prefetch_top_k * 2)

    async def _predict_followups(self, latest_utterance: str) -> list[str]:
        """
        根据配置选择预测策略

        - "llm"：调用大模型预测（更准确，但有网络延迟）
        - "keyword"：关键词提取（更快，精度较低，作为降级方案）
        """
        if self._config.prediction_strategy == "keyword":
            return await self._predict_keyword(latest_utterance)
        return await self._predict_llm(latest_utterance)

    async def _predict_llm(self, latest_utterance: str) -> list[str]:
        """
        使用 LLM 预测后续话题（主预测策略）

        Prompt 设计要点：
        - 提供完整的近期对话上下文（最多 6 轮）
        - 要求输出"文档描述短语"而非问句（更贴近知识库内容形式）
        - 要求编号列表格式，便于解析

        解析逻辑：去掉 "1. " 或 "- " 前缀，提取纯话题文本。
        LLM 预测失败时自动降级到关键词提取。
        """
        with Timer(self._metrics, "slow_thinker", "prediction"):
            conversation = self._stream.get_conversation_text(max_turns=6)
            prompt = PREDICTION_PROMPT.format(
                n=self._config.max_predictions,
                conversation=conversation,
                latest=latest_utterance,
            )
            try:
                response = await self._llm.generate(prompt)
                predictions = []
                for line in response.strip().split("\n"):
                    line = line.strip()
                    # 去掉 "1. "、"2. " 等编号前缀
                    if line and line[0].isdigit():
                        line = line.split(".", 1)[-1].strip()
                    elif line.startswith("- "):
                        line = line[2:].strip()
                    if line:
                        predictions.append(line)
                return predictions[: self._config.max_predictions]
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                # LLM 调用失败，降级到关键词提取
                return await self._predict_keyword(latest_utterance)

    async def _predict_keyword(self, text: str) -> list[str]:
        """
        关键词提取预测（降级/备用策略）

        直接提取文本关键词作为检索查询，
        不需要完整的对话上下文，适合作为 LLM 失败时的兜底。
        """
        keywords = await self._extract_keywords(text)
        return [keywords] if keywords else []

    async def _extract_keywords(self, text: str) -> str:
        """
        用 LLM 提取关键词（逗号分隔的关键词列表）

        失败时的兜底：直接返回原始文本作为关键词（粗糙但可用）。
        """
        try:
            prompt = KEYWORD_EXTRACTION_PROMPT.format(text=text)
            return await self._llm.generate(prompt)
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return text  # 兜底：原文本本身也可以作为检索查询

    async def _retrieve_and_cache(self, query: str, top_k: int | None = None) -> None:
        """
        核心预取逻辑：检索文档块并存入语义缓存

        关键设计决策：用文档块自身的向量作为缓存键（而非查询向量）
        ──────────────────────────────────────────────────────────────
        原因：Slow Thinker 用预测的话题（如"API 认证方式"）检索，
        但 Fast Talker 用用户的实际问题（如"怎么生成 API Key？"）查缓存。

        如果用查询向量做缓存键：
          缓存键 = embed("API 认证方式")
          Fast Talker 查询 = embed("怎么生成 API Key？")
          → 两者向量不同，缓存未命中 ✗

        如果用文档向量做缓存键（当前实现）：
          缓存键 = embed(文档块本身)  ← 与话题无关，只取决于文档内容
          Fast Talker 查询 = embed("怎么生成 API Key？")
          → 与文档块语义接近，缓存命中 ✓
        ──────────────────────────────────────────────────────────────
        """
        k = top_k or self._config.prefetch_top_k

        with Timer(self._metrics, "slow_thinker", "retrieval"):
            # 将预测话题向量化
            query_embedding = await self._embeddings.embed_single(query)

            # 从向量数据库检索，include_embeddings=True 让结果携带文档块自身的向量
            results = self._vector_store.search(
                query_embedding, top_k=k, include_embeddings=True,
            )

            if not results:
                return

            # 将每个文档块以其"自身向量"为键存入缓存
            for result in results:
                # 优先使用文档块自身向量，无则退而使用查询向量
                cache_key = result.embedding if result.embedding is not None else query_embedding
                await self._cache.put(
                    query_embedding=cache_key,
                    text=result.text,
                    metadata=result.metadata,
                    relevance_score=result.score,
                    ttl=self._config.cache_ttl_seconds,
                )

        self._metrics.increment("prefetch_operations")
        logger.debug(f"Pre-fetched {len(results)} chunks for: {query[:50]}...")
