"""Shared test fixtures for all test modules.

提供全项目测试的 Mock 对象和共享 Fixture：
- MockLLM: 可编程的 LLM 模拟（支持意图分类、ReAct 等场景）
- MockEmbedding: 确定性 embedding 输出
- 各模块所需的 fixture（stream, session, tools 等）
- 模块级计时器：统计每个测试文件执行耗时
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import AsyncIterator

import numpy as np
import pytest


# ───────────────────────── Module Timer Plugin ─────────────────────────

# 组件映射：测试文件 → 系统组件名称
COMPONENT_MAP: dict[str, str] = {
    "test_session": "Dialogue — SessionContext 会话状态管理",
    "test_intent_router": "Dialogue — IntentRouter 三路意图路由",
    "test_business_scope": "Dialogue — 业务问答模板",
    "test_emotion_detector": "Dialogue — EmotionDetector 情绪检测引擎",
    "test_memory_manager": "Dialogue — MemoryManager 多轮记忆管理",
    "test_transfer_policy": "Dialogue — TransferPolicy 转人工策略",
    "test_domain_router": "Dialogue — DomainRouter 三域路由",
    "test_task_state_machine": "Dialogue — TaskStateMachine 显式任务流",
    "test_llm_tracing": "LLM — 调用轨迹记录",
    "test_base_tool": "Agent — BaseTool 工具抽象层",
    "test_tools": "Agent — Business Tools 业务工具集(7个)",
    "test_permission_guard": "Agent — PermissionGuard 四级权限拦截",
    "test_react_agent": "Agent — ReactAgent ReAct 推理循环",
    "test_domain_agent": "Agent — DomainAgent 三领域工厂",
    "test_function_calling_agent": "Agent — FunctionCallingAgent 原生函数调用",
    "test_skill_registry": "Agent — SkillRegistry 正式技能注册表",
    "test_mcp_server": "MCP — 业务互操作服务导出",
    "test_kb_manager": "Retrieval — KBManager 知识库热更新",
    "test_hybrid_retriever": "Retrieval — HybridRetriever 混合检索",
    "test_session_logger": "Utils — SessionLogger 会话日志(JSONL)",
    "test_auto_qa": "Utils — AutoQA 自动质检引擎",
    "test_sensevoice_stt": "Voice — SenseVoice STT 语音识别",
    "test_cosyvoice": "Voice — CosyVoice TTS 语音合成",
    "test_fast_talker": "Core — FastTalker 前台响应代理",
    "test_slow_thinker": "Core — SlowThinker 后台预取代理",
    "test_semantic_cache": "Core — SemanticCache 语义缓存",
    "test_memory_router": "Core — MemoryRouter 路由协调器",
    "test_vector_store": "Core — FAISSVectorStore 向量数据库",
    "test_integration": "Core — 端到端集成测试",
}

_module_timings: dict[str, list[float]] = defaultdict(list)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """收集每个测试的 setup/call/teardown 耗时。"""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        module_name = item.module.__name__.split(".")[-1]
        _module_timings[module_name].append(report.duration)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """在测试结束后打印模块级耗时统计表。"""
    if not _module_timings:
        return

    terminalreporter.section("Module Timing Report (模块耗时统计)")
    terminalreporter.write_line(
        f"{'Module':<30} {'Component':<45} {'Tests':>5} {'Total(ms)':>10} {'Avg(ms)':>10}"
    )
    terminalreporter.write_line("─" * 105)

    total_tests = 0
    total_time = 0.0
    for module, durations in sorted(_module_timings.items()):
        component = COMPONENT_MAP.get(module, "Unknown")
        count = len(durations)
        total_ms = sum(durations) * 1000
        avg_ms = total_ms / count if count else 0
        total_tests += count
        total_time += total_ms
        terminalreporter.write_line(
            f"{module:<30} {component:<45} {count:>5} {total_ms:>9.2f} {avg_ms:>9.2f}"
        )

    terminalreporter.write_line("─" * 105)
    terminalreporter.write_line(
        f"{'TOTAL':<30} {'':<45} {total_tests:>5} {total_time:>9.2f} {total_time / total_tests:>9.2f}"
    )

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.llm.base import LLMProvider, ToolCallingResponse
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.metrics import MetricsCollector
from voice_optimized_rag.dialogue.session import SessionContext
from voice_optimized_rag.agent.base_tool import BaseTool, ToolResult


# ───────────────────────── Mock Providers ─────────────────────────

class MockLLM(LLMProvider):
    """Mock LLM that returns predictable responses.

    可通过 response_queue 预设多轮返回值，适合测试 ReAct 循环等场景。
    """

    def __init__(self, response: str = "Mock response") -> None:
        self._response = response
        self.call_count = 0
        self.last_prompt = ""
        self.last_context = ""
        self.last_tools: list[dict] = []
        self._supports_function_calling = False
        # 可预设多轮返回值（先入先出）
        self.response_queue: list[str] = []
        self.tool_call_queue: list[ToolCallingResponse] = []

    @property
    def supports_function_calling(self) -> bool:
        return self._supports_function_calling

    async def generate(self, prompt: str, context: str = "") -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        # 若有预设队列，优先消费
        if self.response_queue:
            return self.response_queue.pop(0)
        if "predict" in prompt.lower() and "follow-up" in prompt.lower():
            return "1. What is the pricing?\n2. How does billing work?\n3. Are there discounts?"
        if "keywords" in prompt.lower():
            return "pricing, billing, discounts"
        return self._response

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        for word in self._response.split():
            yield word + " "

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        context: str = "",
        tool_choice: str = "auto",
    ) -> ToolCallingResponse:
        if not self._supports_function_calling:
            raise NotImplementedError("MockLLM tool calling disabled")

        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        self.last_tools = tools
        if self.tool_call_queue:
            return self.tool_call_queue.pop(0)
        return ToolCallingResponse(content=self._response)


class MockEmbedding(EmbeddingProvider):
    """Mock embedding provider that returns deterministic embeddings."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


class MockTool(BaseTool):
    """通用 Mock 工具，用于 Agent 测试。"""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool",
        permission_level: int = 1,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._permission_level = permission_level
        self._result = result or ToolResult(success=True, data={"status": "ok"}, message="done")
        self.call_count = 0
        self.last_kwargs: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def permission_level(self) -> int:
        return self._permission_level

    async def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        self.last_kwargs = kwargs
        return self._result


# ───────────────────────── Fixtures ─────────────────────────

@pytest.fixture
def dim() -> int:
    return 64


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_embeddings(dim: int) -> MockEmbedding:
    return MockEmbedding(dim=dim)


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def cache(dim: int, metrics: MetricsCollector) -> SemanticCache:
    return SemanticCache(
        dimension=dim,
        max_size=100,
        default_ttl=60.0,
        similarity_threshold=0.5,
        metrics=metrics,
    )


@pytest.fixture
def vector_store(dim: int) -> FAISSVectorStore:
    return FAISSVectorStore(dimension=dim)


@pytest.fixture
def stream() -> ConversationStream:
    return ConversationStream(window_size=10)


@pytest.fixture
def session() -> SessionContext:
    """每个测试获得一个全新的会话上下文。"""
    return SessionContext()


@pytest.fixture
def config() -> VORConfig:
    return VORConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        embedding_dimension=64,
        cache_max_size=100,
        cache_ttl_seconds=60.0,
        cache_similarity_threshold=0.5,
    )
