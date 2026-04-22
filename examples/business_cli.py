#!/usr/bin/env python3
"""Pure business CLI for local Ollama testing without any voice APIs."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_optimized_rag.agent.domain_agent import create_domain_agents
from voice_optimized_rag.agent.permission_guard import TextPermissionGuard
from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import EventType, StreamEvent
from voice_optimized_rag.core.memory_router import MemoryRouter
from voice_optimized_rag.dialogue.business_scope import OUT_OF_SCOPE_RESPONSE
from voice_optimized_rag.dialogue.domain_router import DomainRouter
from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.session import IntentType, SessionContext
from voice_optimized_rag.dialogue.task_state_machine import BusinessTaskStateMachine
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy
from voice_optimized_rag.llm.base import create_llm
from voice_optimized_rag.llm.tracing import LLMTraceRecorder, TraceableLLMProvider
from voice_optimized_rag.utils.auto_qa import AutoQA
from voice_optimized_rag.utils.session_logger import SessionLogger


async def ensure_ollama_runtime(base_url: str, timeout_seconds: float = 10.0) -> None:
    base = base_url.rstrip("/")

    async def _healthy() -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{base}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    if await _healthy():
        return

    if not shutil.which("ollama"):
        raise RuntimeError("未检测到 ollama 命令，无法自动启动本地 LLM。")

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await _healthy():
            return
        await asyncio.sleep(0.5)

    raise RuntimeError("Ollama 本地服务启动超时，请手动执行 ollama serve 后重试。")


def _format_knowledge_trace(trace: dict[str, object]) -> str:
    path = trace.get("path", "unknown")
    retrieval_query = trace.get("retrieval_query", "")
    raw_sources = trace.get("sources", [])
    sources = raw_sources if isinstance(raw_sources, list) else [raw_sources]
    result_count = trace.get("result_count", 0)
    sources_text = ", ".join(str(item) for item in sources) if sources else "无"
    return (
        f"knowledge_rag: path={path}, retrieval_query={retrieval_query}, "
        f"result_count={result_count}, sources={sources_text}"
    )


def _print_trace(lines: list[str]) -> None:
    print("\n[模块决策轨迹]")
    for line in lines:
        print(f"- {line}")


async def run_cli(config: VORConfig, docs_dir: Path | None, trace_enabled: bool = False) -> None:
    await ensure_ollama_runtime(config.llm_base_url or "http://localhost:11434")

    base_llm = create_llm(config)
    llm_trace = LLMTraceRecorder(provider=config.llm_provider, model=config.llm_model)
    llm = TraceableLLMProvider(base_llm, llm_trace)

    router = MemoryRouter(config, llm=llm)
    await router.start()

    if docs_dir and docs_dir.exists():
        count = await router.ingest_directory(docs_dir)
        print(f"知识库已就绪: 当前索引 {router.document_count} 个文档块，本次新增 {count} 个")

    stream = router.stream
    session = SessionContext()
    intent_router = IntentRouter(llm)
    domain_router = DomainRouter(llm)
    memory_manager = MemoryManager(llm, short_term_turns=config.memory_short_term_turns)
    transfer_policy = TransferPolicy(
        stream,
        angry_threshold=config.emotion_angry_threshold,
        max_agent_failures=config.transfer_max_failures,
    )
    permission_guard = TextPermissionGuard(stream)
    domain_agents = create_domain_agents(
        llm=llm,
        permission_guard=permission_guard,
        stream=stream,
        max_iterations=config.agent_max_iterations,
        tool_timeout=config.agent_tool_timeout,
        tool_retry=config.agent_tool_retry,
    )
    task_state_machine = BusinessTaskStateMachine(permission_guard)
    auto_qa = AutoQA()
    session_logger = SessionLogger(config.session_log_dir)

    print("=" * 60)
    print("Local Business CLI")
    print(f"LLM: ollama / {config.llm_model}")
    print("语音接口已禁用，仅支持销售/售后/财务业务测试")
    print("=" * 60)
    print("输入 quit 退出 | 输入 stats 查看统计")
    if trace_enabled:
        print("当前已开启模块决策轨迹输出")

    try:
        while True:
            try:
                user_input = input("\n业务请求: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "退出"}:
                break
            if user_input.lower() == "stats":
                print(f"缓存命中率: {router.metrics.cache_hit_rate:.1%}")
                print(f"对话轮次: {session.turn_count}")
                print(f"当前业务域: {session.current_domain.value}")
                continue
            if user_input.lower() == "trace on":
                trace_enabled = True
                print("已开启模块决策轨迹输出")
                continue
            if user_input.lower() == "trace off":
                trace_enabled = False
                print("已关闭模块决策轨迹输出")
                continue

            session.increment_turn()
            llm_trace.clear()
            trace_lines: list[str] = []
            memory_context = memory_manager.get_context()
            conversation_text = memory_context or stream.get_conversation_text(max_turns=6)
            intent = await intent_router.classify(user_input, session, conversation_text)
            session.current_intent = intent
            trace_lines.append(f"intent_router: {intent_router.last_trace}")

            if intent != IntentType.OUT_OF_SCOPE:
                await domain_router.classify(
                    user_input,
                    session,
                    intent=intent,
                    conversation_text=conversation_text,
                )
                trace_lines.append(f"domain_router: {domain_router.last_trace}")

            should_transfer = await transfer_policy.evaluate(session, user_input)
            trace_lines.append(
                "transfer_policy: trigger transfer" if should_transfer else "transfer_policy: no transfer"
            )
            if should_transfer:
                print(f"转人工: {session.transfer_reason}")
                if trace_enabled:
                    _print_trace(trace_lines)
                await session_logger.log_transfer(session.session_id, session.transfer_reason)
                continue

            if intent == IntentType.TASK:
                task_result = await task_state_machine.handle(user_input, session)
                trace_lines.append(f"task_state_machine: {task_state_machine.last_trace}")
                if task_result.handled:
                    response = task_result.reply_text
                else:
                    response = await domain_agents[session.current_domain].execute(
                        user_input,
                        session,
                        memory_context,
                    )
                    trace_lines.append("domain_agent: fallback tool/agent execution")
            elif intent == IntentType.KNOWLEDGE:
                response, knowledge_trace = await router.query_with_trace(user_input)
                trace_lines.append(_format_knowledge_trace(knowledge_trace))
            else:
                response = OUT_OF_SCOPE_RESPONSE
                trace_lines.append("knowledge/task pipeline skipped -> out_of_scope response")

            qa_result = await auto_qa.check(response)
            if not qa_result.passed:
                response = qa_result.cleaned_response
            trace_lines.append(
                "auto_qa: passed"
                if qa_result.passed
                else f"auto_qa: cleaned response, issues={'; '.join(qa_result.issues)}"
            )

            if intent != IntentType.KNOWLEDGE:
                await stream.publish(StreamEvent(event_type=EventType.USER_UTTERANCE, text=user_input))
                await stream.publish(StreamEvent(event_type=EventType.AGENT_RESPONSE, text=response))

            await memory_manager.add_turn("user", user_input, session)
            await memory_manager.add_turn("assistant", response, session)
            await session_logger.log_turn(
                session.session_id,
                session.turn_count,
                user_input,
                response,
                emotion=session.emotion.value,
                intent=intent.value,
            )

            if trace_enabled:
                trace_lines.extend(llm_trace.format_lines())
                _print_trace(trace_lines)
            print(f"\n系统: {response}")
    finally:
        await session_logger.log_session_end(session.session_id, session.turn_count)
        await router.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local business CLI without voice APIs")
    parser.add_argument("--docs", type=str, default="knowledge_base/", help="知识库目录路径")
    parser.add_argument("--model", type=str, default="llama3.2", help="本地 Ollama 模型")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="本地 embedding 模型")
    parser.add_argument("--embedding-dim", type=int, default=768, help="embedding 维度")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama 服务地址")
    parser.add_argument("--trace", action="store_true", help="打印模块决策轨迹")
    args = parser.parse_args()

    config = VORConfig(
        llm_provider="ollama",
        llm_model=args.model,
        llm_base_url=args.base_url,
        embedding_provider="ollama",
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dim,
        stt_provider="none",
        tts_provider="none",
    )
    docs_dir = Path(args.docs) if args.docs else None
    asyncio.run(run_cli(config, docs_dir, trace_enabled=args.trace))


if __name__ == "__main__":
    main()