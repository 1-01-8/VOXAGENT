"""
业务客服 Demo 入口 —— 纯文本业务链路

整合所有模块的业务对话链路：
    意图路由 → 域路由 → RAG/Function Calling Agent → 质检

用法：
  python examples/customer_service_demo.py --docs knowledge_base/

支持模式：
  --text     纯文本模式（默认，无需音频设备）
  --voice    语音模式（需要麦克风和扬声器）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import EventType, StreamEvent
from voice_optimized_rag.core.memory_router import MemoryRouter
from voice_optimized_rag.dialogue.business_scope import OUT_OF_SCOPE_RESPONSE
from voice_optimized_rag.dialogue.session import SessionContext, IntentType
from voice_optimized_rag.dialogue.domain_router import DomainRouter
from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.task_state_machine import BusinessTaskStateMachine
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy
from voice_optimized_rag.agent.domain_agent import create_domain_agents
from voice_optimized_rag.agent.permission_guard import TextPermissionGuard
from voice_optimized_rag.utils.session_logger import SessionLogger
from voice_optimized_rag.utils.auto_qa import AutoQA
from voice_optimized_rag.llm.base import create_llm


async def run_text_mode(config: VORConfig, docs_dir: Path | None) -> None:
    """纯文本交互模式"""
    # 初始化 MemoryRouter（含 Slow Thinker / Fast Talker）
    router = MemoryRouter(config)
    await router.start()

    # 导入知识库
    if docs_dir and docs_dir.exists():
        count = await router.ingest_directory(docs_dir)
        print(f"📚 知识库已就绪: 当前索引 {router.document_count} 个文档块，本次新增 {count} 个")

    # 初始化对话管理模块
    llm = create_llm(config)
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

    # 初始化 Agent
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

    # 初始化质检和日志
    auto_qa = AutoQA()
    session_logger = SessionLogger(config.session_log_dir)

    print("=" * 60)
    print("📌 业务客服系统 (文本模式)")
    print("=" * 60)
    print("输入 'quit' 退出 | 输入 'stats' 查看统计")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(f"\n📊 缓存命中率: {router.metrics.cache_hit_rate:.1%}")
            print(f"📊 对话轮次: {session.turn_count}")
            print(f"📊 情绪状态: {session.emotion.value}")
            continue

        session.increment_turn()

        # Step 1: 意图路由
        memory_ctx = memory_manager.get_context()
        conversation_text = memory_ctx or stream.get_conversation_text(max_turns=6)
        intent = await intent_router.classify(user_input, session, conversation_text)
        session.current_intent = intent
        if intent != IntentType.OUT_OF_SCOPE:
            await domain_router.classify(
                user_input,
                session,
                intent=intent,
                conversation_text=conversation_text,
            )

        # Step 2: 转人工检查
        should_transfer = await transfer_policy.evaluate(session, user_input)
        if should_transfer:
            print(f"\n🔄 [系统] 正在转接人工客服...")
            print(f"   原因: {session.transfer_reason}")
            await session_logger.log_transfer(
                session.session_id, session.transfer_reason
            )
            break

        # Step 3: 根据意图路由处理
        if intent == IntentType.TASK:
            print(f"   [意图: 任务执行 → {session.current_domain.value} Agent]")
            task_result = await task_state_machine.handle(user_input, session)
            if task_result.handled:
                response = task_result.reply_text
            else:
                response = await domain_agents[session.current_domain].execute(
                    user_input,
                    session,
                    memory_ctx,
                )
        elif intent == IntentType.KNOWLEDGE:
            print(f"   [意图: 知识咨询 → RAG，业务域: {session.current_domain.value}]")
            response = await router.query(user_input)
        else:
            print("   [意图: 超范围 → 业务引导]")
            response = OUT_OF_SCOPE_RESPONSE

        # Step 4: 质检
        qa_result = await auto_qa.check(response)
        if not qa_result.passed:
            print(f"   ⚠️  质检: {qa_result.issues}")
            response = qa_result.cleaned_response

        if intent != IntentType.KNOWLEDGE:
            await stream.publish(StreamEvent(event_type=EventType.USER_UTTERANCE, text=user_input))
            await stream.publish(StreamEvent(event_type=EventType.AGENT_RESPONSE, text=response))

        # Step 5: 记录对话
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

        print(f"\n🤖 客服: {response}")

    # 结束
    await session_logger.log_session_end(
        session.session_id, session.turn_count
    )
    await router.stop()
    print("\n👋 会话结束。")
    print(f"📊 总轮次: {session.turn_count} | 缓存命中率: {router.metrics.cache_hit_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="纯业务客服 Demo")
    parser.add_argument("--docs", type=str, default="knowledge_base/",
                        help="知识库目录路径")
    parser.add_argument("--provider", type=str, default=None,
                        help="LLM 提供商 (openai/ollama/gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM 模型名称")
    args = parser.parse_args()

    kwargs = {}
    if args.provider:
        kwargs["llm_provider"] = args.provider
    if args.model:
        kwargs["llm_model"] = args.model

    config = VORConfig(**kwargs) if kwargs else VORConfig()
    docs_dir = Path(args.docs) if args.docs else None

    asyncio.run(run_text_mode(config, docs_dir))


if __name__ == "__main__":
    main()
