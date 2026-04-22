"""Business MCP server for cross-system interoperability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voice_optimized_rag.agent.domain_agent import create_domain_agents
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.skill_registry import BUSINESS_SKILL_REGISTRY, SkillRegistry
from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.core.memory_router import MemoryRouter
from voice_optimized_rag.dialogue.business_scope import OUT_OF_SCOPE_RESPONSE
from voice_optimized_rag.dialogue.domain_router import DomainRouter
from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.session import IntentType, SessionContext
from voice_optimized_rag.dialogue.task_state_machine import BusinessTaskStateMachine
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy
from voice_optimized_rag.llm.base import create_llm


def serialize_skill_catalog(registry: SkillRegistry = BUSINESS_SKILL_REGISTRY) -> list[dict[str, Any]]:
    """Serialize the business skill registry for MCP resources and tests."""
    catalog: list[dict[str, Any]] = []
    for domain in sorted(registry.domains, key=lambda item: item.value):
        spec = registry.get(domain)
        catalog.append({
            "skill_id": spec.skill_id,
            "domain": spec.domain.value,
            "agent_name": spec.agent_name,
            "responsibility": spec.responsibility,
            "scope_rules": list(spec.scope_rules),
            "tools": [factory().name for factory in spec.tool_factories],
        })
    return catalog


def build_mcp_server(config: VORConfig | None = None, docs_dir: str | None = None):
    """Build a FastMCP server exposing the business capabilities."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise ImportError(
            "Install MCP support first: pip install voice-optimized-rag[mcp]"
        ) from error

    resolved_config = config or VORConfig(stt_provider="none", tts_provider="none")
    llm = create_llm(resolved_config)
    stream = ConversationStream(window_size=resolved_config.conversation_window_size)
    intent_router = IntentRouter(llm)
    domain_router = DomainRouter(llm)
    memory_router = MemoryRouter(resolved_config, llm=llm)
    guard = PermissionGuard(stream)
    domain_agents = create_domain_agents(
        llm=llm,
        permission_guard=guard,
        stream=stream,
        max_iterations=resolved_config.agent_max_iterations,
        tool_timeout=resolved_config.agent_tool_timeout,
        tool_retry=resolved_config.agent_tool_retry,
    )
    task_state_machine = BusinessTaskStateMachine(guard)

    state = {"router_started": False, "docs_loaded": False}
    docs_path = Path(docs_dir) if docs_dir else None

    async def ensure_router_ready() -> None:
        if not state["router_started"]:
            await memory_router.start()
            state["router_started"] = True
        if docs_path and docs_path.exists() and not state["docs_loaded"]:
            await memory_router.ingest_directory(docs_path)
            state["docs_loaded"] = True

    mcp = FastMCP("voxcare-business")

    @mcp.resource("business://skills")
    def business_skills_resource() -> list[dict[str, Any]]:
        return serialize_skill_catalog()

    @mcp.resource("business://config")
    def business_runtime_resource() -> dict[str, Any]:
        return {
            "llm_provider": resolved_config.llm_provider,
            "llm_model": resolved_config.llm_model,
            "embedding_provider": resolved_config.embedding_provider,
            "stt_provider": resolved_config.stt_provider,
            "tts_provider": resolved_config.tts_provider,
            "docs_dir": str(docs_path) if docs_path else "",
        }

    @mcp.tool()
    async def route_business_request(user_request: str) -> dict[str, Any]:
        session = SessionContext()
        intent = await intent_router.classify(user_request, session)
        result: dict[str, Any] = {
            "intent": intent.value,
            "transfer_requested": session.transfer_requested,
            "transfer_reason": session.transfer_reason,
            "domain": "",
        }
        if intent != IntentType.OUT_OF_SCOPE:
            domain = await domain_router.classify(user_request, session, intent=intent)
            result["domain"] = domain.value
        return result

    @mcp.tool()
    async def query_business_knowledge(user_request: str) -> dict[str, Any]:
        await ensure_router_ready()
        answer = await memory_router.query(user_request)
        return {"intent": IntentType.KNOWLEDGE.value, "answer": answer}

    @mcp.tool()
    async def execute_business_task(user_request: str, domain: str = "") -> dict[str, Any]:
        session = SessionContext()
        intent = await intent_router.classify(user_request, session)
        if intent == IntentType.OUT_OF_SCOPE:
            return {"intent": intent.value, "answer": OUT_OF_SCOPE_RESPONSE}

        resolved_domain = await domain_router.classify(user_request, session, intent=intent)
        if domain:
            for candidate in BUSINESS_SKILL_REGISTRY.domains:
                if candidate.value == domain:
                    resolved_domain = candidate
                    session.current_domain = candidate
                    break

        memory = MemoryManager(llm, short_term_turns=resolved_config.memory_short_term_turns)
        transfer_policy = TransferPolicy(
            stream,
            angry_threshold=resolved_config.emotion_angry_threshold,
            max_agent_failures=resolved_config.transfer_max_failures,
        )
        if await transfer_policy.evaluate(session, user_request):
            return {
                "intent": intent.value,
                "domain": resolved_domain.value,
                "transfer_requested": True,
                "answer": f"正在转接人工客服。原因：{session.transfer_reason}",
            }

        task_result = await task_state_machine.handle(user_request, session)
        if task_result.handled:
            answer = task_result.reply_text
        else:
            answer = await domain_agents[resolved_domain].execute(
                user_request,
                session,
                memory.get_context(),
            )
        return {
            "intent": intent.value,
            "domain": resolved_domain.value,
            "transfer_requested": False,
            "answer": answer,
        }

    @mcp.tool()
    async def handle_business_request(user_request: str) -> dict[str, Any]:
        session = SessionContext()
        intent = await intent_router.classify(user_request, session)

        if intent == IntentType.OUT_OF_SCOPE:
            return {"intent": intent.value, "answer": OUT_OF_SCOPE_RESPONSE}

        domain = await domain_router.classify(user_request, session, intent=intent)
        if intent == IntentType.KNOWLEDGE:
            await ensure_router_ready()
            answer = await memory_router.query(user_request)
        else:
            memory = MemoryManager(llm, short_term_turns=resolved_config.memory_short_term_turns)
            task_result = await task_state_machine.handle(user_request, session)
            if task_result.handled:
                answer = task_result.reply_text
            else:
                answer = await domain_agents[domain].execute(user_request, session, memory.get_context())

        return {
            "intent": intent.value,
            "domain": domain.value,
            "answer": answer,
        }

    @mcp.prompt()
    def business_triage_prompt(user_request: str) -> str:
        return (
            "请把下面的请求整理成业务工单输入，必须聚焦销售、售后、财务三类业务。\n"
            f"用户请求：{user_request}"
        )

    return mcp