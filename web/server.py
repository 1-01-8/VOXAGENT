"""实时语音客服 WebSocket 服务端

架构：Browser Mic → WebSocket(PCM) → SenseVoice STT → Agent Pipeline → CosyVoice TTS → WebSocket(PCM) → Browser Speaker

在语音模型未部署时自动降级为 "文字模拟模式"：
- STT 由前端文字输入替代
- TTS 由后端返回文字（前端用 Web Speech API 朗读）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ─── 项目内部导入 ───
from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import ConversationStream, EventType, StreamEvent
from voice_optimized_rag.dialogue.session import SessionContext, EmotionState, IntentType
from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.emotion_detector import EmotionDetector
from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy
from voice_optimized_rag.agent.react_agent import ReactAgent
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.tools.query_tools import QueryOrderTool, QueryInventoryTool, GetCustomerInfoTool, CheckPromotionTool
from voice_optimized_rag.agent.tools.write_tools import UpdateAddressTool, CancelOrderTool
from voice_optimized_rag.agent.tools.finance_tools import ApplyRefundTool
from voice_optimized_rag.utils.auto_qa import AutoQA
from voice_optimized_rag.utils.session_logger import SessionLogger

logger = logging.getLogger("voice_web")

# ─── 全局组件（lifespan 中初始化） ───
_components: dict = {}


def _try_load_llm(config: VORConfig):
    """尝试加载 LLM，失败时返回一个简单的 Echo LLM。"""
    try:
        from voice_optimized_rag.llm.base import create_llm
        return create_llm(config)
    except Exception as e:
        logger.warning(f"LLM 加载失败 ({type(e).__name__}: {e})，使用 Echo 模式")
        return _EchoLLM()


def _try_load_voice(config: VORConfig):
    """尝试加载语音模型，失败时返回 None（降级到文字模式）。"""
    stt, tts = None, None

    if config.stt_provider and config.stt_provider != "none":
        try:
            from voice_optimized_rag.voice.stt import create_stt
            stt = create_stt(
                config.stt_provider,
                model_id=config.sensevoice_model,
                device=config.sensevoice_device,
                model_size=config.whisper_model,
                api_key=config.llm_api_key,
            )
            logger.info(f"STT loaded: {config.stt_provider}")
        except Exception as e:
            logger.warning(f"STT 不可用 ({type(e).__name__}: {e})，降级为文字输入")

    if config.tts_provider and config.tts_provider != "none":
        try:
            from voice_optimized_rag.voice.tts import create_tts
            tts = create_tts(
                config.tts_provider,
                model_id=config.cosyvoice_model,
                device=config.cosyvoice_device,
                default_speaker=config.cosyvoice_default_speaker,
                api_key=config.llm_api_key,
            )
            logger.info(f"TTS loaded: {config.tts_provider}")
        except Exception as e:
            logger.warning(f"TTS 不可用 ({type(e).__name__}: {e})，降级为文字输出")

    return stt, tts


class _EchoLLM:
    """无 API Key 时的降级 LLM，回显用户输入 + 提示。"""
    async def generate(self, prompt: str, context: str = "") -> str:
        if "task" in prompt.lower() or "knowledge" in prompt.lower() or "chitchat" in prompt.lower():
            return "knowledge"
        if "压缩" in prompt or "摘要" in prompt:
            return "（摘要占位）"
        return f"[Echo] 收到您的消息。当前为演示模式，请配置 VOR_LLM_API_KEY 以启用真实对话。"

    async def stream(self, prompt: str, context: str = ""):
        resp = await self.generate(prompt, context)
        for word in resp.split():
            yield word + " "


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化所有组件。"""
    config = VORConfig()
    llm = _try_load_llm(config)
    stt, tts = _try_load_voice(config)

    stream = ConversationStream(window_size=config.conversation_window_size)
    intent_router = IntentRouter(llm)
    emotion_detector = EmotionDetector(stream)
    memory_manager = MemoryManager(llm, short_term_turns=config.memory_short_term_turns)
    transfer_policy = TransferPolicy(
        stream,
        angry_threshold=config.emotion_angry_threshold,
        max_agent_failures=config.transfer_max_failures,
    )
    guard = PermissionGuard(stream, confirm_timeout=15.0)
    tools = [
        QueryOrderTool(), QueryInventoryTool(), GetCustomerInfoTool(),
        CheckPromotionTool(), UpdateAddressTool(), CancelOrderTool(),
        ApplyRefundTool(),
    ]
    agent = ReactAgent(
        llm=llm, tools=tools, permission_guard=guard, stream=stream,
        max_iterations=config.agent_max_iterations,
        tool_timeout=config.agent_tool_timeout,
        tool_retry=config.agent_tool_retry,
    )
    auto_qa = AutoQA()
    session_logger = SessionLogger(config.session_log_dir)

    _components.update(
        config=config, llm=llm, stt=stt, tts=tts, stream=stream,
        intent_router=intent_router, emotion_detector=emotion_detector,
        memory_manager=memory_manager, transfer_policy=transfer_policy,
        agent=agent, auto_qa=auto_qa, session_logger=session_logger,
        guard=guard,
    )

    voice_mode = "语音模式" if (stt and tts) else "文字模式"
    llm_mode = "LLM 已连接" if not isinstance(llm, _EchoLLM) else "Echo 演示模式"
    logger.info(f"✅ 服务已启动 | {voice_mode} | {llm_mode}")

    yield

    _components.clear()
    logger.info("🛑 服务已关闭")


# ─── FastAPI App ───

app = FastAPI(title="VoiceAgentRAG 实时语音客服", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "voice_mode": _components.get("stt") is not None and _components.get("tts") is not None,
        "llm_mode": "echo" if isinstance(_components.get("llm"), _EchoLLM) else "real",
    }


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """主 WebSocket 端点：处理文字 / 语音交互。

    协议（JSON 消息）：
    → 客户端发送: {"type": "text",  "text": "..."}
    → 客户端发送: {"type": "audio", "data": "<base64 PCM>", "sample_rate": 16000}
    ← 服务端回复: {"type": "reply", "text": "...", "emotion": "...", "intent": "...",
                    "audio": "<base64 PCM>"|null, "transferred": false, "timing_ms": 123}
    ← 服务端回复: {"type": "status", "message": "..."}
    ← 服务端回复: {"type": "confirm", "tool_name": "...", "description": "..."}
    → 客户端回复: {"type": "confirm_response", "confirmed": true/false}
    """
    await ws.accept()
    session = SessionContext()
    session_id = session.session_id

    # 订阅确认事件
    confirm_sub = None

    await ws.send_json({
        "type": "status",
        "message": f"连接成功 | 会话 {session_id[:8]}",
        "session_id": session_id,
        "voice_mode": _components.get("stt") is not None,
    })

    # 每个 WebSocket 会话独立的记忆管理器（避免跨会话记忆混淆）
    per_session_memory = MemoryManager(
        _components["llm"],
        short_term_turns=_components["config"].memory_short_term_turns,
    )

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "text")
            t0 = time.perf_counter()

            # ── 1. 获取用户文本 ──
            user_text = ""
            detected_emotion = "neutral"

            if msg_type == "audio" and _components.get("stt"):
                import base64
                audio_bytes = base64.b64decode(msg["data"])
                stt = _components["stt"]
                # 使用 transcribe_with_emotion 获取完整结果（含情绪标签）
                if hasattr(stt, "transcribe_with_emotion"):
                    result = await stt.transcribe_with_emotion(audio_bytes)
                    user_text = result.text
                    detected_emotion = result.emotion
                else:
                    raw = await stt.transcribe(audio_bytes)
                    user_text = raw if isinstance(raw, str) else str(raw)
                await ws.send_json({"type": "transcript", "text": user_text, "emotion": detected_emotion})
            else:
                user_text = msg.get("text", "").strip()

            if not user_text:
                continue

            session.increment_turn()

            # ── 2. 情绪检测 ──
            emotion_detector = _components["emotion_detector"]
            emotion = await emotion_detector.update(detected_emotion, session)

            # ── 3. 意图路由 ──
            intent_router = _components["intent_router"]
            intent = await intent_router.classify(user_text, session)

            # ── 4. 转人工检查 ──
            transfer_policy = _components["transfer_policy"]
            should_transfer = await transfer_policy.evaluate(session, user_text)

            if should_transfer:
                reply_text = f"非常抱歉给您带来不便，正在为您转接人工客服。原因：{session.transfer_reason}"
                elapsed_ms = (time.perf_counter() - t0) * 1000
                await ws.send_json({
                    "type": "reply",
                    "text": reply_text,
                    "emotion": session.emotion.value,
                    "intent": "transfer",
                    "audio": None,
                    "transferred": True,
                    "timing_ms": round(elapsed_ms, 1),
                })
                await _components["session_logger"].log_transfer(session_id, session.transfer_reason, user_text)
                continue

            # ── 5. 记忆管理 ──
            await per_session_memory.add_turn("user", user_text, session)

            # ── 6. 生成回复 ──
            agent = _components["agent"]
            reply_text = ""

            if intent == IntentType.TASK:
                await ws.send_json({"type": "status", "message": "正在处理您的请求..."})
                reply_text = await agent.execute(user_text, session)
            elif intent == IntentType.KNOWLEDGE:
                context = per_session_memory.get_context()
                llm = _components["llm"]
                reply_text = await llm.generate(
                    f"基于上下文回答用户问题：\n用户：{user_text}",
                    context=context,
                )
            else:
                llm = _components["llm"]
                reply_text = await llm.generate(f"友好地回复用户的闲聊：{user_text}")

            # ── 7. 质检 ──
            auto_qa = _components["auto_qa"]
            qa_result = await auto_qa.check(reply_text)
            if not qa_result.passed:
                reply_text = qa_result.cleaned_response

            # ── 8. 记忆 + 日志 ──
            await per_session_memory.add_turn("assistant", reply_text, session)
            await _components["session_logger"].log_turn(
                session_id=session_id,
                turn_index=session.turn_count,
                user_text=user_text,
                agent_text=reply_text,
                emotion=session.emotion.value,
                intent=intent.value,
            )

            # ── 9. TTS（若可用）──
            audio_b64 = None
            if _components.get("tts"):
                try:
                    tts = _components["tts"]
                    audio_bytes = await tts.synthesize(reply_text)
                    import base64
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                except Exception as e:
                    logger.warning(f"TTS 失败: {e}")

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── 10. 发送回复 ──
            await ws.send_json({
                "type": "reply",
                "text": reply_text,
                "emotion": session.emotion.value,
                "intent": intent.value,
                "audio": audio_b64,
                "transferred": False,
                "timing_ms": round(elapsed_ms, 1),
            })

    except WebSocketDisconnect:
        logger.info(f"会话 {session_id[:8]} 断开")
        await _components["session_logger"].log_session_end(
            session_id, total_turns=session.turn_count,
        )
    except Exception as e:
        logger.exception(f"WebSocket 异常: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """流式语音 WebSocket 端点：逐 chunk 接收音频，实时返回 partial/final 转写。

    协议（二进制 + JSON 混合）：
    → 客户端发送: binary frame（PCM int16, 16kHz, mono, 100ms/chunk）
    → 客户端发送: {"type": "end_stream"} — 标记本轮录音结束
    ← 服务端回复: {"type": "partial", "text": "...", "emotion": "..."} — 中间结果
    ← 服务端回复: {"type": "final", "text": "...", "emotion": "..."} — 端点确认
    ← 服务端回复: {"type": "reply", ...} — Agent pipeline 回复（同 /ws/chat 格式）
    """
    await ws.accept()

    from voice_optimized_rag.voice.sensevoice_stt import StreamingSenseVoiceSTT, SenseVoiceSTT

    stt = _components.get("stt")
    if not stt or not isinstance(stt, SenseVoiceSTT):
        await ws.send_json({"type": "error", "message": "流式 STT 不可用，需要 SenseVoice 模型"})
        await ws.close()
        return

    streaming_stt = StreamingSenseVoiceSTT(stt)
    streaming_stt.reset()

    session = SessionContext()
    session_id = session.session_id
    per_session_memory = MemoryManager(
        _components["llm"],
        short_term_turns=_components["config"].memory_short_term_turns,
    )

    await ws.send_json({
        "type": "status",
        "message": f"流式连接就绪 | 会话 {session_id[:8]}",
        "session_id": session_id,
    })

    try:
        while True:
            message = await ws.receive()

            # 二进制帧 = 音频 chunk
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                result = await streaming_stt.feed_chunk(chunk_bytes)

                if result:
                    msg_type = "final" if result.is_final else "partial"
                    await ws.send_json({
                        "type": msg_type,
                        "text": result.text,
                        "emotion": result.emotion,
                    })

                    # Final 结果触发 pipeline 处理
                    if result.is_final and result.text.strip():
                        t0 = time.perf_counter()
                        reply_text = await _process_pipeline(
                            result.text, result.emotion, session, per_session_memory
                        )
                        elapsed_ms = (time.perf_counter() - t0) * 1000

                        # TTS
                        audio_b64 = None
                        if _components.get("tts"):
                            try:
                                import base64
                                audio_bytes = await _components["tts"].synthesize(reply_text)
                                audio_b64 = base64.b64encode(audio_bytes).decode()
                            except Exception as e:
                                logger.warning(f"TTS 失败: {e}")

                        await ws.send_json({
                            "type": "reply",
                            "text": reply_text,
                            "emotion": session.emotion.value,
                            "intent": "task",
                            "audio": audio_b64,
                            "transferred": False,
                            "timing_ms": round(elapsed_ms, 1),
                        })

                        # 重置流式状态，准备下一句
                        streaming_stt.reset()

            # JSON 文本帧 = 控制消息
            elif "text" in message and message["text"]:
                msg = json.loads(message["text"])
                if msg.get("type") == "end_stream":
                    # 强制 flush 残余 buffer
                    streaming_stt.reset()
                    await ws.send_json({"type": "status", "message": "stream reset"})
                elif msg.get("type") == "close":
                    break

    except WebSocketDisconnect:
        logger.info(f"流式会话 {session_id[:8]} 断开")
    except Exception as e:
        logger.exception(f"流式 WebSocket 异常: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _process_pipeline(
    user_text: str, detected_emotion: str, session: SessionContext, memory: MemoryManager
) -> str:
    """共享的 Agent pipeline 处理逻辑（供 /ws/chat 和 /ws/stream 复用）"""
    session.increment_turn()

    # 情绪
    emotion_detector = _components["emotion_detector"]
    await emotion_detector.update(detected_emotion, session)

    # 意图
    intent_router = _components["intent_router"]
    intent = await intent_router.classify(user_text, session)

    # 转人工
    transfer_policy = _components["transfer_policy"]
    should_transfer = await transfer_policy.evaluate(session, user_text)
    if should_transfer:
        return f"非常抱歉给您带来不便，正在为您转接人工客服。原因：{session.transfer_reason}"

    # 记忆
    await memory.add_turn("user", user_text, session)

    # 生成回复
    agent = _components["agent"]
    from voice_optimized_rag.dialogue.session import IntentType
    if intent == IntentType.TASK:
        reply_text = await agent.execute(user_text, session)
    elif intent == IntentType.KNOWLEDGE:
        context = memory.get_context()
        llm = _components["llm"]
        reply_text = await llm.generate(
            f"基于上下文回答用户问题：\n用户：{user_text}", context=context
        )
    else:
        llm = _components["llm"]
        reply_text = await llm.generate(f"友好地回复用户的闲聊：{user_text}")

    # 质检
    auto_qa = _components["auto_qa"]
    qa_result = await auto_qa.check(reply_text)
    if not qa_result.passed:
        reply_text = qa_result.cleaned_response

    # 日志
    await memory.add_turn("assistant", reply_text, session)
    await _components["session_logger"].log_turn(
        session_id=session.session_id,
        turn_index=session.turn_count,
        user_text=user_text,
        agent_text=reply_text,
        emotion=session.emotion.value,
        intent=intent.value,
    )

    return reply_text


# ─── 确认事件处理 ───

@app.websocket("/ws/confirm")
async def websocket_confirm(ws: WebSocket):
    """处理权限确认的独立 WebSocket 通道。"""
    await ws.accept()
    guard = _components.get("guard")
    if not guard:
        await ws.close(reason="Guard not initialized")
        return

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "confirm_response":
                await guard.handle_confirm_response(msg.get("confirmed", False))
    except WebSocketDisconnect:
        pass


def main():
    """启动服务入口。"""
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 启动 VoiceAgentRAG Web 服务 @ http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
