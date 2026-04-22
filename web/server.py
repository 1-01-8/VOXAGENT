"""实时语音客服 WebSocket 服务端

架构：Browser Mic → WebSocket(PCM) → Cloud STT → Agent Pipeline → Cloud TTS → WebSocket(PCM) → Browser Speaker

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
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ─── 项目内部导入 ───
from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import ConversationStream, EventType, StreamEvent
from voice_optimized_rag.dialogue.business_scope import OUT_OF_SCOPE_RESPONSE, build_business_answer_prompt
from voice_optimized_rag.dialogue.domain_router import DomainRouter
from voice_optimized_rag.dialogue.session import SessionContext, EmotionState, IntentType
from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.emotion_detector import EmotionDetector
from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.task_state_machine import BusinessTaskStateMachine
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy
from voice_optimized_rag.agent.domain_agent import create_domain_agents
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.llm.base import LLMProvider
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
                api_key=(
                    config.siliconflow_api_key
                    if config.stt_provider == "siliconflow"
                    else config.llm_api_key
                ),
                sf_stt_model=config.siliconflow_stt_model,
                sf_base_url=config.siliconflow_base_url,
            )
            logger.info(f"STT loaded: {config.stt_provider}")
        except Exception as e:
            logger.warning(f"STT 不可用 ({type(e).__name__}: {e})，降级为文字输入")

    if config.tts_provider and config.tts_provider != "none":
        try:
            from voice_optimized_rag.voice.tts import create_tts
            tts = create_tts(
                config.tts_provider,
                api_key=(
                    config.siliconflow_api_key
                    if config.tts_provider == "siliconflow"
                    else config.llm_api_key
                ),
                sf_tts_model=config.siliconflow_tts_model,
                sf_tts_voice=config.siliconflow_tts_voice,
                sf_base_url=config.siliconflow_base_url,
                sf_tts_sample_rate=config.siliconflow_tts_sample_rate,
                sf_tts_format=config.siliconflow_tts_format,
                sf_tts_speed=config.siliconflow_tts_speed,
            )
            logger.info(f"TTS loaded: {config.tts_provider}")
            # 启动时预热：触发模型加载 + JIT 编译 + 音色缓存（首条请求不再等 30s）
            try:
                ensure_model = getattr(tts, "_ensure_model", None)
                if callable(ensure_model):
                    ensure_model()
            except Exception as e:
                logger.warning(f"TTS 预热失败: {e}")
        except Exception as e:
            logger.warning(f"TTS 不可用 ({type(e).__name__}: {e})，降级为文字输出")

    return stt, tts


class _EchoLLM(LLMProvider):
    """无 API Key 时的降级 LLM，回显用户输入 + 提示。"""
    async def generate(self, prompt: str, context: str = "") -> str:
        prompt_lower = prompt.lower()
        if "task" in prompt_lower or "knowledge" in prompt_lower or "out_of_scope" in prompt_lower:
            if any(keyword in prompt for keyword in ("退款", "订单", "地址", "取消")):
                return "task"
            if any(keyword in prompt for keyword in ("价格", "报价", "套餐", "发票", "功能")):
                return "knowledge"
            return "out_of_scope"
        if "压缩" in prompt or "摘要" in prompt:
            return "（摘要占位）"
        return f"[Echo] 收到您的业务请求。当前为演示模式，请配置 VOR_LLM_API_KEY 以启用真实对话。"

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
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
    domain_router = DomainRouter(llm)
    emotion_detector = EmotionDetector(stream)
    memory_manager = MemoryManager(llm, short_term_turns=config.memory_short_term_turns)
    transfer_policy = TransferPolicy(
        stream,
        angry_threshold=config.emotion_angry_threshold,
        max_agent_failures=config.transfer_max_failures,
    )
    guard = PermissionGuard(stream, confirm_timeout=15.0)
    domain_agents = create_domain_agents(
        llm=llm,
        permission_guard=guard,
        stream=stream,
        max_iterations=config.agent_max_iterations,
        tool_timeout=config.agent_tool_timeout,
        tool_retry=config.agent_tool_retry,
    )
    task_state_machine = BusinessTaskStateMachine(guard)
    auto_qa = AutoQA()
    session_logger = SessionLogger(config.session_log_dir)

    _components.update(
        config=config, llm=llm, stt=stt, tts=tts, stream=stream,
        intent_router=intent_router, domain_router=domain_router,
        emotion_detector=emotion_detector,
        memory_manager=memory_manager, transfer_policy=transfer_policy,
        domain_agents=domain_agents, task_state_machine=task_state_machine,
        auto_qa=auto_qa, session_logger=session_logger,
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
            intent = await intent_router.classify(user_text, session, per_session_memory.get_context())
            session.current_intent = intent
            if intent != IntentType.OUT_OF_SCOPE:
                domain_router = _components["domain_router"]
                await domain_router.classify(
                    user_text,
                    session,
                    intent=intent,
                    conversation_text=per_session_memory.get_context(),
                )

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

            # ── 6. 生成回复（LLM→TTS 全流水线）──
            domain_agents = _components["domain_agents"]
            tts = _components.get("tts")
            llm = _components["llm"]

            # 流水线：LLM token → 文本缓冲 → 片段送 TTS → 音频块推给前端
            # 触发 flush 的条件：遇到标点 或 缓冲达到阈值
            import re, base64
            FLUSH_PUNCT = set("，。！？；：,.!?;:\n")
            MIN_FLUSH_CHARS = 6  # 最少多少字才送 TTS（太短音色不稳）

            reply_parts: list[str] = []
            tts_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

            async def _tts_consumer():
                """从 tts_queue 取文本片段，合成并发送 audio_chunk。

                若 tts 支持 `synthesize_http_stream`（如 SiliconFlowTTS），
                按 HTTP chunked 模式边接收边下推，TTFB < 1s；
                否则退回整段 synthesize。
                """
                seg_idx = 0
                t_first_audio = None
                supports_http_stream = hasattr(tts, "synthesize_http_stream")
                # 采样率用于 RTF 计算（SF 默认 24kHz，本地 CosyVoice 也 24kHz）
                sr = getattr(tts, "sample_rate", 24000) if tts else 24000
                while True:
                    seg = await tts_queue.get()
                    if seg is None:
                        await ws.send_json({"type": "audio_end"})
                        return
                    if not tts:
                        continue
                    t_s = time.perf_counter()
                    try:
                        if supports_http_stream:
                            total = 0
                            first_chunk_t = None
                            async for pcm_chunk in tts.synthesize_http_stream(seg):
                                if first_chunk_t is None:
                                    first_chunk_t = time.perf_counter()
                                    if t_first_audio is None:
                                        t_first_audio = first_chunk_t
                                        logger.info(
                                            f"[PIPE-TIMER] FIRST_AUDIO from_t0="
                                            f"{(t_first_audio-t0)*1000:.0f}ms "
                                            f"seg0_chars={len(seg)} "
                                            f"seg0_ttfb={(first_chunk_t-t_s)*1000:.0f}ms"
                                        )
                                total += len(pcm_chunk)
                                await ws.send_json({
                                    "type": "audio_chunk",
                                    "audio": base64.b64encode(pcm_chunk).decode(),
                                    "sample_rate": sr,
                                })
                            wall_ms = (time.perf_counter() - t_s) * 1000
                            audio_ms = (total // 2) * 1000.0 / sr if total else 0
                            logger.info(
                                f"[PIPE-TIMER] seg#{seg_idx} chars={len(seg)} "
                                f"tts_ms={wall_ms:.0f} audio_ms={audio_ms:.0f} "
                                f"RTF={wall_ms/audio_ms if audio_ms else 0:.2f} (stream)"
                            )
                        else:
                            pcm = await tts.synthesize(seg)
                            wall_ms = (time.perf_counter() - t_s) * 1000
                            audio_ms = (len(pcm) // 2) * 1000.0 / sr if pcm else 0
                            if t_first_audio is None:
                                t_first_audio = time.perf_counter()
                                logger.info(
                                    f"[PIPE-TIMER] FIRST_AUDIO from_t0="
                                    f"{(t_first_audio-t0)*1000:.0f}ms "
                                    f"seg0_chars={len(seg)} seg0_wall={wall_ms:.0f}ms"
                                )
                            logger.info(
                                f"[PIPE-TIMER] seg#{seg_idx} chars={len(seg)} "
                                f"tts_ms={wall_ms:.0f} audio_ms={audio_ms:.0f} "
                                f"RTF={wall_ms/audio_ms if audio_ms else 0:.2f}"
                            )
                            await ws.send_json({
                                "type": "audio_chunk",
                                "audio": base64.b64encode(pcm).decode(),
                                "sample_rate": sr,
                            })
                    except Exception as e:
                        logger.warning(f"TTS seg#{seg_idx} 失败: {e}")
                    seg_idx += 1

            consumer_task = asyncio.create_task(_tts_consumer()) if tts else None

            async def _produce_from_stream(prompt_text: str, context: str = ""):
                """从 LLM 流读 token，按标点/长度切片喂给 tts_queue。"""
                buf = ""
                t_llm0 = time.perf_counter()
                t_first_tok = None
                total_chars = 0
                async for tok in llm.stream(prompt_text, context=context):
                    if t_first_tok is None:
                        t_first_tok = time.perf_counter()
                        logger.info(
                            f"[PIPE-TIMER] LLM_FIRST_TOKEN +{(t_first_tok-t0)*1000:.0f}ms"
                        )
                    buf += tok
                    total_chars += len(tok)
                    # 找最后一个标点，切出可发射的前段
                    while True:
                        idx = -1
                        for i, ch in enumerate(buf):
                            if ch in FLUSH_PUNCT:
                                idx = i
                        if idx >= 0 and idx + 1 >= MIN_FLUSH_CHARS:
                            seg, buf = buf[:idx + 1], buf[idx + 1:]
                            seg = seg.strip()
                            if seg:
                                reply_parts.append(seg)
                                if tts:
                                    await tts_queue.put(seg)
                            break  # 本轮只切一段，继续收下一 token
                        else:
                            break
                # 残余缓冲
                if buf.strip():
                    reply_parts.append(buf.strip())
                    if tts:
                        await tts_queue.put(buf.strip())
                logger.info(
                    f"[PIPE-TIMER] LLM_DONE wall={(time.perf_counter()-t_llm0)*1000:.0f}ms "
                    f"chars={total_chars} segs={len(reply_parts)}"
                )

            t_gen0 = time.perf_counter()
            if intent == IntentType.TASK:
                await ws.send_json({"type": "status", "message": "正在处理您的请求..."})
                task_result = await _components["task_state_machine"].handle(user_text, session)
                if task_result.handled:
                    reply_text = task_result.reply_text
                else:
                    reply_text = await domain_agents[session.current_domain].execute(
                        user_text,
                        session,
                        per_session_memory.get_context(),
                    )
                reply_parts.append(reply_text)
                if tts:
                    await tts_queue.put(reply_text)
            elif intent == IntentType.KNOWLEDGE:
                context = per_session_memory.get_context()
                await _produce_from_stream(
                    build_business_answer_prompt(user_text),
                    context=context,
                )
            else:
                reply_text = OUT_OF_SCOPE_RESPONSE
                reply_parts.append(reply_text)
                if tts:
                    await tts_queue.put(reply_text)

            # 关 TTS 队列，等消费者清空
            if tts:
                await tts_queue.put(None)

            reply_text = "".join(reply_parts)
            logger.info(
                f"[PIPE-TIMER] intent={intent.value} gen_ms={(time.perf_counter()-t_gen0)*1000:.0f} "
                f"reply_len={len(reply_text)}"
            )

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

            # ── 9. 文字立即发回（不等 TTS）──
            elapsed_ms = (time.perf_counter() - t0) * 1000
            await ws.send_json({
                "type": "reply",
                "text": reply_text,
                "emotion": session.emotion.value,
                "intent": intent.value,
                "audio": None,
                "transferred": False,
                "timing_ms": round(elapsed_ms, 1),
            })

            # ── 10. 等待 TTS 消费者清空队列并发送 audio_end ──
            # 注意：TTS 音频已经在 _produce_from_stream / TASK 分支里通过 tts_queue
            # 交给 _tts_consumer 边合成边下推了，这里不再重复合成 reply_text，
            # 否则前端会同时收到两套 PCM，产生 "多重声效 / 回声" 的听感。
            if consumer_task is not None:
                try:
                    await consumer_task
                except Exception as e:
                    logger.warning(f"TTS consumer 异常: {e}")

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

    ⚠️ [DEPRECATED · v1.0.0+] 本端点依赖本地 `SenseVoiceSTT`（FunASR），v1.0.0 起
    项目默认 STT provider 已切换为 SiliconFlow（云端非流式）。当前配置下本端点会直接
    返回 "流式 STT 不可用" 并关闭连接，主链路请使用 `/ws/chat`。
    保留本端点是为了 VOR_STT_PROVIDER=sensevoice 回滚场景与将来的 SF 流式 ASR 接入。

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
    intent = await intent_router.classify(user_text, session, memory.get_context())
    session.current_intent = intent
    if intent != IntentType.OUT_OF_SCOPE:
        domain_router = _components["domain_router"]
        await domain_router.classify(
            user_text,
            session,
            intent=intent,
            conversation_text=memory.get_context(),
        )

    # 转人工
    transfer_policy = _components["transfer_policy"]
    should_transfer = await transfer_policy.evaluate(session, user_text)
    if should_transfer:
        return f"非常抱歉给您带来不便，正在为您转接人工客服。原因：{session.transfer_reason}"

    # 记忆
    await memory.add_turn("user", user_text, session)

    # 生成回复
    domain_agents = _components["domain_agents"]
    if intent == IntentType.TASK:
        task_result = await _components["task_state_machine"].handle(user_text, session)
        if task_result.handled:
            reply_text = task_result.reply_text
        else:
            reply_text = await domain_agents[session.current_domain].execute(
                user_text,
                session,
                memory.get_context(),
            )
    elif intent == IntentType.KNOWLEDGE:
        context = memory.get_context()
        llm = _components["llm"]
        reply_text = await llm.generate(
            build_business_answer_prompt(user_text),
            context=context,
        )
    else:
        reply_text = OUT_OF_SCOPE_RESPONSE

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
