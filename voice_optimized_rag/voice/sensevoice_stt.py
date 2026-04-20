"""
SenseVoice STT 封装 —— 语音识别 + 情绪标签一体化

基于阿里达摩院 SenseVoice-Small 模型，单次推理同时输出：
- 转写文字（STT）
- 情绪标签（平静 / 疑惑 / 不满 / 愤怒）
- 事件标签（笑声 / 掌声等）

相比 Whisper large 快 15x，无需单独的情绪识别模型。

流式处理策略：
SenseVoice 本身是离线模型（需要完整音频段），通过以下方式实现"流式"效果：
1. 音频分块缓冲（chunk_duration_ms 大小的滑动窗口）
2. 能量 VAD 检测语音活动（避免对静音做推理）
3. 端点检测（连续静音 >= endpoint_silence_ms 视为一句话结束）
4. 分段推理：每检测到一个完整语音段，送入 SenseVoice 推理
5. 增量输出：partial 结果 + final 结果的两级回调
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import numpy as np

from voice_optimized_rag.voice.stt import STTProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("sensevoice_stt")


@dataclass
class SenseVoiceResult:
    """SenseVoice 推理结果，包含转写文字和情绪/事件标签"""
    text: str
    emotion: str = "neutral"  # neutral / happy / confused / angry
    event: str = ""           # laughter / applause / etc.
    language: str = "zh"
    confidence: float = 0.0
    is_final: bool = True     # True=端点确认的完整结果; False=中间 partial 结果


# 情绪标签映射：SenseVoice 输出 → 系统标准标签
EMOTION_MAP = {
    "😊": "happy",
    "😠": "angry",
    "😔": "sad",
    "😐": "neutral",
    "🤔": "confused",
    "Happy": "happy",
    "Angry": "angry",
    "Sad": "sad",
    "Neutral": "neutral",
}


class SenseVoiceSTT(STTProvider):
    """
    SenseVoice-Small STT 提供商

    同时输出语音转写和情绪标签，无需独立情绪模型。
    通过 FunASR 框架加载，支持离线（整段推理）模式。
    """

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: str = "cuda:0",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model = None
        self._last_result: Optional[SenseVoiceResult] = None

    def _ensure_model(self) -> None:
        """延迟加载模型（首次调用时初始化）"""
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "Install FunASR for SenseVoice: pip install funasr"
            )
        self._model = AutoModel(
            model=self._model_id,
            trust_remote_code=True,
            device=self._device,
        )
        logger.info(f"Loaded SenseVoice model: {self._model_id} on {self._device}")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        转写音频并提取情绪标签

        Returns:
            纯转写文字（情绪标签通过 last_result 获取）
        """
        result = await self.transcribe_with_emotion(audio_data, sample_rate)
        return result.text

    async def transcribe_with_emotion(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> SenseVoiceResult:
        """
        转写音频，同时返回情绪和事件标签（离线整段推理）

        SenseVoice 的输出格式通常为:
          "<|zh|><|NEUTRAL|><|Speech|><|woitn|>转写文字内容"
        需要解析出各个标签和纯文本。
        """
        self._ensure_model()

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_running_loop()
        raw_result = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                input=audio_np,
                cache={},
                language="auto",
                use_itn=True,
            ),
        )

        result = self._parse_result(raw_result)
        result.is_final = True
        self._last_result = result
        return result

    def _parse_result(self, raw_result) -> SenseVoiceResult:
        """解析 SenseVoice 原始输出为结构化结果"""
        if not raw_result or not isinstance(raw_result, list) or len(raw_result) == 0:
            return SenseVoiceResult(text="")

        item = raw_result[0]
        raw_text = item.get("text", "") if isinstance(item, dict) else str(item)

        # 解析 SenseVoice 标签格式: <|lang|><|emotion|><|event|><|itn|>text
        emotion = "neutral"
        event = ""
        language = "zh"
        text = raw_text

        # 提取 <|...|> 标签
        import re
        tags = re.findall(r"<\|([^|]+)\|>", raw_text)
        # 去除所有标签，保留纯文本
        text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()

        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in ("zh", "en", "ja", "ko", "yue"):
                language = tag_lower
            elif tag_lower in ("neutral", "happy", "angry", "sad"):
                emotion = tag_lower
            elif tag_lower in ("speech", "music", "noise"):
                event = tag_lower
            elif tag in EMOTION_MAP:
                emotion = EMOTION_MAP[tag]

        return SenseVoiceResult(
            text=text,
            emotion=emotion,
            event=event,
            language=language,
        )

    @property
    def last_result(self) -> Optional[SenseVoiceResult]:
        """获取最近一次转写的完整结果（含情绪标签）"""
        return self._last_result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 流式 STT 封装 —— 基于 VAD + 分段推理实现 chunk-based streaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StreamingSenseVoiceSTT:
    """
    流式 SenseVoice STT —— 通过 VAD 分段 + 增量推理实现伪流式

    设计原理：
    ┌───────────────────────────────────────────────────────────────┐
    │  实时音频流                                                    │
    │    ↓                                                          │
    │  [Chunk Buffer] ← 每次接收 chunk_duration_ms 的音频           │
    │    ↓                                                          │
    │  [Energy VAD] ← 计算帧能量，判断是否有语音活动                  │
    │    ↓                                                          │
    │  ┌─ 有语音 → 追加到 speech_buffer                             │
    │  │           每 partial_interval_ms 送一次推理 → partial 结果  │
    │  │                                                            │
    │  └─ 静音 >= endpoint_silence_ms → 端点检测                    │
    │              送整个 speech_buffer 推理 → final 结果             │
    │              清空 buffer，开始下一句                            │
    └───────────────────────────────────────────────────────────────┘

    使用方式：
        streaming_stt = StreamingSenseVoiceSTT(stt_provider)
        async for result in streaming_stt.stream(audio_chunk_iterator):
            if result.is_final:
                print(f"[FINAL] {result.text}")
            else:
                print(f"[PARTIAL] {result.text}")
    """

    def __init__(
        self,
        stt: SenseVoiceSTT,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        energy_threshold: float = 0.005,
        endpoint_silence_ms: int = 700,
        partial_interval_ms: int = 1000,
        min_speech_ms: int = 300,
        max_speech_ms: int = 15000,
    ) -> None:
        """
        Args:
            stt: 底层 SenseVoice 离线 STT 实例
            sample_rate: 音频采样率 (Hz)
            chunk_duration_ms: 每个输入 chunk 的时长 (ms)
            energy_threshold: 能量 VAD 阈值（0-1 范围的 float32 RMS）
            endpoint_silence_ms: 端点检测静音时长 (ms)；连续静音超过此值视为一句话结束
            partial_interval_ms: partial 结果推理间隔 (ms)；说话过程中多久推一次中间结果
            min_speech_ms: 最小语音段长度 (ms)；过短的段丢弃（避免噪声触发）
            max_speech_ms: 最大语音段长度 (ms)；超长段强制切割送推理
        """
        self._stt = stt
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._energy_threshold = energy_threshold
        self._endpoint_silence_ms = endpoint_silence_ms
        self._partial_interval_ms = partial_interval_ms
        self._min_speech_ms = min_speech_ms
        self._max_speech_ms = max_speech_ms

        # 每个 chunk 的采样点数
        self._chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        # 端点检测需要的连续静音 chunk 数
        self._endpoint_chunks = int(endpoint_silence_ms / chunk_duration_ms)
        # partial 推理间隔 chunk 数
        self._partial_chunks = int(partial_interval_ms / chunk_duration_ms)
        # 最小/最大语音段的采样点数
        self._min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._max_speech_samples = int(sample_rate * max_speech_ms / 1000)

    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SenseVoiceResult]:
        """
        流式处理音频 chunk，产出 partial + final 结果

        Args:
            audio_chunks: 异步迭代器，每次 yield chunk_duration_ms 的 PCM bytes

        Yields:
            SenseVoiceResult (is_final=False 为中间结果, is_final=True 为端点确认结果)
        """
        speech_buffer: list[np.ndarray] = []  # 当前语音段的 chunk 列表
        silence_count = 0                      # 连续静音 chunk 计数
        speech_count = 0                       # 当前段已积累的语音 chunk 计数
        last_partial_at = 0                    # 上次 partial 推理时的 chunk 计数
        is_speaking = False                    # 当前是否处于语音活动状态

        async for chunk_bytes in audio_chunks:
            # bytes → float32 numpy
            chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # ── VAD: 计算 RMS 能量 ──
            rms = float(np.sqrt(np.mean(chunk_np ** 2)))
            has_speech = rms > self._energy_threshold

            if has_speech:
                # 有语音活动
                silence_count = 0
                speech_buffer.append(chunk_np)
                speech_count += 1

                if not is_speaking:
                    is_speaking = True
                    logger.debug("VAD: speech start")

                # ── Partial 推理：每 partial_interval_ms 送一次 ──
                if (speech_count - last_partial_at) >= self._partial_chunks:
                    last_partial_at = speech_count
                    partial_result = await self._run_inference(speech_buffer)
                    if partial_result.text:
                        partial_result.is_final = False
                        yield partial_result

                # ── 强制切割：超过 max_speech_ms ──
                total_samples = sum(len(c) for c in speech_buffer)
                if total_samples >= self._max_speech_samples:
                    logger.info("Max speech length reached, force endpoint")
                    final_result = await self._run_inference(speech_buffer)
                    final_result.is_final = True
                    yield final_result
                    # 重置状态
                    speech_buffer.clear()
                    speech_count = 0
                    last_partial_at = 0
                    is_speaking = False

            else:
                # 静音
                silence_count += 1

                if is_speaking:
                    # 语音结束后仍保留 buffer（等端点确认）
                    speech_buffer.append(chunk_np)  # 保留尾部静音防截断

                    # ── 端点检测：连续静音 >= endpoint_silence_ms ──
                    if silence_count >= self._endpoint_chunks:
                        # 计算有效语音时长（排除尾部静音）
                        trailing_silence_samples = silence_count * self._chunk_samples
                        total_samples = sum(len(c) for c in speech_buffer)
                        speech_samples = total_samples - trailing_silence_samples

                        if speech_samples >= self._min_speech_samples:
                            # 有效语音段 → final 推理（含尾部静音以避免截断）
                            final_result = await self._run_inference(speech_buffer)
                            final_result.is_final = True
                            yield final_result
                        else:
                            logger.debug(
                                f"VAD: discarding short segment "
                                f"({speech_samples / self._sample_rate * 1000:.0f}ms)"
                            )

                        # 重置状态
                        speech_buffer.clear()
                        speech_count = 0
                        last_partial_at = 0
                        silence_count = 0
                        is_speaking = False

        # ── 流结束时，处理残余 buffer ──
        if speech_buffer:
            total_samples = sum(len(c) for c in speech_buffer)
            if total_samples >= self._min_speech_samples:
                final_result = await self._run_inference(speech_buffer)
                final_result.is_final = True
                yield final_result

    async def feed_chunk(self, chunk_bytes: bytes) -> Optional[SenseVoiceResult]:
        """
        单 chunk 喂入接口（适用于非 async-iterator 场景）

        调用方每次传入一个 chunk，内部维护状态。
        返回 None 表示无新结果；返回 SenseVoiceResult 表示有新的 partial/final。

        使用方式：
            streaming = StreamingSenseVoiceSTT(stt)
            streaming.reset()
            while audio_coming:
                result = await streaming.feed_chunk(chunk)
                if result and result.is_final:
                    process_final(result)
        """
        if not hasattr(self, "_feed_state"):
            self.reset()

        state = self._feed_state
        chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(chunk_np ** 2)))
        has_speech = rms > self._energy_threshold

        if has_speech:
            state["silence_count"] = 0
            state["speech_buffer"].append(chunk_np)
            state["speech_count"] += 1

            if not state["is_speaking"]:
                state["is_speaking"] = True

            # Partial 推理
            if (state["speech_count"] - state["last_partial_at"]) >= self._partial_chunks:
                state["last_partial_at"] = state["speech_count"]
                result = await self._run_inference(state["speech_buffer"])
                if result.text:
                    result.is_final = False
                    return result

            # 强制切割
            total_samples = sum(len(c) for c in state["speech_buffer"])
            if total_samples >= self._max_speech_samples:
                result = await self._run_inference(state["speech_buffer"])
                result.is_final = True
                self._reset_feed_state()
                return result

        else:
            state["silence_count"] += 1

            if state["is_speaking"]:
                state["speech_buffer"].append(chunk_np)

                if state["silence_count"] >= self._endpoint_chunks:
                    trailing_silence_samples = state["silence_count"] * self._chunk_samples
                    total_samples = sum(len(c) for c in state["speech_buffer"])
                    speech_samples = total_samples - trailing_silence_samples

                    if speech_samples >= self._min_speech_samples:
                        result = await self._run_inference(state["speech_buffer"])
                        result.is_final = True
                        self._reset_feed_state()
                        return result
                    else:
                        self._reset_feed_state()

        return None

    def reset(self) -> None:
        """重置流式状态（新对话开始时调用）"""
        self._feed_state = {
            "speech_buffer": [],
            "silence_count": 0,
            "speech_count": 0,
            "last_partial_at": 0,
            "is_speaking": False,
        }

    def _reset_feed_state(self) -> None:
        """内部状态重置"""
        self._feed_state["speech_buffer"] = []
        self._feed_state["silence_count"] = 0
        self._feed_state["speech_count"] = 0
        self._feed_state["last_partial_at"] = 0
        self._feed_state["is_speaking"] = False

    async def _run_inference(self, chunks: list[np.ndarray]) -> SenseVoiceResult:
        """
        拼接 chunk 并送入 SenseVoice 推理

        将 float32 numpy 列表拼接为 int16 bytes 后调用底层 STT。
        """
        if not chunks:
            return SenseVoiceResult(text="")

        # 拼接所有 chunk 为完整音频段
        audio_concat = np.concatenate(chunks)
        # 转回 int16 bytes（SenseVoice 接口期望 int16 PCM）
        audio_int16 = (audio_concat * 32768).clip(-32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # 调用底层离线推理
        result = await self._stt.transcribe_with_emotion(audio_bytes, self._sample_rate)
        return result
