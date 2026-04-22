"""Tests for StreamingSenseVoiceSTT — chunk-based 流式语音识别

系统组件: Voice — 流式 STT (VAD + 分段推理)
源文件:   voice_optimized_rag/voice/sensevoice_stt.py (StreamingSenseVoiceSTT class)
职责:     将离线 SenseVoice 通过 VAD + chunk 缓冲实现伪流式识别

测试覆盖：
- 能量 VAD 检测（语音/静音区分）
- 端点检测（连续静音触发 final 结果）
- Partial 结果输出（说话过程中间推理）
- 短语音段丢弃（低于 min_speech_ms 不触发推理）
- 最大长度强制切割（超过 max_speech_ms 自动分段）
- feed_chunk 接口（非 async-iterator 场景）
- reset 状态重置
- 流结束残余 buffer 处理
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.skip("本地 Streaming SenseVoice STT 部署已按要求停用，此测试文件整体跳过。", allow_module_level=True)

from voice_optimized_rag.voice.sensevoice_stt import (
    SenseVoiceSTT,
    SenseVoiceResult,
    StreamingSenseVoiceSTT,
)


# ─── 辅助工具 ───


def make_speech_chunk(duration_ms: int = 100, sample_rate: int = 16000, amplitude: float = 0.1) -> bytes:
    """生成含语音能量的 PCM chunk（正弦波模拟）"""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    # 440Hz 正弦波，幅度 amplitude
    signal = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return signal.tobytes()


def make_silence_chunk(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """生成静音 PCM chunk"""
    n_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


async def chunks_to_async_iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """将 chunk 列表转为异步迭代器"""
    for chunk in chunks:
        yield chunk


class MockSenseVoiceSTT(SenseVoiceSTT):
    """Mock SenseVoice STT，不加载模型，直接返回模拟结果"""

    def __init__(self):
        super().__init__(model_id="mock", device="cpu")
        self._model = MagicMock()  # 避免触发 _ensure_model 的 import
        self._call_count = 0

    async def transcribe_with_emotion(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> SenseVoiceResult:
        self._call_count += 1
        # 根据音频长度模拟不同文本
        n_samples = len(audio_data) // 2  # int16 = 2 bytes/sample
        duration_ms = n_samples / sample_rate * 1000
        return SenseVoiceResult(
            text=f"识别结果_{self._call_count}",
            emotion="neutral",
            is_final=True,
        )


# ─── 测试类 ───


@pytest.fixture
def mock_stt() -> MockSenseVoiceSTT:
    return MockSenseVoiceSTT()


@pytest.fixture
def streaming_stt(mock_stt) -> StreamingSenseVoiceSTT:
    return StreamingSenseVoiceSTT(
        stt=mock_stt,
        sample_rate=16000,
        chunk_duration_ms=100,
        energy_threshold=0.005,
        endpoint_silence_ms=300,     # 3 个 chunk 的静音触发端点
        partial_interval_ms=500,     # 5 个 chunk 出一次 partial
        min_speech_ms=200,           # 至少 2 个 chunk 才算有效语音
        max_speech_ms=2000,          # 20 个 chunk 强制切割
    )


class TestVADDetection:
    """VAD 能量检测逻辑"""

    @pytest.mark.asyncio
    async def test_speech_detected(self, streaming_stt: StreamingSenseVoiceSTT):
        """有语音能量的 chunk 应被识别为语音"""
        speech = make_speech_chunk(100, amplitude=0.1)
        chunk_np = np.frombuffer(speech, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(chunk_np ** 2)))
        assert rms > streaming_stt._energy_threshold

    @pytest.mark.asyncio
    async def test_silence_not_detected(self, streaming_stt: StreamingSenseVoiceSTT):
        """静音 chunk 不应被识别为语音"""
        silence = make_silence_chunk(100)
        chunk_np = np.frombuffer(silence, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(chunk_np ** 2)))
        assert rms <= streaming_stt._energy_threshold


class TestEndpointDetection:
    """端点检测（连续静音触发 final）"""

    @pytest.mark.asyncio
    async def test_endpoint_triggers_final(self, streaming_stt: StreamingSenseVoiceSTT):
        """语音后连续静音应触发 final 结果"""
        chunks = []
        # 5 个语音 chunk (500ms > min_speech_ms 200ms)
        for _ in range(5):
            chunks.append(make_speech_chunk(100))
        # 3 个静音 chunk (300ms = endpoint_silence_ms)
        for _ in range(3):
            chunks.append(make_silence_chunk(100))

        results = []
        async for result in streaming_stt.stream(chunks_to_async_iter(chunks)):
            results.append(result)

        # 应至少有一个 final 结果
        final_results = [r for r in results if r.is_final]
        assert len(final_results) >= 1
        assert final_results[-1].text != ""

    @pytest.mark.asyncio
    async def test_short_speech_discarded(self, streaming_stt: StreamingSenseVoiceSTT):
        """短语音段（< min_speech_ms）应被丢弃"""
        chunks = []
        # 1 个语音 chunk (100ms < min_speech_ms 200ms)
        chunks.append(make_speech_chunk(100))
        # 3 个静音 chunk 触发端点
        for _ in range(3):
            chunks.append(make_silence_chunk(100))

        results = []
        async for result in streaming_stt.stream(chunks_to_async_iter(chunks)):
            results.append(result)

        # 不应产出 final 结果（太短被丢弃）
        final_results = [r for r in results if r.is_final]
        assert len(final_results) == 0


class TestPartialResults:
    """Partial 中间结果输出"""

    @pytest.mark.asyncio
    async def test_partial_emitted_during_speech(self, streaming_stt: StreamingSenseVoiceSTT):
        """持续说话应每 partial_interval_ms 输出一次 partial"""
        chunks = []
        # 10 个语音 chunk (1000ms)，partial_interval=500ms → 应有 2 次 partial
        for _ in range(10):
            chunks.append(make_speech_chunk(100))
        # 加静音触发端点
        for _ in range(3):
            chunks.append(make_silence_chunk(100))

        results = []
        async for result in streaming_stt.stream(chunks_to_async_iter(chunks)):
            results.append(result)

        partial_results = [r for r in results if not r.is_final]
        # 5 个 chunk 时第一次 partial, 10 个 chunk 时第二次 partial
        assert len(partial_results) >= 1


class TestMaxSpeechCutoff:
    """最大语音段强制切割"""

    @pytest.mark.asyncio
    async def test_max_speech_forces_final(self, streaming_stt: StreamingSenseVoiceSTT):
        """超过 max_speech_ms 应强制切割并输出 final"""
        chunks = []
        # 25 个语音 chunk (2500ms > max_speech_ms 2000ms)
        for _ in range(25):
            chunks.append(make_speech_chunk(100))

        results = []
        async for result in streaming_stt.stream(chunks_to_async_iter(chunks)):
            results.append(result)

        final_results = [r for r in results if r.is_final]
        # 应至少因超长而切割一次
        assert len(final_results) >= 1


class TestFeedChunk:
    """feed_chunk 单 chunk 喂入接口"""

    @pytest.mark.asyncio
    async def test_feed_returns_none_during_speech(self, streaming_stt: StreamingSenseVoiceSTT):
        """说话开始的前几个 chunk 应返回 None（未到 partial 间隔）"""
        streaming_stt.reset()
        result = await streaming_stt.feed_chunk(make_speech_chunk(100))
        # 第一个 chunk 不够 partial_interval，应为 None
        assert result is None

    @pytest.mark.asyncio
    async def test_feed_returns_final_after_silence(self, streaming_stt: StreamingSenseVoiceSTT):
        """语音后连续静音应通过 feed_chunk 返回 final"""
        streaming_stt.reset()

        # 喂入 3 个语音 chunk
        for _ in range(3):
            await streaming_stt.feed_chunk(make_speech_chunk(100))

        # 喂入 3 个静音 chunk
        result = None
        for _ in range(3):
            r = await streaming_stt.feed_chunk(make_silence_chunk(100))
            if r is not None:
                result = r

        assert result is not None
        assert result.is_final is True

    @pytest.mark.asyncio
    async def test_feed_partial_at_interval(self, streaming_stt: StreamingSenseVoiceSTT):
        """feed_chunk 到达 partial_interval 时应返回 partial"""
        streaming_stt.reset()

        results = []
        # 喂入 6 个语音 chunk (600ms > partial_interval 500ms)
        for _ in range(6):
            r = await streaming_stt.feed_chunk(make_speech_chunk(100))
            if r is not None:
                results.append(r)

        # 应有一个 partial 结果（第 5 个 chunk 时触发）
        assert len(results) >= 1
        assert results[0].is_final is False


class TestReset:
    """状态重置"""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, streaming_stt: StreamingSenseVoiceSTT):
        """reset() 后状态应清空"""
        streaming_stt.reset()
        # 喂入一些数据
        await streaming_stt.feed_chunk(make_speech_chunk(100))
        await streaming_stt.feed_chunk(make_speech_chunk(100))

        # Reset
        streaming_stt.reset()
        state = streaming_stt._feed_state
        assert state["speech_buffer"] == []
        assert state["silence_count"] == 0
        assert state["speech_count"] == 0
        assert state["is_speaking"] is False


class TestStreamEnd:
    """流结束处理"""

    @pytest.mark.asyncio
    async def test_residual_buffer_processed(self, streaming_stt: StreamingSenseVoiceSTT):
        """流结束时残余 buffer 应被处理"""
        chunks = []
        # 5 个语音 chunk，无静音结尾（流突然结束）
        for _ in range(5):
            chunks.append(make_speech_chunk(100))

        results = []
        async for result in streaming_stt.stream(chunks_to_async_iter(chunks)):
            results.append(result)

        # 流结束后应 flush 残余 buffer 为 final
        final_results = [r for r in results if r.is_final]
        assert len(final_results) >= 1
