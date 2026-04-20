"""
SiliconFlow Cloud STT —— 使用硅基流动托管的 FunAudioLLM/SenseVoiceSmall API

相比本地 SenseVoice (FunASR) 部署：
- 无需 GPU / cuDNN / TensorRT，启动即用
- 推理延迟由云端保障（单段 < 1s，无 RTF 劣化问题）
- API 端点: POST https://api.siliconflow.cn/v1/audio/transcriptions
- Multipart form: file + model=FunAudioLLM/SenseVoiceSmall

暴露接口与本地 SenseVoiceSTT 对齐（transcribe / transcribe_with_emotion）
使 web.server 侧无需改动调用逻辑。
"""

from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass
from typing import Optional

from voice_optimized_rag.voice.stt import STTProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("siliconflow_stt")


@dataclass
class SiliconFlowSTTResult:
    """兼容 SenseVoiceResult 的转写结果。云端 API 仅返回文本，情绪字段置为 neutral。"""
    text: str
    emotion: str = "neutral"
    event: str = ""
    language: str = "zh"
    confidence: float = 0.0
    is_final: bool = True


class SiliconFlowSTT(STTProvider):
    """硅基流动云端 STT Provider (FunAudioLLM/SenseVoiceSmall)."""

    _DEFAULT_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
    _DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("SiliconFlow STT 需要 api_key (VOR_SILICONFLOW_API_KEY)")
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/audio/transcriptions"
        self._timeout = timeout
        self._client = None  # lazy init (httpx.AsyncClient 需在 event loop 内)
        self._last_result: Optional[SiliconFlowSTTResult] = None
        logger.info(f"SiliconFlow STT ready | model={model}")

    async def _get_client(self):
        import httpx
        if self._client is None:
            # 使用连接池 + HTTP/2，减少每次请求的 TLS 握手
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
                http2=False,  # SF 侧稳定性：HTTP/1.1 已足够
            )
        return self._client

    @staticmethod
    def _pcm_to_wav(audio_data: bytes, sample_rate: int) -> bytes:
        """原始 PCM 16-bit mono → WAV container (API 需要容器格式)。"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return buf.getvalue()

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        result = await self.transcribe_with_emotion(audio_data, sample_rate)
        return result.text

    async def transcribe_with_emotion(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> SiliconFlowSTTResult:
        """兼容 SenseVoiceSTT.transcribe_with_emotion 的接口。"""
        import time as _time
        t0 = _time.perf_counter()

        wav_bytes = self._pcm_to_wav(audio_data, sample_rate)
        client = await self._get_client()

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self._model}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            resp = await client.post(self._url, files=files, data=data, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            logger.warning(f"SiliconFlow STT 调用失败: {e}")
            return SiliconFlowSTTResult(text="", is_final=True)

        text = (payload or {}).get("text", "").strip()
        wall_ms = (_time.perf_counter() - t0) * 1000
        logger.info(f"[STT-TIMER] sf_stt wall={wall_ms:.0f}ms chars={len(text)}")

        result = SiliconFlowSTTResult(text=text, is_final=True)
        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[SiliconFlowSTTResult]:
        return self._last_result

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
