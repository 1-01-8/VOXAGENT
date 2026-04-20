"""
SiliconFlow Cloud TTS —— 使用硅基流动托管的 FunAudioLLM/CosyVoice2-0.5B API

相比本地 CosyVoice2 部署：
- 无需 GPU / cuDNN / TensorRT，启动即用（原先本地 RTF ≈ 6-10）
- 云端 RTF 实测 < 1，首字节延迟 ~400-800ms
- 支持 `stream=true` 的流式 chunked 响应 → 边生成边播
- API: POST https://api.siliconflow.cn/v1/audio/speech
  请求体 JSON:
    {
      "model": "FunAudioLLM/CosyVoice2-0.5B",
      "input": "...文本...",
      "voice": "FunAudioLLM/CosyVoice2-0.5B:alex",
      "response_format": "pcm",
      "sample_rate": 24000,
      "stream": true
    }
  响应: chunked raw PCM (16-bit LE mono)

提供与本地 CosyVoiceTTS 对齐的接口:
- synthesize(text)              -> 整段 PCM
- synthesize_stream(text)       -> 句级流式 AsyncIterator[bytes]
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator, Optional

from voice_optimized_rag.voice.tts import TTSProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("siliconflow_tts")


class SiliconFlowTTS(TTSProvider):
    """硅基流动云端 TTS Provider (FunAudioLLM/CosyVoice2-0.5B)."""

    _DEFAULT_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
    # SF 默认预置音色 (系统音色) — alex 为中性男声。如需其它：
    # anna / bella / benjamin / charles / claire / david / diana 等
    _DEFAULT_VOICE_SUFFIX = "alex"

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        voice: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        sample_rate: int = 24000,
        response_format: str = "pcm",
        speed: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("SiliconFlow TTS 需要 api_key (VOR_SILICONFLOW_API_KEY)")
        self._api_key = api_key
        self._model = model
        # voice 可传 "alex" 或完整 "FunAudioLLM/CosyVoice2-0.5B:alex"
        v = (voice or self._DEFAULT_VOICE_SUFFIX).strip()
        self._voice = v if ":" in v else f"{model}:{v}"
        self._url = f"{base_url.rstrip('/')}/audio/speech"
        self._sample_rate = sample_rate
        self._response_format = response_format
        self._speed = speed
        self._timeout = timeout
        self._client = None
        logger.info(
            f"SiliconFlow TTS ready | model={model} voice={self._voice} "
            f"sr={sample_rate} fmt={response_format}"
        )

    # 让 web.server 的 PIPE-TIMER 用 24kHz 还是 16kHz 计算 audio_ms
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def _get_client(self):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            )
        return self._client

    def _build_payload(self, text: str, stream: bool) -> dict:
        return {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": self._response_format,
            "sample_rate": self._sample_rate,
            "stream": stream,
            "speed": self._speed,
        }

    # ─────────────────────────────────────────────────────────────
    # 整段合成：一次请求拿回全部 PCM
    # ─────────────────────────────────────────────────────────────
    async def synthesize(self, text: str, **_: dict) -> bytes:
        if not text.strip():
            return b""
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                self._url,
                json=self._build_payload(text, stream=False),
                headers=headers,
            )
            resp.raise_for_status()
            pcm = resp.content
        except Exception as e:
            logger.warning(f"SiliconFlow TTS 调用失败: {e}")
            return b""
        wall_ms = (time.perf_counter() - t0) * 1000
        audio_ms = (len(pcm) // 2) * 1000.0 / self._sample_rate if pcm else 0
        rtf = wall_ms / audio_ms if audio_ms else 0
        logger.info(
            f"[TTS-TIMER] sf_tts chars={len(text)} wall={wall_ms:.0f}ms "
            f"audio={audio_ms:.0f}ms RTF={rtf:.2f}"
        )
        return pcm

    # ─────────────────────────────────────────────────────────────
    # 流式合成（HTTP chunked transfer）：首字节即可下推
    # ─────────────────────────────────────────────────────────────
    async def synthesize_http_stream(
        self, text: str, chunk_size: int = 4800
    ) -> AsyncIterator[bytes]:
        """对单句文本发起 stream=true 请求，按固定 PCM 块大小持续 yield。

        chunk_size 默认 4800 字节 ≈ 100ms @24kHz/16bit，足够前端顺畅播放。
        """
        if not text.strip():
            return
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(text, stream=True)
        t0 = time.perf_counter()
        first_byte_logged = False
        total_bytes = 0
        try:
            async with client.stream(
                "POST", self._url, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                buf = b""
                async for raw in resp.aiter_bytes():
                    if not raw:
                        continue
                    if not first_byte_logged:
                        first_byte_logged = True
                        logger.info(
                            f"[TTS-TIMER] sf_stream TTFB={(time.perf_counter()-t0)*1000:.0f}ms "
                            f"chars={len(text)}"
                        )
                    buf += raw
                    # 按 chunk_size 切齐下发，便于前端恒定节奏播放
                    while len(buf) >= chunk_size:
                        yield buf[:chunk_size]
                        buf = buf[chunk_size:]
                        total_bytes += chunk_size
                if buf:
                    yield buf
                    total_bytes += len(buf)
        except Exception as e:
            logger.warning(f"SiliconFlow TTS stream 调用失败: {e}")
            return
        wall_ms = (time.perf_counter() - t0) * 1000
        audio_ms = (total_bytes // 2) * 1000.0 / self._sample_rate if total_bytes else 0
        logger.info(
            f"[TTS-TIMER] sf_stream DONE wall={wall_ms:.0f}ms "
            f"audio={audio_ms:.0f}ms RTF={wall_ms/audio_ms if audio_ms else 0:.2f}"
        )

    # ─────────────────────────────────────────────────────────────
    # 句级流式（兼容 CosyVoiceTTS.synthesize_stream 接口）
    # —— 对每句调用云端 stream=true；多句并发预取以隐藏网络延迟
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _split_sentences(text: str, min_len: int = 6) -> list[str]:
        parts = re.split(r"(?<=[。！？!?.；;\n])", text)
        merged: list[str] = []
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            buf += p
            if len(buf) >= min_len:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] = merged[-1] + buf
            else:
                merged.append(buf)
        return merged

    async def synthesize_stream(
        self, text: str, **_: dict
    ) -> AsyncIterator[bytes]:
        """句级流式：按标点切句 → 并发预取下一句 → 严格按顺序 yield。

        这是给不支持 HTTP chunked 消费的调用方用的兜底路径。
        对 web.server 这样的场景建议直接用 synthesize_http_stream。
        """
        sentences = self._split_sentences(text) or [text]
        logger.info(f"[TTS-TIMER] sf_stream split={len(sentences)}")

        # 预取窗口：同时发起 2 个请求，保证拿到当前句的同时下一句已在路上
        prefetch = 2
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=prefetch + 1)

        async def _producer():
            try:
                sem = asyncio.Semaphore(prefetch)
                tasks: list[asyncio.Task] = []

                async def _one(sent: str) -> bytes:
                    async with sem:
                        return await self.synthesize(sent)

                for sent in sentences:
                    tasks.append(asyncio.create_task(_one(sent)))

                for task in tasks:
                    pcm = await task
                    if pcm:
                        await queue.put(pcm)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(_producer())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not producer.done():
                producer.cancel()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
