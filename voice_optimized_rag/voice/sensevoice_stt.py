"""
本地 SenseVoice STT 模块已按要求整体注释停用。

说明：
- 原本地 FunASR / SenseVoice 推理实现不再参与项目运行。
- 如需语音识别，请改用云端 provider：
  - voice.siliconflow_stt.SiliconFlowSTT
  - voice.stt.OpenAISTT
- 如需恢复本地实现，请查看 git 历史版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from voice_optimized_rag.voice.stt import STTProvider


EMOTION_MAP: dict[str, str] = {}


@dataclass
class SenseVoiceResult:
    text: str
    emotion: str = "neutral"
    event: str = ""
    language: str = "zh"
    confidence: float = 0.0
    is_final: bool = True


class SenseVoiceSTT(STTProvider):
    """本地 SenseVoice STT 已停用。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "SenseVoice 本地 STT 部署已被注释停用，请改用 SiliconFlowSTT 或 OpenAISTT。"
        )

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        raise RuntimeError(
            "SenseVoice 本地 STT 部署已被注释停用，请改用 SiliconFlowSTT 或 OpenAISTT。"
        )

    async def transcribe_with_emotion(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> SenseVoiceResult:
        raise RuntimeError(
            "SenseVoice 本地 STT 部署已被注释停用，请改用 SiliconFlowSTT 或 OpenAISTT。"
        )

    @property
    def last_result(self) -> Optional[SenseVoiceResult]:
        return None


class StreamingSenseVoiceSTT:
    """本地 Streaming SenseVoice STT 已停用。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "StreamingSenseVoiceSTT 已被注释停用，请改用云端 STT 或前端文字输入。"
        )

    async def feed_chunk(self, chunk_bytes: bytes):
        raise RuntimeError(
            "StreamingSenseVoiceSTT 已被注释停用，请改用云端 STT 或前端文字输入。"
        )

    def reset(self) -> None:
        raise RuntimeError(
            "StreamingSenseVoiceSTT 已被注释停用，请改用云端 STT 或前端文字输入。"
        )