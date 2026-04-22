"""
本地 CosyVoice TTS 模块已按要求整体注释停用。

说明：
- 原本地 CosyVoice2 / 音色克隆实现不再参与项目运行。
- 如需语音合成，请改用云端 provider：
  - voice.siliconflow_tts.SiliconFlowTTS
  - voice.tts.OpenAITTS
  - voice.tts.EdgeTTS
- 如需恢复本地实现，请查看 git 历史版本。
"""

from __future__ import annotations

from typing import Optional

from voice_optimized_rag.voice.tts import TTSProvider


class CosyVoiceTTS(TTSProvider):
    """本地 CosyVoice TTS 已停用。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "CosyVoice 本地 TTS 部署已被注释停用，请改用 SiliconFlowTTS、OpenAITTS 或 EdgeTTS。"
        )

    def set_reference_audio(self, audio_data: bytes) -> None:
        raise RuntimeError(
            "CosyVoice 本地 TTS 部署已被注释停用，请改用云端 TTS。"
        )

    async def synthesize(
        self,
        text: str,
        mode: str = "standard",
        reference_audio: Optional[bytes] = None,
        speaker: Optional[str] = None,
        instruct_text: Optional[str] = None,
    ) -> bytes:
        raise RuntimeError(
            "CosyVoice 本地 TTS 部署已被注释停用，请改用 SiliconFlowTTS、OpenAITTS 或 EdgeTTS。"
        )

    async def synthesize_stream(
        self,
        text: str,
        mode: str = "standard",
        speaker: Optional[str] = None,
    ):
        raise RuntimeError(
            "CosyVoice 本地 TTS 部署已被注释停用，请改用 SiliconFlowTTS、OpenAITTS 或 EdgeTTS。"
        )