"""Text-to-Speech abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("tts")


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to convert to speech.

        Returns:
            Raw audio bytes (16-bit PCM).
        """


class EdgeTTS(TTSProvider):
    """Free TTS using Microsoft Edge's TTS service (no API key needed)."""

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("Install edge-tts: pip install voice-optimized-rag[voice]")
        self._voice = voice

    async def synthesize(self, text: str) -> bytes:
        import edge_tts
        import io

        communicate = edge_tts.Communicate(text, self._voice)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)


class OpenAITTS(TTSProvider):
    """TTS using OpenAI's API."""

    def __init__(self, api_key: str, voice: str = "alloy", model: str = "tts-1") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install voice-optimized-rag[openai]")
        self._client = AsyncOpenAI(api_key=api_key)
        self._voice = voice
        self._model = model

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        )
        return response.content


def create_tts(provider: str, **kwargs) -> TTSProvider:
    """Factory function to create a TTS provider."""
    if provider == "edge":
        return EdgeTTS(voice=kwargs.get("voice", "en-US-AriaNeural"))
    elif provider == "openai":
        return OpenAITTS(
            api_key=kwargs["api_key"],
            voice=kwargs.get("voice", "alloy"),
        )
    elif provider == "cosyvoice":
        # from voice_optimized_rag.voice.cosyvoice import CosyVoiceTTS
        # return CosyVoiceTTS(
        #     model_id=kwargs.get("model_id", "iic/CosyVoice2-0.5B"),
        #     device=kwargs.get("device", "cuda:0"),
        #     default_speaker=kwargs.get("default_speaker", "中文女"),
        # )
        raise RuntimeError("create_tts('cosyvoice') 已停用，本地 TTS 部署已被注释关闭。")
    elif provider == "siliconflow":
        from voice_optimized_rag.voice.siliconflow_tts import SiliconFlowTTS
        return SiliconFlowTTS(
            api_key=kwargs["api_key"],
            model=kwargs.get("sf_tts_model", "FunAudioLLM/CosyVoice2-0.5B"),
            voice=kwargs.get("sf_tts_voice", "alex"),
            base_url=kwargs.get("sf_base_url", "https://api.siliconflow.cn/v1"),
            sample_rate=kwargs.get("sf_tts_sample_rate", 24000),
            response_format=kwargs.get("sf_tts_format", "pcm"),
            speed=kwargs.get("sf_tts_speed", 1.0),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
