"""
CosyVoice TTS + 音色克隆 一体化封装

基于阿里达摩院 CosyVoice 2 模型：
- 标准合成模式（mode="standard"）：使用预设音色
- 音色克隆模式（mode="clone"）：使用参考音频零样本克隆
- Instruct 模式（mode="instruct"）：通过文本指令控制风格

单一模型同时提供 TTS 和音色克隆，无需分离部署。
"""

from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator, Optional

from voice_optimized_rag.voice.tts import TTSProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("cosyvoice")


class CosyVoiceTTS(TTSProvider):
    """
    CosyVoice 2 TTS 提供商，同时支持语音合成和音色克隆

    用法：
        # 标准合成
        audio = await tts.synthesize("你好，请问有什么可以帮您？")

        # 音色克隆
        audio = await tts.synthesize(
            "你好",
            mode="clone",
            reference_audio=ref_bytes,
        )
    """

    def __init__(
        self,
        model_id: str = "iic/CosyVoice2-0.5B",
        device: str = "cuda:0",
        default_speaker: str = "中文女",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._default_speaker = default_speaker
        self._model = None
        self._reference_audio: Optional[bytes] = None

    def _ensure_model(self) -> None:
        """延迟加载模型"""
        if self._model is not None:
            return
        try:
            from cosyvoice import CosyVoice2
        except ImportError:
            raise ImportError(
                "Install CosyVoice: git clone https://github.com/FunAudioLLM/CosyVoice && cd CosyVoice && pip install -e ."
            )
        self._model = CosyVoice2(self._model_id, load_jit=True, load_trt=False)
        logger.info(f"Loaded CosyVoice model: {self._model_id}")

    def set_reference_audio(self, audio_data: bytes) -> None:
        """设置参考音频（供后续克隆模式使用，3-10秒最佳）"""
        self._reference_audio = audio_data
        logger.info(f"Reference audio set: {len(audio_data)} bytes")

    async def synthesize(
        self,
        text: str,
        mode: str = "standard",
        reference_audio: Optional[bytes] = None,
        speaker: Optional[str] = None,
        instruct_text: Optional[str] = None,
    ) -> bytes:
        """
        合成语音

        Args:
            text: 要合成的文本
            mode: 合成模式 - "standard" | "clone" | "instruct"
            reference_audio: 参考音频（克隆模式）
            speaker: 预设说话人名称（标准模式）
            instruct_text: 风格指令（instruct 模式，如"用温柔的声音说"）

        Returns:
            合成的 PCM 音频字节
        """
        self._ensure_model()

        loop = asyncio.get_running_loop()

        if mode == "clone":
            ref = reference_audio or self._reference_audio
            if ref is None:
                raise ValueError("Clone mode requires reference_audio")
            audio = await loop.run_in_executor(
                None,
                lambda: self._synthesize_clone(text, ref),
            )
        elif mode == "instruct":
            spk = speaker or self._default_speaker
            inst = instruct_text or ""
            audio = await loop.run_in_executor(
                None,
                lambda: self._synthesize_instruct(text, spk, inst),
            )
        else:
            spk = speaker or self._default_speaker
            audio = await loop.run_in_executor(
                None,
                lambda: self._synthesize_standard(text, spk),
            )

        return audio

    async def synthesize_stream(
        self,
        text: str,
        mode: str = "standard",
        speaker: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """
        流式合成（边生成边输出）

        CosyVoice 2 支持流式 token 预测，首字节延迟 < 150ms。
        使用 asyncio.Queue 桥接同步生成器与异步迭代器，避免阻塞事件循环。

        Yields:
            音频数据块（PCM 格式）
        """
        self._ensure_model()
        spk = speaker or self._default_speaker
        loop = asyncio.get_running_loop()

        # 使用 Queue 桥接：在线程中运行同步生成器，逐块推入 Queue
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=8)

        async def _producer():
            """在线程池中运行同步生成器，将块推入 queue"""
            def _run_sync():
                try:
                    for chunk in self._model.inference_sft(text, spk, stream=True):
                        pcm = self._tensor_to_pcm(chunk["tts_speech"])
                        # 从工作线程安全地放入 asyncio Queue
                        asyncio.run_coroutine_threadsafe(
                            queue.put(pcm), loop
                        ).result(timeout=5.0)
                finally:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None), loop  # sentinel: 生成结束
                    ).result(timeout=5.0)

            await loop.run_in_executor(None, _run_sync)

        # 启动后台生产者任务
        producer_task = asyncio.create_task(_producer())

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not producer_task.done():
                producer_task.cancel()

    def _synthesize_standard(self, text: str, speaker: str) -> bytes:
        """标准预设音色合成"""
        result = self._model.inference_sft(text, speaker, stream=False)
        audio_tensor = next(result)["tts_speech"]
        return self._tensor_to_pcm(audio_tensor)

    def _synthesize_clone(self, text: str, reference_audio: bytes) -> bytes:
        """零样本音色克隆合成"""
        import numpy as np
        ref_np = np.frombuffer(reference_audio, dtype=np.int16).astype(np.float32) / 32768.0
        prompt_speech_16k = self._resample_if_needed(ref_np, 16000)

        result = self._model.inference_zero_shot(
            text, "", prompt_speech_16k, stream=False
        )
        audio_tensor = next(result)["tts_speech"]
        return self._tensor_to_pcm(audio_tensor)

    def _synthesize_instruct(self, text: str, speaker: str, instruct_text: str) -> bytes:
        """指令控制风格合成"""
        result = self._model.inference_instruct2(
            text, instruct_text, speaker, stream=False
        )
        audio_tensor = next(result)["tts_speech"]
        return self._tensor_to_pcm(audio_tensor)

    @staticmethod
    def _tensor_to_pcm(audio_tensor) -> bytes:
        """将模型输出的 tensor 转换为 16-bit PCM 字节"""
        import numpy as np
        if hasattr(audio_tensor, "numpy"):
            audio_np = audio_tensor.numpy()
        else:
            audio_np = np.array(audio_tensor)
        audio_np = audio_np.flatten()
        audio_int16 = (audio_np * 32768).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def _resample_if_needed(audio: "np.ndarray", target_sr: int) -> "np.ndarray":
        """如有需要，重采样音频到目标采样率"""
        return audio  # 简化实现，实际可能需要 librosa.resample
