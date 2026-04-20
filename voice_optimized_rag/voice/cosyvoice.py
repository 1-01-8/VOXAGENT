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
        self._cached_spk_id: Optional[str] = None

    def _ensure_model(self) -> None:
        """延迟加载模型"""
        if self._model is not None:
            return
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError:
            raise ImportError(
                "Install CosyVoice: git clone https://github.com/FunAudioLLM/CosyVoice && cd CosyVoice && pip install -e ."
            )
        import os
        import torch
        target_idx = int(self._device.split(":")[-1]) if "cuda" in self._device else 0
        with torch.cuda.device(target_idx):
            self._model = CosyVoice2(self._model_id, load_jit=True, load_trt=False, fp16=True)
        logger.info(f"Loaded CosyVoice model: {self._model_id} on cuda:{target_idx}, sr={self._model.sample_rate}")

        # ── 预缓存参考音色为 "default"，之后合成走 inference_sft 跳过参考音频处理 ──
        ref_path = os.environ.get("VOR_COSYVOICE_REFERENCE_AUDIO") \
            or "/home/xxm/VoxCareAgent/CosyVoice/asset/zero_shot_prompt.wav"
        try:
            self._model.add_zero_shot_spk(
                "希望你以后能够做的比我还好呦。", ref_path, "default",
            )
            self._cached_spk_id = "default"
            logger.info(f"Cached speaker embedding from {ref_path}")
        except Exception as e:
            self._cached_spk_id = None
            logger.warning(f"音色预缓存失败，将走 zero-shot: {e}")

        # ── 预热：合成一句短语触发 JIT 编译 ──
        try:
            with torch.cuda.device(target_idx):
                if self._cached_spk_id:
                    list(self._model.inference_zero_shot(
                        "你好", "", "", zero_shot_spk_id=self._cached_spk_id, stream=False,
                    ))
                else:
                    list(self._model.inference_zero_shot(
                        "你好", "希望你以后能够做的比我还好呦。", ref_path, stream=False,
                    ))
            logger.info("CosyVoice warmup done")
        except Exception as e:
            logger.warning(f"CosyVoice 预热失败: {e}")

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

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中英文标点切句，保留标点。短句合并到 >= min_len 以摊薄 per-call 开销。"""
        import re
        parts = re.split(r"(?<=[。！？!?.；;\n])", text)
        merged: list[str] = []
        buf = ""
        min_len = 8
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
        self,
        text: str,
        mode: str = "standard",
        speaker: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """
        句级流式合成：按标点切句，每句用**非流式** zero-shot 推理
        （实测非流式 RTF≈0.97，CosyVoice2 流式 RTF≈5.7 慢 6 倍）。

        首句合成完立即推送，后续句在前句播放期间生成，感知延迟 ~1s。

        Yields:
            每句的完整 PCM 字节
        """
        self._ensure_model()
        loop = asyncio.get_running_loop()

        sentences = self._split_sentences(text) or [text]
        import time as _time
        logger.info(f"[TTS-TIMER] split into {len(sentences)} sentences: {[len(s) for s in sentences]}")

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=4)
        t_call = _time.perf_counter()

        def _synth_one(sent: str) -> bytes:
            """非流式合成一句。"""
            import torch
            if self._cached_spk_id:
                chunks = [c["tts_speech"] for c in self._model.inference_zero_shot(
                    sent, "", "", zero_shot_spk_id=self._cached_spk_id, stream=False,
                )]
            else:
                import os
                ref_path = os.environ.get("VOR_COSYVOICE_REFERENCE_AUDIO") \
                    or "/home/xxm/VoxCareAgent/CosyVoice/asset/zero_shot_prompt.wav"
                chunks = [c["tts_speech"] for c in self._model.inference_zero_shot(
                    sent, "希望你以后能够做的比我还好呦。", ref_path, stream=False,
                )]
            tensor = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
            return self._tensor_to_pcm(tensor)

        async def _producer():
            try:
                for idx, sent in enumerate(sentences):
                    t_s = _time.perf_counter()
                    pcm = await loop.run_in_executor(None, _synth_one, sent)
                    audio_ms = (len(pcm) // 2) / 24.0
                    wall_ms = (_time.perf_counter() - t_s) * 1000
                    rtf = wall_ms / audio_ms if audio_ms else 0
                    tag = "FIRST" if idx == 0 else f"#{idx}"
                    logger.info(
                        f"[TTS-TIMER] sent{tag} chars={len(sent)} wall={wall_ms:.0f}ms "
                        f"audio={audio_ms:.0f}ms RTF={rtf:.2f} "
                        f"from_call={(_time.perf_counter()-t_call)*1000:.0f}ms"
                    )
                    await queue.put(pcm)
            finally:
                await queue.put(None)

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
        """有缓存 spk_id 时走 zero_shot 快路径（跳过参考音频处理）。"""
        if self._cached_spk_id:
            chunks = [c["tts_speech"] for c in self._model.inference_zero_shot(
                text, "", "", zero_shot_spk_id=self._cached_spk_id, stream=False,
            )]
        else:
            import os
            ref_path = os.environ.get("VOR_COSYVOICE_REFERENCE_AUDIO") \
                or "/home/xxm/VoxCareAgent/CosyVoice/asset/zero_shot_prompt.wav"
            chunks = [c["tts_speech"] for c in self._model.inference_zero_shot(
                text, "希望你以后能够做的比我还好呦。", ref_path, stream=False,
            )]
        import torch
        audio_tensor = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
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
