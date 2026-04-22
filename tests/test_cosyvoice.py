"""Tests for voice/cosyvoice.py — CosyVoice TTS 模块

系统组件: Voice — CosyVoice TTS 语音合成
源文件:   voice_optimized_rag/voice/cosyvoice.py
职责:     CosyVoice 2 文本转语音（standard/clone/instruct 三模式）

测试覆盖：
- _tensor_to_pcm 静态方法（tensor → 16-bit PCM 转换）
- synthesize 模式分发逻辑（standard/clone/instruct）
- clone 模式缺少参考音频时的异常处理
- set_reference_audio

注意：不加载实际模型，仅测试工具方法和参数验证。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.skip("本地 CosyVoice TTS 部署已按要求停用，此测试文件整体跳过。", allow_module_level=True)

from voice_optimized_rag.voice.cosyvoice import CosyVoiceTTS


@pytest.fixture
def tts() -> CosyVoiceTTS:
    """创建 TTS 实例但不加载模型"""
    instance = CosyVoiceTTS.__new__(CosyVoiceTTS)
    instance._model_id = "test"
    instance._device = "cpu"
    instance._default_speaker = "中文女"
    instance._model = None
    instance._reference_audio = None
    return instance


class TestCosyVoicePCM:
    """PCM 转换方法测试"""

    def test_tensor_to_pcm_numpy(self):
        """numpy 数组应正确转换为 16-bit PCM"""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = CosyVoiceTTS._tensor_to_pcm(audio)
        assert isinstance(pcm, bytes)
        # 5 个 int16 样本 = 10 字节
        assert len(pcm) == 10

        # 验证转换精度
        restored = np.frombuffer(pcm, dtype=np.int16)
        assert restored[0] == 0
        assert restored[1] == 16384  # 0.5 * 32768
        assert restored[3] == 32767  # 1.0 被 clip 到 32767

    def test_tensor_to_pcm_mock_tensor(self):
        """模拟 PyTorch tensor（有 .numpy() 方法）"""

        class FakeTensor:
            def numpy(self):
                return np.array([0.25, -0.25], dtype=np.float32)

        pcm = CosyVoiceTTS._tensor_to_pcm(FakeTensor())
        assert len(pcm) == 4  # 2 samples * 2 bytes

    def test_tensor_to_pcm_clipping(self):
        """超出 [-1, 1] 范围的值应被裁剪"""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        pcm = CosyVoiceTTS._tensor_to_pcm(audio)
        restored = np.frombuffer(pcm, dtype=np.int16)
        assert restored[0] == 32767   # clipped to max
        assert restored[1] == -32768  # clipped to min


class TestCosyVoiceParams:
    """参数验证和模式分发测试"""

    def test_set_reference_audio(self, tts: CosyVoiceTTS):
        """设置参考音频"""
        tts.set_reference_audio(b"\x00" * 1000)
        assert tts._reference_audio is not None
        assert len(tts._reference_audio) == 1000

    @pytest.mark.asyncio
    async def test_clone_without_reference_raises(self, tts: CosyVoiceTTS):
        """克隆模式缺少参考音频应抛出 ValueError"""
        # 手动设置 _model 以跳过 _ensure_model
        tts._model = "fake_model"
        with pytest.raises(ValueError, match="reference_audio"):
            await tts.synthesize("你好", mode="clone")

    def test_resample_passthrough(self):
        """简化版 resample 应直接返回原数组"""
        audio = np.zeros(100, dtype=np.float32)
        result = CosyVoiceTTS._resample_if_needed(audio, 16000)
        assert np.array_equal(result, audio)
