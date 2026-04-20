"""Tests for voice/sensevoice_stt.py — SenseVoice 解析逻辑

系统组件: Voice — SenseVoice STT 语音识别
源文件:   voice_optimized_rag/voice/sensevoice_stt.py
职责:     SenseVoice-Small 推理结果解析（文本+情绪+事件标签提取）

测试覆盖：
- _parse_result 标签解析（语言、情绪、事件）
- 空结果处理
- 情绪标签映射（EMOTION_MAP）
- last_result 属性

注意：不加载实际模型（Mock _model），仅测试解析逻辑。
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.voice.sensevoice_stt import SenseVoiceSTT, SenseVoiceResult, EMOTION_MAP


@pytest.fixture
def stt() -> SenseVoiceSTT:
    """创建 STT 实例但不加载模型"""
    instance = SenseVoiceSTT.__new__(SenseVoiceSTT)
    instance._model_id = "test"
    instance._device = "cpu"
    instance._model = None
    instance._last_result = None
    return instance


class TestSenseVoiceParsing:
    """SenseVoice 结果解析测试"""

    def test_parse_standard_format(self, stt: SenseVoiceSTT):
        """标准 SenseVoice 输出格式解析"""
        raw = [{"text": "<|zh|><|ANGRY|><|Speech|><|woitn|>我的订单怎么还没到"}]
        result = stt._parse_result(raw)
        assert result.text == "我的订单怎么还没到"
        assert result.emotion == "angry"
        assert result.language == "zh"

    def test_parse_neutral_emotion(self, stt: SenseVoiceSTT):
        """中性情绪解析"""
        raw = [{"text": "<|zh|><|NEUTRAL|><|Speech|>你好请问有什么可以帮您"}]
        result = stt._parse_result(raw)
        assert result.emotion == "neutral"
        assert result.text == "你好请问有什么可以帮您"

    def test_parse_english(self, stt: SenseVoiceSTT):
        """英文语言标签"""
        raw = [{"text": "<|en|><|HAPPY|><|Speech|>Thank you very much"}]
        result = stt._parse_result(raw)
        assert result.language == "en"
        assert result.emotion == "happy"

    def test_parse_empty_result(self, stt: SenseVoiceSTT):
        """空结果应返回默认值"""
        result = stt._parse_result([])
        assert result.text == ""
        assert result.emotion == "neutral"

    def test_parse_none_result(self, stt: SenseVoiceSTT):
        """None 输入应返回默认值"""
        result = stt._parse_result(None)
        assert result.text == ""

    def test_parse_no_tags(self, stt: SenseVoiceSTT):
        """纯文本无标签应正常返回"""
        raw = [{"text": "纯文本内容"}]
        result = stt._parse_result(raw)
        assert result.text == "纯文本内容"
        assert result.emotion == "neutral"

    def test_emotion_map_emojis(self):
        """Emoji 到标准标签的映射应正确"""
        assert EMOTION_MAP["😊"] == "happy"
        assert EMOTION_MAP["😠"] == "angry"
        assert EMOTION_MAP["😐"] == "neutral"

    def test_last_result_property(self, stt: SenseVoiceSTT):
        """last_result 应返回最近一次结果"""
        assert stt.last_result is None
        stt._last_result = SenseVoiceResult(text="test", emotion="happy")
        assert stt.last_result.emotion == "happy"
