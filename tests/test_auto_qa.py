"""Tests for utils/auto_qa.py — 自动质检模块

系统组件: Utils — AutoQA 自动质检引擎
源文件:   voice_optimized_rag/utils/auto_qa.py
职责:     敏感词过滤 + 违规模式拦截 + 语义一致性校验，防止幻觉输出

测试覆盖：
- 敏感词检测
- 禁止模式检测（正则）
- 正常文本通过质检
- 语义一致性检测（含 Mock Embedding）
- 多重问题同时触发
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.utils.auto_qa import AutoQA, FALLBACK_RESPONSE
from tests.conftest import MockEmbedding


class TestAutoQA:
    """自动质检引擎测试"""

    @pytest.mark.asyncio
    async def test_clean_response_passes(self):
        """正常回复应通过质检"""
        qa = AutoQA()
        result = await qa.check("您好，这款产品目前有现货，可以直接下单。")
        assert result.passed is True
        assert result.issues == []

    @pytest.mark.asyncio
    async def test_sensitive_word_detected(self):
        """包含敏感词应标记但不替换（仅敏感词无 forbidden 模式）"""
        qa = AutoQA()
        result = await qa.check("我们保证这个产品绝对不会出问题")
        assert result.passed is False
        assert any("敏感词" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_forbidden_pattern_triggers_fallback(self):
        """违规模式应触发兜底回复"""
        qa = AutoQA()
        # 匹配 "竞争对手...差/烂/垃圾"
        result = await qa.check("竞争对手的产品很差")
        assert result.passed is False
        assert result.cleaned_response == FALLBACK_RESPONSE

    @pytest.mark.asyncio
    async def test_false_promise_pattern(self):
        """虚假承诺模式应被检测"""
        qa = AutoQA()
        result = await qa.check("我们承诺三个月内回报翻倍")
        assert result.passed is False
        assert any("违规" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_consistency_check_low(self):
        """语义一致性低于阈值应标记"""
        mock_emb = MockEmbedding(dim=64)
        qa = AutoQA(embedding_provider=mock_emb, consistency_threshold=0.99)

        # 两段完全不同的文本，余弦相似度应较低
        result = await qa.check(
            "今天天气真好适合出去玩",
            source_context="本产品采用进口材质，防水等级 IP68",
        )
        # 相似度大概率低于 0.99 阈值
        assert any("语义一致性" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_consistency_no_embedding(self):
        """无 embedding provider 时应跳过一致性检查"""
        qa = AutoQA()  # 不传 embedding_provider
        result = await qa.check("回答内容", source_context="来源文档")
        # 不应因一致性检测失败
        assert not any("语义一致性" in issue for issue in result.issues)

    @pytest.mark.asyncio
    async def test_multiple_issues(self):
        """多个问题应同时报告"""
        qa = AutoQA()
        result = await qa.check("我们保证竞争对手的产品很垃圾")
        assert result.passed is False
        assert len(result.issues) >= 2  # 至少有敏感词 + 违规模式
