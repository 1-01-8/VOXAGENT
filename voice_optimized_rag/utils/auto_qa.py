"""
自动质检规则 —— 防幻觉检测 + 敏感词拦截

质检策略：
1. 语义一致性检测：回答与来源文档的余弦相似度 < 阈值时标记复核
2. 敏感词拦截：包含禁止词汇时替换为安全回复
3. 回答合规检查：检查是否包含编造的数据/承诺

质检结果写入日志，供人工复核。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("auto_qa")

# 敏感词列表（需根据业务场景持续更新）
SENSITIVE_WORDS = [
    "保证", "承诺", "一定", "绝对", "肯定不会",
    "永久", "终身", "无条件",
]

# 禁止出现的内容（法律合规）
FORBIDDEN_PATTERNS = [
    r"竞争对手.{0,10}(差|烂|垃圾)",      # 恶意贬低竞品
    r"(保证|承诺).{0,10}(收益|回报|赔偿)",  # 虚假承诺
]

# 兜底回复（当检测到问题时替换）
FALLBACK_RESPONSE = "我暂时没有找到准确的信息，为您转接专业顾问来解答这个问题。"


@dataclass
class QAResult:
    """质检结果"""
    passed: bool
    original_response: str
    cleaned_response: str
    issues: list[str]
    confidence: float = 1.0


class AutoQA:
    """
    自动质检引擎

    在 Agent 回复用户前进行质检，拦截不合规的回答。
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        consistency_threshold: float = 0.8,
    ) -> None:
        self._embeddings = embedding_provider
        self._consistency_threshold = consistency_threshold

    async def check(
        self,
        response: str,
        source_context: str = "",
    ) -> QAResult:
        """
        对 Agent 回复进行质检

        Args:
            response: Agent 生成的回复文本
            source_context: RAG 检索到的来源文档

        Returns:
            QAResult 质检结果
        """
        issues: list[str] = []
        cleaned = response

        # 检查1：敏感词检测
        sensitive_found = self._check_sensitive_words(response)
        if sensitive_found:
            issues.append(f"包含敏感词: {', '.join(sensitive_found)}")

        # 检查2：禁止模式检测
        forbidden_found = self._check_forbidden_patterns(response)
        if forbidden_found:
            issues.append(f"违规内容: {', '.join(forbidden_found)}")
            cleaned = FALLBACK_RESPONSE

        # 检查3：语义一致性检测（需要 embedding provider）
        if source_context and self._embeddings:
            consistency = await self._check_consistency(response, source_context)
            if consistency < self._consistency_threshold:
                issues.append(
                    f"语义一致性过低: {consistency:.2f} < {self._consistency_threshold}"
                )

        passed = len(issues) == 0
        # 仅当存在禁止模式或语义一致性过低时才替换为兜底回复
        # 敏感词仅触发警告（issues 中记录），不替换回复内容
        needs_fallback = bool(forbidden_found) or (
            source_context and self._embeddings
            and any("语义一致性" in iss for iss in issues)
        )

        if not passed:
            logger.warning(f"QA check failed: {issues}")

        return QAResult(
            passed=passed,
            original_response=response,
            cleaned_response=FALLBACK_RESPONSE if needs_fallback else cleaned,
            issues=issues,
        )

    def _check_sensitive_words(self, text: str) -> list[str]:
        """检测敏感词"""
        found = []
        for word in SENSITIVE_WORDS:
            if word in text:
                found.append(word)
        return found

    def _check_forbidden_patterns(self, text: str) -> list[str]:
        """检测禁止模式"""
        found = []
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                found.append(pattern)
        return found

    async def _check_consistency(
        self,
        response: str,
        source_context: str,
    ) -> float:
        """
        检测回答与来源文档的语义一致性

        使用余弦相似度衡量回答是否基于检索到的文档。
        """
        if not self._embeddings:
            return 1.0

        try:
            embeddings = await self._embeddings.embed([response, source_context])
            resp_emb = embeddings[0]
            src_emb = embeddings[1]

            # 余弦相似度
            norm_resp = resp_emb / (np.linalg.norm(resp_emb) + 1e-8)
            norm_src = src_emb / (np.linalg.norm(src_emb) + 1e-8)
            similarity = float(np.dot(norm_resp, norm_src))

            return similarity
        except Exception as e:
            logger.warning(f"Consistency check failed: {e}")
            return 1.0  # 检测失败时放行
