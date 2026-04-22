"""Helpers for detecting follow-up turns in multi-turn business conversations."""

from __future__ import annotations

import re

FOLLOW_UP_KEYWORDS = (
    "订单号",
    "单号",
    "手机号",
    "手机号码",
    "电话",
    "客户号",
    "会员号",
    "原因",
    "退款原因",
    "地址",
    "收货地址",
    "发票抬头",
    "税号",
    "金额",
    "验证码",
    "因为",
    "买错",
    "不想要",
    "重复下单",
)

KNOWLEDGE_FOLLOW_UP_KEYWORDS = (
    "商品",
    "产品",
    "目录",
    "清单",
    "列表",
    "产品线",
    "模块",
    "型号",
    "编号",
    "功能",
    "特点",
    "场景",
)

STRUCTURED_VALUE_RE = re.compile(
    r"(?:\b(?:ord|rf)[-_ ]?\d{3,}\b|1\d{10}|\d{4,})",
    re.IGNORECASE,
)

PRODUCT_LOOKUP_RE = re.compile(r"[a-z][a-z0-9_-]{2,23}", re.IGNORECASE)


def looks_like_task_follow_up(utterance: str) -> bool:
    """Return True when the utterance looks like slot-filling follow-up info."""
    normalized = "".join(utterance.strip().split()).lower()
    if not normalized:
        return False

    if any(keyword in normalized for keyword in FOLLOW_UP_KEYWORDS):
        return True

    return len(normalized) <= 32 and STRUCTURED_VALUE_RE.search(normalized) is not None


def looks_like_sales_knowledge_follow_up(utterance: str) -> bool:
    """Return True when the utterance looks like a short sales/product follow-up."""
    normalized = "".join(utterance.strip().split()).lower()
    if not normalized:
        return False

    if any(keyword in normalized for keyword in KNOWLEDGE_FOLLOW_UP_KEYWORDS):
        return True

    return len(normalized) <= 24 and PRODUCT_LOOKUP_RE.fullmatch(normalized) is not None