"""Task slot extraction and reuse helpers for multi-turn business flows."""

from __future__ import annotations

import re
from typing import Any

from voice_optimized_rag.dialogue.session import (
    AgentDomain,
    IntentType,
    SessionContext,
    SlotInfo,
    TaskWorkflow,
)

ORDER_ID_RE = re.compile(r"(?:订单号(?:是|为)?[:：]?\s*)?(ord[-_ ]?\d+|\d{4,})", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(1\d{10})(?!\d)")
AMOUNT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:元|块|人民币|rmb|¥|￥)", re.IGNORECASE)

REASON_HINTS = (
    "买错",
    "不想要",
    "不合适",
    "重复下单",
    "误拍",
    "拍错",
    "下错",
    "质量",
    "故障",
    "延迟",
    "与描述不符",
    "发错",
    "漏发",
    "预算",
    "不需要了",
    "信息填错",
    "地址错了",
    "搬家",
    "改派",
)

GENERIC_REFUND_UTTERANCES = {
    "我要退款",
    "申请退款",
    "退款",
    "我要退费",
    "退货退款",
}

GENERIC_CANCEL_UTTERANCES = {
    "取消订单",
    "我要取消订单",
    "取消这个订单",
    "撤销订单",
}

ADDRESS_PREFIX_RE = re.compile(
    r"(?:新地址(?:是|为)?|地址(?:改成|修改为|改为)|改到|改成)[:：]?\s*(.+)$"
)


def sync_task_slots_from_utterance(session: SessionContext, utterance: str) -> None:
    """Update session slots from the latest task utterance."""
    if session.current_intent != IntentType.TASK:
        return

    if session.active_workflow != TaskWorkflow.NONE:
        sync_workflow_slots(session, session.active_workflow, utterance)
        return

    if session.current_domain == AgentDomain.FINANCE:
        sync_workflow_slots(session, TaskWorkflow.REFUND, utterance)

    elif session.current_domain == AgentDomain.AFTER_SALES:
        _sync_common_order_slots(session, utterance)


def merge_arguments_with_session(
    session: SessionContext,
    arguments: dict[str, Any],
    parameter_names: list[str],
) -> dict[str, Any]:
    """Fill missing tool arguments from session slots collected in previous turns."""
    merged = dict(arguments)
    for name in parameter_names:
        if merged.get(name):
            continue
        slot = session.slots.get(name)
        if slot and slot.value is not None:
            merged[name] = slot.value
    return merged


def format_slot_context(session: SessionContext) -> str:
    """Render filled task slots as a short context block for the agent prompt."""
    lines: list[str] = []
    for name in sorted(session.slots.keys()):
        slot = session.slots[name]
        if slot.value is None:
            continue
        lines.append(f"- {name}: {slot.value}")
    return "\n".join(lines)


def sync_workflow_slots(session: SessionContext, workflow: TaskWorkflow, utterance: str) -> None:
    """Update slots for a concrete deterministic workflow."""
    if workflow == TaskWorkflow.REFUND:
        _ensure_finance_slots(session)
        _sync_common_order_slots(session, utterance)
        amount = extract_amount(utterance)
        if amount is not None:
            session.slots["amount"].value = amount
        reason = extract_reason_text(utterance, workflow)
        if reason:
            session.slots["reason"].value = reason
        return

    if workflow == TaskWorkflow.CANCEL_ORDER:
        session.slots.setdefault("order_id", SlotInfo(name="order_id", required=True, prompt="请提供订单号。"))
        session.slots.setdefault("reason", SlotInfo(name="reason", required=False, prompt="如方便可提供取消原因。"))
        _sync_common_order_slots(session, utterance)
        reason = extract_reason_text(utterance, workflow)
        if reason:
            session.slots["reason"].value = reason
        return

    if workflow == TaskWorkflow.UPDATE_ADDRESS:
        session.slots.setdefault("order_id", SlotInfo(name="order_id", required=True, prompt="请提供订单号。"))
        session.slots.setdefault("new_address", SlotInfo(name="new_address", required=True, prompt="请提供新的收货地址。"))
        _sync_common_order_slots(session, utterance)
        new_address = extract_new_address(utterance)
        if new_address:
            session.slots["new_address"].value = new_address


def _ensure_finance_slots(session: SessionContext) -> None:
    session.slots.setdefault(
        "order_id",
        SlotInfo(name="order_id", required=True, prompt="请提供订单号。"),
    )
    session.slots.setdefault(
        "phone",
        SlotInfo(name="phone", required=False, prompt="请提供手机号。"),
    )
    session.slots.setdefault(
        "reason",
        SlotInfo(name="reason", required=True, prompt="请提供退款原因。"),
    )
    session.slots.setdefault(
        "amount",
        SlotInfo(name="amount", required=False, prompt="如有固定退款金额请提供。"),
    )


def _sync_common_order_slots(session: SessionContext, utterance: str) -> None:
    order_id = extract_order_id(utterance)
    if order_id:
        session.slots.setdefault("order_id", SlotInfo(name="order_id", required=True))
        session.slots["order_id"].value = order_id

    phone = extract_phone(utterance)
    if phone:
        session.slots.setdefault("phone", SlotInfo(name="phone", required=False))
        session.slots["phone"].value = phone


def extract_order_id(text: str) -> str | None:
    match = ORDER_ID_RE.search(text.strip())
    return match.group(1) if match else None


def extract_phone(text: str) -> str | None:
    match = PHONE_RE.search(text)
    return match.group(1) if match else None


def extract_amount(text: str) -> float | None:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_reason_text(text: str, workflow: TaskWorkflow = TaskWorkflow.REFUND) -> str | None:
    normalized = text.strip()
    if not normalized:
        return None
    generic_utterances = GENERIC_REFUND_UTTERANCES if workflow == TaskWorkflow.REFUND else GENERIC_CANCEL_UTTERANCES
    if normalized in generic_utterances:
        return None
    if extract_phone(normalized) or ORDER_ID_RE.fullmatch(normalized):
        return None
    if (
        ("订单号" in normalized or "单号" in normalized or "手机号" in normalized)
        and not any(token in normalized for token in REASON_HINTS)
        and not re.search(r"(?:退款原因|原因|因为|由于)", normalized)
    ):
        return None

    for pattern in (
        r"(?:退款原因|原因)(?:是|为)?[:：]?\s*(.+)$",
        r"(?:因为|由于)\s*(.+)$",
    ):
        match = re.search(pattern, normalized)
        if match:
            reason = match.group(1).strip(" ：:，,。.")
            if reason:
                return reason

    if any(token in normalized for token in REASON_HINTS):
        return normalized.strip(" ：:，,。.")

    if 2 <= len(normalized) <= 24 and not extract_amount(normalized):
        return normalized.strip(" ：:，,。.")

    return None


def extract_new_address(text: str) -> str | None:
    normalized = text.strip()
    if not normalized:
        return None

    match = ADDRESS_PREFIX_RE.search(normalized)
    if match:
        address = match.group(1).strip(" ：:，,。.")
        return address or None

    if any(keyword in normalized for keyword in ("省", "市", "区", "路", "街", "号", "室", "栋", "单元")):
        if len(normalized) >= 6 and "订单号" not in normalized:
            return normalized.strip(" ：:，,。.")
    return None