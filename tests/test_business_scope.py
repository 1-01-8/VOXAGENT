"""Tests for dialogue/business_scope.py — 业务问答模板。"""

from __future__ import annotations

from voice_optimized_rag.dialogue.business_scope import build_business_answer_prompt


def test_sales_prompt_uses_structured_sections():
    prompt = build_business_answer_prompt("你们的价格、套餐、试用和折扣是什么？")

    assert "一、价格总览" in prompt
    assert "二、套餐对比" in prompt
    assert "三、试用政策" in prompt
    assert "四、折扣与优惠" in prompt


def test_general_business_prompt_stays_generic():
    prompt = build_business_answer_prompt("退款规则是什么？")

    assert "一、价格总览" not in prompt
    assert "上下文" in prompt


def test_product_intro_prompt_uses_supported_business_guidance():
    prompt = build_business_answer_prompt("你介绍一下你们的产品")

    assert "一、产品概览" in prompt
    assert "产品介绍、功能说明、适用场景" in prompt
    assert "不要直接拒答" in prompt


def test_product_catalog_prompt_handles_goods_catalog_queries():
    prompt = build_business_answer_prompt("商品目录")

    assert "一、商品目录总览" in prompt
    assert "产品目录" in prompt


def test_product_lookup_prompt_handles_short_model_token():
    prompt = build_business_answer_prompt("nove")

    assert "短商品名、型号、编号或模块代号" in prompt
    assert "不要拒答" in prompt


def test_entity_explainer_prompt_handles_vendor_what_is_query():
    prompt = build_business_answer_prompt("Zendesk是什么")

    assert "一、它是什么" in prompt
    assert "外部厂商、平台、SaaS 产品、竞品" in prompt


def test_entity_explainer_prompt_handles_reverse_what_is_query():
    prompt = build_business_answer_prompt("什么是Zendesk")

    assert "一、它是什么" in prompt
    assert "必须直接解释，不要拒答" in prompt