"""Shared business-only prompts and responses."""

from __future__ import annotations

import re


OUT_OF_SCOPE_RESPONSE = (
    "当前系统仅支持销售、售后和财务相关业务咨询或办理，请直接描述您的业务问题。"
)

SALES_STRUCTURED_KEYWORDS = (
    "价格",
    "报价",
    "套餐",
    "方案",
    "试用",
    "折扣",
    "优惠",
    "活动",
    "购买",
    "版本",
    "升级",
)

PRODUCT_INTRO_KEYWORDS = (
    "商品",
    "产品",
    "功能",
    "特点",
    "亮点",
    "能力",
    "适用",
    "场景",
    "模块",
)

PRODUCT_CATALOG_KEYWORDS = (
    "目录",
    "清单",
    "列表",
    "产品线",
    "类目",
    "型号",
    "编号",
)

PRODUCT_LOOKUP_RE = re.compile(r"[a-z][a-z0-9_-]{2,23}", re.IGNORECASE)


def is_structured_sales_query(user_text: str) -> bool:
    normalized = user_text.lower()
    return any(keyword in normalized for keyword in SALES_STRUCTURED_KEYWORDS)


def is_product_intro_query(user_text: str) -> bool:
    normalized = user_text.lower()
    if any(subject in normalized for subject in ("产品", "商品", "模块")) and any(
        keyword in normalized for keyword in ("介绍", "了解", "功能", "特点", "亮点", "适合", "说明")
    ):
        return True
    if normalized.startswith(("介绍产品", "产品介绍", "介绍商品", "商品介绍", "给我介绍商品")):
        return True
    return any(keyword in normalized for keyword in PRODUCT_INTRO_KEYWORDS) and "介绍" in normalized


def is_product_catalog_query(user_text: str) -> bool:
    normalized = user_text.lower()
    has_subject = any(subject in normalized for subject in ("商品", "产品", "模块"))
    has_catalog = any(keyword in normalized for keyword in PRODUCT_CATALOG_KEYWORDS)
    has_list_request = any(keyword in normalized for keyword in ("有哪些", "都有什么", "全部", "全部产品"))
    return (has_subject and has_catalog) or (has_subject and has_list_request)


def is_product_lookup_query(user_text: str) -> bool:
    normalized = "".join(user_text.strip().split()).lower()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in ("型号", "编号")):
        return True
    return PRODUCT_LOOKUP_RE.fullmatch(normalized) is not None


def is_entity_explainer_query(user_text: str) -> bool:
    normalized = "".join(user_text.strip().split())
    if not normalized:
        return False
    if normalized.startswith("什么是") and len(normalized) > 3:
        return True
    if normalized.endswith("是什么") and len(normalized) > 3:
        return True
    return any(keyword in normalized for keyword in ("是做什么的", "是干嘛的", "是什么平台", "是什么产品"))


def build_structured_sales_answer_prompt(user_text: str) -> str:
    return (
        "请严格依据上下文回答这类销售咨询，并使用以下固定结构输出。\n"
        "输出结构：\n"
        "一、价格总览\n"
        "二、套餐对比\n"
        "三、试用政策\n"
        "四、折扣与优惠\n"
        "五、适用建议\n"
        "六、未提及项\n\n"
        "规则：\n"
        "1. 如果上下文里有多个价格档位或套餐，必须尽量一次列全，不要只摘第一档。\n"
        "2. 如果上下文是英文，翻译成自然、准确的中文。\n"
        "3. 对试用、折扣、年付/月付、附加费、增购包等信息优先显式列出。\n"
        "4. 某一项上下文未提及时，明确写“未提及”，不要臆造。\n\n"
        f"用户问题：{user_text}"
    )


def build_product_intro_answer_prompt(user_text: str) -> str:
    return (
        "请严格依据上下文回答这类产品介绍型咨询，并使用以下固定结构输出。\n"
        "输出结构：\n"
        "一、产品概览\n"
        "二、核心功能\n"
        "三、适用场景\n"
        "四、价格与试用\n"
        "五、推荐下一步\n"
        "六、未提及项\n\n"
        "规则：\n"
        "1. 产品介绍、功能说明、适用场景、价格套餐、库存状态、试用政策、优惠活动都属于业务范围。\n"
        "2. 用户问题即使比较笼统，例如“介绍产品”或“你们有什么产品”，也应优先结合上下文给出产品概览，不要直接拒答。\n"
        "3. 如果上下文里有多项功能或多个产品能力，尽量归纳完整，不要只摘一句。\n"
        "4. 价格、套餐、试用若上下文未提及，要明确写“未提及”，不要臆造。\n"
        "5. 只有当用户明显在闲聊、问天气、问身份、讲笑话或提出非销售/售后/财务业务请求时，才回复固定拒答话术。\n\n"
        f"用户问题：{user_text}"
    )


def build_product_catalog_answer_prompt(user_text: str) -> str:
    return (
        "请严格依据上下文回答这类商品目录或产品线咨询，并使用以下固定结构输出。\n"
        "输出结构：\n"
        "一、商品目录总览\n"
        "二、核心模块或产品线\n"
        "三、主要能力\n"
        "四、价格与试用\n"
        "五、适合谁使用\n"
        "六、未提及项\n\n"
        "规则：\n"
        "1. “商品目录”“产品目录”“你们有哪些商品”“都有什么产品”都属于销售知识咨询，不要拒答。\n"
        "2. 如果知识库里主要是平台与模块信息，应按平台+模块方式整理回答。\n"
        "3. 如上下文未出现完整目录，要明确说明当前已知的产品线或模块，不要臆造。\n"
        "4. 价格、试用、优惠若未提及，明确写“未提及”。\n\n"
        f"用户问题：{user_text}"
    )


def build_product_lookup_answer_prompt(user_text: str) -> str:
    return (
        "请把这类短商品名、型号、编号或模块代号视为销售知识查询，而不是超范围请求。\n"
        "规则：\n"
        "1. 如果上下文能匹配到对应产品或模块，直接给出简洁介绍。\n"
        "2. 如果只能匹配到相近产品，请先说明最接近的已知产品。\n"
        "3. 如果上下文无法确认精确匹配，不要拒答；请明确说明未找到精确匹配，并请用户补充更完整的商品名称或编号。\n"
        "4. 只有明显闲聊或非业务请求时，才使用固定拒答话术。\n\n"
        f"用户问题：{user_text}"
    )


def build_entity_explainer_answer_prompt(user_text: str) -> str:
    return (
        "请严格依据上下文回答这类实体说明型咨询，并使用以下固定结构输出。\n"
        "输出结构：\n"
        "一、它是什么\n"
        "二、主要定位\n"
        "三、核心能力或方案\n"
        "四、价格与购买\n"
        "五、适用对象\n"
        "六、未提及项\n\n"
        "规则：\n"
        "1. 用户问“X是什么”“什么是X”“X是做什么的”时，只要上下文里有该实体的业务资料，就必须直接解释，不要拒答。\n"
        "2. 知识库中的外部厂商、平台、SaaS 产品、竞品或参考案例资料，都属于可回答的业务知识范围。\n"
        "3. 优先先用一句话说明该实体是什么，再补充定位、核心能力、价格购买和适用对象。\n"
        "4. 如果上下文未提及价格、试用、付款方式或适用对象，要明确写“未提及”，不要臆造。\n"
        "5. 只有明显闲聊、问天气、问身份、讲笑话或与销售/售后/财务业务无关时，才回复固定拒答话术。\n\n"
        f"用户问题：{user_text}"
    )


def build_business_answer_prompt(user_text: str) -> str:
    """Build a business-only answering prompt for direct LLM generation."""
    if is_structured_sales_query(user_text):
        return build_structured_sales_answer_prompt(user_text)

    if is_entity_explainer_query(user_text):
        return build_entity_explainer_answer_prompt(user_text)

    if is_product_catalog_query(user_text):
        return build_product_catalog_answer_prompt(user_text)

    if is_product_intro_query(user_text):
        return build_product_intro_answer_prompt(user_text)

    if is_product_lookup_query(user_text):
        return build_product_lookup_answer_prompt(user_text)

    return (
        "请严格依据上下文回答销售、售后、财务相关业务内容。\n"
        "以下都属于支持范围：产品介绍、功能说明、价格套餐、库存状态、试用政策、优惠活动、订单、物流、退款、取消订单、修改地址、发票和对账。\n"
        "知识库中的外部厂商、平台、SaaS 产品、竞品资料，只要与销售、售后或财务管理相关，也属于可回答的业务知识范围。\n"
        "如果用户问题比较宽泛，但仍明显属于上述业务范围，请优先总结上下文并直接回答，不要因为问题笼统而拒答。\n"
        "如果上下文中包含英文信息，请翻译成自然、准确、简洁的中文。\n"
        f"只有当用户明显在闲聊、问天气、问身份、讲笑话，或请求与销售/售后/财务无关时，才直接回复：{OUT_OF_SCOPE_RESPONSE}\n"
        f"用户问题：{user_text}"
    )