"""硅基流动 (SiliconFlow) LLM 提供商

硅基流动 API 兼容 OpenAI 协议，支持以下开源/闭源模型：
- DeepSeek-V3 / DeepSeek-R1 (推理增强)
- Qwen2.5-72B / Qwen2.5-Coder
- GLM-4-9B / GLM-4-Plus
- Yi-Lightning / Yi-1.5-34B
- InternLM2.5-20B
- Llama-3.3-70B

API Base: https://api.siliconflow.cn/v1
文档:     https://docs.siliconflow.cn/
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("siliconflow_llm")

# 硅基流动常用模型 ID 速查
SILICONFLOW_MODELS = {
    # DeepSeek 系列
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v2.5": "deepseek-ai/DeepSeek-V2.5",
    # Qwen 系列
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    # GLM 系列
    "glm-4-9b": "THUDM/glm-4-9b-chat",
    "glm-4-plus": "THUDM/GLM-4-Plus",
    # Yi 系列
    "yi-lightning": "01-ai/Yi-Lightning",
    # InternLM 系列
    "internlm2.5-20b": "internlm/internlm2_5-20b-chat",
    # Llama 系列
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


class SiliconFlowProvider(LLMProvider):
    """硅基流动 LLM 提供商

    通过 OpenAI 兼容协议调用硅基流动的推理服务。
    支持 generate (同步) 和 stream (流式) 两种模式。

    使用方式:
        .env 配置:
            VOR_LLM_PROVIDER=siliconflow
            VOR_LLM_API_KEY=sk-xxxxx
            VOR_LLM_MODEL=deepseek-v3       # 可用简写或完整模型 ID

        代码调用:
            provider = SiliconFlowProvider(api_key="sk-xxx", model="deepseek-v3")
            result = await provider.generate("你好")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        base_url: str = SILICONFLOW_BASE_URL,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "Install openai SDK for SiliconFlow: pip install openai"
            )

        # 优先从参数取，其次从环境变量
        key = api_key or os.environ.get("VOR_LLM_API_KEY", "") or os.environ.get("SILICONFLOW_API_KEY", "")
        if not key:
            raise ValueError(
                "SiliconFlow API key required. Set VOR_LLM_API_KEY or SILICONFLOW_API_KEY."
            )

        # 模型名简写映射
        resolved_model = SILICONFLOW_MODELS.get(model.lower(), model)

        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
        )
        self._model = resolved_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info(f"SiliconFlow provider initialized: model={self._model}")

    async def generate(self, prompt: str, context: str = "") -> str:
        """同步生成完整回复"""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(prompt, context),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """流式生成回复（逐 token 输出）"""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(prompt, context),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
