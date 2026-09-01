"""DeepSeek LLM 客户端封装

DeepSeek 兼容 OpenAI API 协议，可直接用 langchain_openai.ChatOpenAI。
- deepseek-chat: 通用对话模型 (V3), 快且便宜
- deepseek-reasoner: 推理模型 (R1), 复杂鉴别诊断时用
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import settings


@lru_cache(maxsize=1)
def get_llm(model: str = "deepseek-chat") -> ChatOpenAI:
    """返回 DeepSeek LLM 单例

    Args:
        model: deepseek-chat (默认) 或 deepseek-reasoner
    """
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY，请在 backend/.env 中填入你的 Key "
            "(申请地址: https://platform.deepseek.com/api_keys)"
        )
    return ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com",
        temperature=0.3,    # 医疗场景需低温度，减少幻觉
        streaming=True,
        max_tokens=2048,
    )
