"""LLM 客户端封装

通过硅基流动 (SiliconFlow) 的 OpenAI 兼容接口调用模型。
默认模型由 config.llm_model 指定（.env 里 LLM_MODEL 可覆盖）。
- 当前: deepseek-ai/DeepSeek-V4-Flash
- 其他模型: 可调用 get_llm(model="...") 覆盖
- 温度: 默认 0.3；**查询改写走 0.0**（见 get_llm 的 temperature 参数）
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import settings

# 默认温度。评估指标不可复现的根因就是这个 0.3 —— 改写每跑一次关键词选择就可能不同
# （同一份代码连跑两次 hit_rate 差 4 题）。需要可复现的调用方应显式传 temperature=0。
DEFAULT_TEMPERATURE = 0.3


@lru_cache(maxsize=8)
def _build_llm(model: str, temperature: float, timeout: float) -> ChatOpenAI:
    """真正的构造点。**只接受具体值**，默认值补全在外层 get_llm 做。

    `timeout` 参与缓存键：否则改了 `settings.llm_timeout_sec` 之后仍会拿到旧客户端
    （lru_cache 只认参数）。它是本项目**唯一真正生效的超时** —— 节点级
    `TimeoutPolicy` 在 sync 节点上会被 langgraph 直接拒绝，见 config 的说明。
    """
    if not settings.llm_api_key:
        raise RuntimeError(
            "未配置 LLM_API_KEY，请在 backend/.env 中填入你的 Key"
        )
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        streaming=True,     # 前端 SSE 打字机效果需要
        max_tokens=2048,
        timeout=timeout,    # 单次请求上限，可在 .env 用 LLM_TIMEOUT_SEC 覆盖
        max_retries=1,      # 失败重试 1 次，快速暴露问题
    )


def get_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """返回 LLM 客户端

    Args:
        model: 模型名，默认取 settings.llm_model
        temperature: 采样温度，默认 DEFAULT_TEMPERATURE(0.3)。
                     关键词抽取/结构化输出这类任务应传 0.0，否则结果不可复现。

    ⚠️ 缓存键是 (model, temperature, timeout)。默认值补全必须在本函数里做，不能直接
    把 lru_cache 套在公开签名上 —— `get_llm()` 与 `get_llm(None)` 生成的缓存键不同，
    会白白多造几个客户端实例。所以这里先归一化，再交给 _build_llm。
    """
    return _build_llm(
        model or settings.llm_model,
        DEFAULT_TEMPERATURE if temperature is None else temperature,
        settings.llm_timeout_sec,
    )
