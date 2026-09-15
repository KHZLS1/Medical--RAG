"""查询改写模块

在检索前对用户 query 做增强，提升召回率。
用户提问往往口语化，先用 LLM 改写成关键词增强版再去向量检索，
让 bge embedding 能更精准命中相关问答。

策略：
  - rewrite_query: 单次改写（默认），口语化 → 关键词增强版
  - multi_query:   多路改写（可选），生成 N 个变体用于多路召回

集成位置见 rag_chain.retrieve()：
  检索用改写后的 query，Reranker 与 Prompt 用原始 question。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache

from .config import settings
from .llm import get_llm

REWRITE_PROMPT = """你是一个医学搜索查询改写助手。
请将用户的口语化问题改写为更适合向量检索的查询，要求：
1. 提取核心医学关键词（疾病、症状、药物、检查等）
2. 补充相关医学术语与同义词
3. 去除"能不能""可以吗"等无意义口语词汇
4. 保持原意不变，不引入用户未提及的新病症
5. 只输出改写后的查询，不要解释、不要加引号、不要换行

用户问题：{question}
改写后的查询："""

@lru_cache(maxsize=256)
def rewrite_query(question: str) -> str:
    """单次改写：口语化问题 → 关键词增强版

        示例：
          "高血压患者能吃党参吗" → "高血压 党参 用药禁忌 药物相互作用 安全性"
        改写失败时回退到原始 question，保证检索不中断。
        """
    if not settings.query_rewrite_enabled:
        return question

    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        return rewritten or question
    except Exception as e:
        print(f"[查询改写] 失败，回退原始问题: {e}")
        return question

MULTI_QUERY_PROMPT = """你是医学搜索助手。请为以下问题生成 {n} 个不同角度的检索查询变体，用于多路召回。
要求：
1. 每行一个变体，不要编号、不要解释
2. 角度可包括：疾病机制、用药安全、检查指标、日常注意事项等
3. 保持与原问题相关，不引入无关病症

用户问题：{question}
{n} 个检索变体："""

def multi_query(question: str, n: int = 3) -> list[str]:
    """多路改写：生成 n 个变体 query（含原始 query）用于多路召回

        示例：
          原始："高血压患者能吃党参吗"
          变体1："党参 高血压 禁忌 相互作用"
          变体2："高血压中药调理 党参安全性"
          变体3："高血压患者 人参 党参 用药注意"
        """
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "n": n})
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        queries.insert(0, question)
        return queries[:n + 1]
    except Exception as e:
        print(f"[多路改写] 失败，回退原始问题: {e}")
        return [question]

CONTEXT_REWRITE_PROMPT = """你是一个医学对话上下文理解助手。
请根据对话历史，将用户当前的问题改写为一个完整、独立的检索查询。

要求：
1. 补全省略的主语、指代（如"它""这个病""上述症状"等）
2. 结合之前讨论的疾病/症状，使改写后的问题可以独立被检索
3. 保持医学专业性，提取关键术语
4. 只输出改写后的查询，不要解释、不要加引号、不要换行

【对话历史】
{history}

【当前问题】
{question}

【改写后的查询】："""

def _format_history(history: list[dict], max_turns: int = 6) -> str:
    """将历史消息格式化为文本，只取最近 max_turns 条"""
    # 取最近 N 条（用户+助手算2条，所以6条约3轮）
    recent = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    for msg in recent:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines) if lines else "（无历史对话）"

@lru_cache(maxsize=256)
def _cached_context_rewrite(history_key: str, question: str) -> str:
    """带缓存的上下文改写：lru_cache 要求键可哈希，
    故把 history 先格式化成字符串再作为缓存键（与 Prompt 入参一致）。"""
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(CONTEXT_REWRITE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({
            "history": history_key,
            "question": question,
        }).strip()
        return rewritten or question
    except Exception as e:
        print(f"[上下文改写] 失败，回退原始问题: {e}")
        return question


def _fallback_with_context(question: str, history: list[dict] | None) -> str:
    """改写失败/退化时的兜底：用最近一轮用户主题补全当前问题。

    防止"那应该怎么做"这类指代性追问，在 LLM 改写异常时退化成裸词去检索，
    误命中"怎么治/怎么做"等无关文档（表现为多轮上下文丢失）。
    """
    if not history:
        return question
    last_user = ""
    for m in reversed(history):
        if m.get("role") == "user":
            last_user = m["content"]
            break
    topic = last_user.strip()[:40] if last_user else ""
    if not topic:
        return question
    return f"{topic} {question}".strip()


def rewrite_query_with_context(question: str, history: list[dict] | None = None) -> str:
    """结合多轮历史改写当前问题，返回可独立检索的完整 query

    Args:
        question: 当前用户问题
        history: 历史消息列表，格式 [{"role": "user", "content": "..."}, ...]

    Returns:
        改写后的完整查询字符串
    """
    if not settings.query_rewrite_enabled:
        return question

    # 没有历史或历史为空，直接走单次改写
    if not history or len(history) == 0:
        return rewrite_query(question)

    # history 统一格式化成字符串，同时作为缓存键（可哈希）与 Prompt 入参
    history_key = _format_history(history)

    # _cached_context_rewrite 内部做 lru_cache + try/except：LLM 异常时原样返回 question。
    # 若改写退化回原问题（说明未能结合上下文），用最近一轮用户主题兜底，避免裸词误检索。
    rewritten = _cached_context_rewrite(history_key, question)
    if rewritten == question:
        return _fallback_with_context(question, history)
    return rewritten


TITLE_PROMPT = """请为以下医学问答生成一个简短的对话标题（不超过15个字）。
要求：
1. 概括用户问题的核心内容（疾病、症状、药物等）
2. 简洁明了，不要加引号、不要加句号、不要换行
3. 只输出标题文本，不要任何解释

用户问题：{question}
对话标题："""


def generate_title(question: str) -> str:
    """用 LLM 生成简短对话标题，失败时回退到截取前20字"""
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(TITLE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        title = chain.invoke({"question": question}).strip()
        title = title.split("\n")[0].strip()
        if len(title) > 30:
            title = title[:30]
        return title or question[:20]
    except Exception as e:
        print(f"[标题生成] 失败，回退截取: {e}")
        return question[:20] + ("..." if len(question) > 20 else "")