"""LangChain RAG 链组装：医疗 Prompt + 检索 + DeepSeek

使用 LCEL (LangChain Expression Language) 组装流式 RAG 链。
关键点:
  - 检索结果按相关性排序后格式化为带引用编号的 context
  - Prompt 严格约束 LLM 仅基于 context 作答，禁止编造
  - 通过 astream() 支持流式输出
"""
import asyncio
import json
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .vectorstore import get_retriever
from .reranker import rerank_documents
from .query_rewriter import rewrite_query

# 医疗专属 Prompt - 关键！约束 LLM 行为
MEDICAL_PROMPT = """你是一名严谨、专业的医学助手。请仅基于【医学资料】回答用户的医疗问题。

【医学资料】
{context}

【回答规则】
1. 只能基于上述资料作答，不得使用资料外的知识，更不能编造。
2. 如果资料不足以回答问题，请直接说明"现有医学资料无法回答该问题，建议咨询执业医师"。
3. 回答应结构化、专业且通俗，尽量包含：可能的病因、建议检查、日常注意事项、是否需要就医。
4. 严禁给出具体药物剂量或处方建议；如涉及用药，提示"请遵医嘱"。
5. 若问题涉及急症（如胸痛、呼吸困难、剧烈头痛、意识丧失、大出血），优先提示"请立即拨打120或前往急诊"。
6. 回答末尾用 [1] [2] 等标注引用的资料编号。
7. 末尾必须附加免责声明："⚠️ 本回答仅基于公开医学资料供参考，不能替代执业医师诊断，请结合实际情况就医。"

【用户问题】
{question}
"""


def _format_docs_with_sources(docs: list[Document]) -> tuple[str, list[dict]]:
    """把检索到的文档格式化为带编号的 context，并返回来源元数据"""
    if not docs:
        return "（无相关资料）", []

    blocks = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        blocks.append(f"[{i}] {doc.page_content}")
        sources.append({
            "index": i,
            "department": meta.get("department", ""),
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "snippet": doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content,
        })
    return "\n\n".join(blocks), sources


def retrieve(question: str, history: list[dict] | None = None) -> tuple[list[Document], str, list[dict]]:
    """检索并格式化：上下文改写 → 向量检索 → Reranker 精排，只检索一次

    返回 (docs, context, sources)，供 stream_answer 复用。
    """
    # 查询改写：有上下文时用上下文改写，否则用普通改写
    from .query_rewriter import rewrite_query_with_context
    enhanced_query = rewrite_query_with_context(question, history)

    # 向量检索（用改写后的 query 召回 Top-20）
    retriever = get_retriever(k=20)
    docs = retriever.invoke(enhanced_query)

    # Reranker 精排（用原始 question 做精细匹配，更贴合用户真实意图）
    docs = rerank_documents(question, docs, top_k=5)

    context, sources = _format_docs_with_sources(docs)
    return docs, context, sources


def build_generation_chain():
    """构建生成链（不含检索）：{context, question} -> 文本

    检索由 stream_answer 单独调用 retrieve() 完成，避免链内部重复检索。
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT)
    return prompt | llm | StrOutputParser()


# 单例
_chain = None


def get_generation_chain():
    global _chain
    if _chain is None:
        _chain = build_generation_chain()
    return _chain


async def stream_answer(
    question: str,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """流式问答生成器

    检索只执行一次：context 喂给生成链，sources 最后推送。
    yield 顺序:
      1. 多个 {"type": "token", "data": "..."} 流式文本块
      2. 最后一个 {"type": "sources", "data": [...]} 引用来源
    """
    chain = get_generation_chain()

    # 检索一次：context 供生成，sources 供最后推送。
    # retrieve() 内部包含「LLM 查询改写 + 向量检索 + Reranker 精排」，全是同步的
    # CPU/GPU 密集调用，放进线程池执行，避免阻塞事件循环拖慢其他请求。
    _, context, sources = await asyncio.to_thread(retrieve, question, history)

    # 流式生成回答（复用同一批检索结果，不重复检索）
    async for chunk in chain.astream({"context": context, "question": question}):
        yield json.dumps({"type": "token", "data": chunk}, ensure_ascii=False)

    # 最后推送来源
    yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)
