"""LangChain RAG 链组装：医疗 Prompt + 检索 + DeepSeek

使用 LCEL (LangChain Expression Language) 组装流式 RAG 链。
关键点:
  - 检索结果按相关性排序后格式化为带引用编号的 context
  - Prompt 严格约束 LLM 仅基于 context 作答，禁止编造
  - 通过 astream() 支持流式输出
"""
import json
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .vectorstore import get_retriever


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


def build_rag_chain():
    """构建 LCEL RAG 链

    流程: question -> 检索 docs -> 组装 context -> Prompt -> LLM -> 字符串
    """
    retriever = get_retriever(k=5)
    llm = get_llm()

    # 自定义链: 同时输出 context 与 sources，便于接口返回引用
    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        context, sources = _format_docs_with_sources(docs)
        return {"context": context, "question": question, "_sources": sources}

    prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT)

    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 单例
_chain = None


def get_rag_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain


async def stream_answer(question: str) -> AsyncIterator[str]:
    """流式问答生成器

    yield 顺序:
      1. 多个 {"type": "token", "data": "..."} 流式文本块
      2. 最后一个 {"type": "sources", "data": [...]} 引用来源
    """
    chain = get_rag_chain()

    # 先单独跑检索拿到 sources (复用 chain 内部会重复检索，这里简化处理)
    retriever = get_retriever(k=5)
    docs = retriever.invoke(question)
    _, sources = _format_docs_with_sources(docs)

    # 流式生成回答
    async for chunk in chain.astream(question):
        yield json.dumps({"type": "token", "data": chunk}, ensure_ascii=False)

    # 最后推送来源
    yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)
