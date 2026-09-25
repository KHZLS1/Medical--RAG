"""LangChain RAG 链组装：医疗 Prompt + 生成链 + 流式转发

关键点:
  - 检索链路已迁到 LangGraph（app/graph.py），本模块只保留
    「检索结果格式化」与「生成链」，供 graph 的节点复用、也供评估脚本引用。
  - retrieve() 是 graph 检索子图的薄封装（唯一事实来源）。
  - stream_answer() 转发到 graph.astream_answer()。
  - 生成链有三条，按「证据状态 × 意图」分流（分流在 graph 里，不在这里）：
      get_generation_chain()   医学路径，有资料支撑（MEDICAL_PROMPT）
      get_chat_chain()         会话语义路径，不检索、不带引用（方案 B）
      get_insufficient_chain() 医学问诊但库中无证据 → 禁止引用编号（方案 ②）
"""
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
# 医疗专属 Prompt - 关键！约束 LLM 行为
MEDICAL_PROMPT = """你是一名严谨、专业的医学助手。请仅基于【医学资料】回答用户的医疗问题。

【医学资料】
{context}

【用户情况】（用户在本次对话中亲自描述的主诉，仅作参考；可能为空）
{user_statement}

【回答规则】
1. 注意区分症状归属：
   - 用户症状 = 仅指【用户情况】里用户亲自描述的症状/问题；
   - 【医学资料】中列举的症状 = 该疾病在资料中可能的表现，属于资料内容，
     绝不是用户本人的症状，严禁当作"患者已出现这些症状"来回答，
     也不得使用"您此前描述的/结合您出现的……"等把资料症状归给用户的措辞。
2. 若【用户情况】为空或用户未描述任何具体症状，不得从资料中搬运症状强加给用户，
   用户没提就只回答用户实际问到的内容。
3. 若【用户问题】本身不构成医疗问诊（例如只是"好的""嗯""谢谢"这类确认、寒暄、
   感谢），不要复述或展开任何疾病内容——即便上文正在讨论某个病。只用一两句话
   简短回应，并主动问一句是否还有其他想了解的。
4. 只能基于上述资料作答，不得使用资料外的知识，更不能编造。
5. 如果资料不足以回答问题，请直接说明"现有医学资料无法回答该问题，建议咨询执业医师"。
6. 回答应结构化、专业且通俗，尽量包含：可能的病因、建议检查、日常注意事项、是否需要就医。
7. 严禁给出具体药物剂量或处方建议；如涉及用药，提示"请遵医嘱"。
8. 若问题涉及急症（如胸痛、呼吸困难、剧烈头痛、意识丧失、大出血），优先提示"请立即拨打120或前往急诊"。
9. 回答末尾用 [1] [2] 等标注引用的资料编号。
10. 末尾必须附加免责声明："⚠️ 本回答仅基于公开医学资料供参考，不能替代执业医师诊断，请结合实际情况就医。"

【用户问题】
{question}
"""

# 免责声明（与 MEDICAL_PROMPT 规则 10 同文）。
# 这里刻意**不**把 MEDICAL_PROMPT 改成引用本常量：那份 Prompt 本轮冻结不动，
# 少改一处就少一个回归面。代价是同文存在两份，改文案时两处同步。
# 用法是**代码确定性追加**（answer.rstrip() + "\n\n" + DISCLAIMER），
# 不让 LLM 生成——这样"有没有免责声明"是可断言的。
DISCLAIMER = "⚠️ 本回答仅基于公开医学资料供参考，不能替代执业医师诊断，请结合实际情况就医。"


def _format_docs_with_sources(docs: list[Document]) -> tuple[str, list[dict]]:
    """把检索到的文档格式化为带编号的 context，并返回来源元数据

    注意：只取 department/title/source 三个 metadata 键，所以 reranker 写回的
    rerank_score 不会泄漏到给前端的 sources 载荷里。
    """
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
            "full_text": doc.page_content,
        })
    return "\n\n".join(blocks), sources


def retrieve(question: str, history: list[dict] | None = None) -> tuple[list[Document], str, list[dict], str]:
    """检索并格式化：委托给 LangGraph 检索子图（唯一事实来源）

    图定义见 app/graph.py：rewrite → route → retrieve → rerank
    返回 (docs, context, sources, enhanced_query)。

    注意：检索子图里**没有**意图分流节点，所以本函数对任何输入都会真的检索
    （eval_rag.py 依赖这一点）。分流只发生在 graph.astream_answer 的完整图里。

    这里刻意不再手写检索逻辑——检索链路只在 graph.py 里维护一份，
    避免"图"与"旧函数"两处逻辑漂移。
    """
    from .graph import get_retrieval_graph   # 延迟导入，避免与 graph 循环依赖

    state = get_retrieval_graph().invoke({
        "question": question,
        "history": history or [],
        "trace": [],
    })
    return (
        state.get("docs", []),
        state.get("context", "（无相关资料）"),
        state.get("sources", []),
        state.get("enhanced_query", question),
    )


def build_generation_chain():
    """构建生成链（不含检索）：{context, question, user_statement} -> 文本

    被 graph.node_generate 复用；retrieve() 由检索子图单独负责，链内不重复检索。
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


# ============================================================================
# 会话语义路径（方案 B）：不检索，直接基于历史自然回应
# ============================================================================
CHAT_PROMPT = """你是一名医学助手，正在和用户进行多轮对话。

【最近的对话】
{history}

【用户这一轮的话】
{question}

【任务】
用户这句话不构成医疗问诊（属于确认、寒暄、感谢或收尾）。请自然、简短地回应：
1. 一到两句话，不超过 40 字；
2. 不要复述、不要展开任何疾病、症状、用药或检查内容——即使上文正在讨论某个病；
3. 不要输出 [1][2] 之类的引用标注，也不要附加免责声明；
4. 如果用户是在确认或收尾，顺势问一句是否还有其他想了解的即可。

直接输出回应内容，不要任何前缀、标题或解释。"""


def build_chat_chain():
    """闲聊生成链：{history, question} -> 短回应（不检索、不带引用）"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
    return prompt | llm | StrOutputParser()


_chat_chain = None


def get_chat_chain():
    global _chat_chain
    if _chat_chain is None:
        _chat_chain = build_chat_chain()
    return _chat_chain


# ============================================================================
# 证据不足兜底路径（方案 ②）：无资料可用时的独立 Prompt
# ============================================================================
# 为什么要独立一条链，而不是复用 MEDICAL_PROMPT：
# MEDICAL_PROMPT 的规则 9/10 是**无条件**的（必须标引用编号、必须加免责声明）。
# context 为空时模型被逼着标编号，只能硬编一个 [1] —— 这就是「不了」那串假引用的来源。
# 把「无证据」整条路径挪出医学 Prompt 后，规则 9/10 就只在真有资料时生效，
# 不必去改动那两条规则（改 Prompt 的风险更大）。
INSUFFICIENT_PROMPT = """你是一名严谨、专业的医学助手。

【当前情况】
{situation}

【最近的对话】
{history}

【用户这一轮的话】
{question}

【回答要求】
1. 除【回答要求】外，你**没有任何可用资料**。严禁凭自身知识编造或补充任何疾病、
   症状、用药、检查、诊断内容。
2. 严禁输出 [1] [2] 之类的引用编号——没有资料可引，编一个编号就是伪造出处。
3. 若当前情况是"医学问诊但没有资料"：如实说明在现有医学资料中没有找到与用户问题
   相关的内容、无法给出确切回答；请用户补充更具体的信息（症状部位、持续时间、
   既往病史等），或直接咨询执业医师。
4. 无论当前情况是哪一种，只要【用户这一轮的话】本身与医学无关（例如让你写文章/写诗、
   写代码、闲聊），**都不要执行该请求**——一句话说明你只能回答医学相关问题即可，
   不要输出任何非医学的创作或建议内容。本条优先于第 3、5 条。
5. 若当前情况是"并非医学问诊"：不要提资料、检索或"无法回答"，只用一两句话自然
   回应，并顺口问一句是否还有其他想了解的。
6. 不要输出免责声明——需要时由程序在末尾统一追加。
7. 三到五句话以内。直接给出内容，不要标题、不要前缀、不要解释你的判断过程。

【回答】"""


def build_insufficient_chain():
    """证据不足兜底链：{situation, history, question} -> 文本（禁引用编号）"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(INSUFFICIENT_PROMPT)
    return prompt | llm | StrOutputParser()


_insufficient_chain = None


def get_insufficient_chain():
    global _insufficient_chain
    if _insufficient_chain is None:
        _insufficient_chain = build_insufficient_chain()
    return _insufficient_chain


async def stream_answer(
    question: str,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """流式问答生成器：全程走 LangGraph（意图分流 + 检索子图 + 生成节点）

    实现在 app.graph.astream_answer()，这里只做转发，保持 main.py 的调用方式不变。

    yield 顺序:
      1. {"type": "rewrite", "data": "..."}  改写后的检索词（仅医学路径有）
      2. {"type": "trace",   "data": [...]}  各步（累计数组）
      3. 多个 {"type": "token", "data": "..."} 流式文本块
         （末了可能再补一个 token：节点在流式内容之外追加的确定性文本，
           如兜底路径的免责声明，见 graph.node_answer_insufficient 的 answer_suffix）
      4. {"type": "sources", "data": [...]}  引用来源（最后一条；闲聊与兜底路径为空数组）
    """
    from .graph import astream_answer   # 延迟导入，避免与 graph 循环依赖

    async for payload in astream_answer(question, history):
        yield payload
