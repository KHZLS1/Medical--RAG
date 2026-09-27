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

import re

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

# L2 判定存在"未被资料支持的句子"时追加的提示（阶段四）。
# 由**代码确定性拼接**（不让 LLM 写）——与 DISCLAIMER 同一哲学：这样"有没有提示"可断言。
GROUNDING_HEDGE = (
    "⚠️ 说明：上述回答中的部分内容未能与本次检索到的资料完全对应，"
    "请以执业医师的判断为准。"
)

# 引用编号。限 1~2 位：MEDICAL_PROMPT 的 context 最多给 5 条资料（reranker_top_k），
# 两位数上限足够，且能避免把 "[2024]" 这种年份误判成编号。
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
# 剥掉编号后可能留下 " 。" —— 只收敛"紧贴中文标点/右括号"的前导空格，不碰正文缩进
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+(?=[，。；、！？：）)】」])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def check_citations(answer: str, sources: list[dict] | None) -> tuple[str, list[int], list[int]]:
    """校验回答里的引用编号是否都能在 sources 里找到对应（纯函数，零 LLM）。

    返回 (cleaned_answer, invalid, cited)：
      invalid —— 越界编号（引用了不存在的资料），已从正文剥除；
      cited   —— 出现过的全部编号（去重升序），供观测"漏标"（有资料却零引用）。

    为什么合法编号集合取自 sources 里**实际的 index** 而不是 len(sources)：
    index 是 _format_docs_with_sources 按 enumerate(start=1) 生成的，正常必然连续；
    但载荷异常时（index 缺失）退化为 1..n，避免把整篇回答的编号全判成越界。

    为什么"有 sources 但回答零引用"（漏标）不改文本：补一个引用编号等于**替模型
    编出处**，比漏标更糟；那通常是模型没遵守规则 9，属 Prompt 层问题。
    """
    srcs = sources or []
    valid = {
        int(s["index"]) for s in srcs
        if isinstance(s, dict) and str(s.get("index", "")).isdigit()
    }
    if not valid:
        valid = set(range(1, len(srcs) + 1))

    text = answer or ""
    cited = sorted({int(m.group(1)) for m in _CITE_RE.finditer(text)})
    invalid = [n for n in cited if n not in valid]
    if not invalid:
        return text, [], cited

    bad = set(invalid)
    cleaned = _CITE_RE.sub(lambda m: "" if int(m.group(1)) in bad else m.group(0), text)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned, invalid, cited


# ============================================================================
# 生成侧自省后处理（T59 改道：partial 三档的替代方案）
# ============================================================================
# 背景：`partial` 三档原计划按 top_score 区间切，实测已证伪 —— 真·可回答
# 0.658~1.000 与"库里沾边但答不了" 0.576 / 0.914 / 0.647 严重重叠，单靠分数切不开。
# 于是退一步：不问"资料够不够"，只问"**模型自己有没有说答不上来**"——这是分数之外
# 唯一确定性的信号。命中就把引用编号剥掉：编号本身真实存在（不算伪造出处），
# 但它让"答不上来"看着有出处，与项目最初那个 bug（假 [1]）同源。
#
# 保守设计（宁可漏剥，不可误剥 —— 误剥会丢掉真回答的出处）：
#   1. 只扫回答**开头** head_sentences 句。整篇答不上来的回答必然开头自陈；而
#      "该药剂量资料未提供，其余如下…"这类中段提及说明主体答得上来，不该剥。
#   2. 命中后剥**全部**编号（含合法编号），因为整篇没有可引之处。
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]+")

UNANSWERABLE_HINTS: tuple[str, ...] = (
    # 「资料」作主语 + 否定谓语
    "资料未提供", "资料中未提供", "资料里未提供",
    "资料未包含", "资料中未包含", "资料里未包含",
    "资料未涉及", "资料中未涉及",
    "资料未提及", "资料中未提及", "资料里未提及",
    "资料没有提及", "资料中没有提及", "资料里没有提及",
    "资料中没有", "资料中无", "资料里没有",
    "资料未明确", "资料中没有明确", "资料未说明", "资料中没有说明",
    "资料不足以",
    # 「找不到」类
    "未找到相关", "没有找到相关", "未检索到", "没有检索到", "未能找到",
    # 直接自陈无法作答
    "无法回答该问题", "无法回答这个问题", "无法回答上述",
    "无法给出确切", "无法确定该", "无法做出准确",
    "现有医学资料无法", "现有资料无法", "以上资料无法", "无法从资料",
)


def detect_unanswerable(answer: str, head_sentences: int = 2) -> str:
    """检测回答是否自认"现有资料答不了这个问题"（纯函数，零 LLM）。

    只扫开头 head_sentences 句，是刻意加的位置约束：整篇答不上来的回答必然开头
    就自陈，而出现在中后段的"资料未提供"通常只是某一小点的补充说明，不该据此
    剥掉整篇的引用。

    返回命中的短语（空串 = 未命中），供 trace 观测与单测断言。
    """
    text = (answer or "").strip()
    if not text:
        return ""
    head = "".join(p for p in _SENTENCE_SPLIT_RE.split(text)[:head_sentences] if p)
    if not head:
        return ""
    for hint in UNANSWERABLE_HINTS:
        if hint in head:
            return hint
    return ""


def strip_all_citations(answer: str) -> tuple[str, list[int]]:
    """剥掉回答里**全部**引用编号（含合法编号），返回 (cleaned, 被剥编号)。

    与 `check_citations` 的分工：那个只剥"sources 里不存在"的越界编号；这个剥全部，
    因为它服务的是"整篇答不上来"这个判定 —— 此时没有任何编号有真实出处。
    """
    text = answer or ""
    cited = sorted({int(m.group(1)) for m in _CITE_RE.finditer(text)})
    if not cited:
        return text, []
    cleaned = _CITE_RE.sub("", text)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned, cited


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


# ============================================================================
# 忠实性核查（阶段四 L2）：逐句核查回答里的医学论断是否有 context 依据
# ============================================================================
# 为什么默认关：每轮 +1~2s，且"未被支持"的判准本身有误报。先让 L1（确定性正则）
# 跑一段、拿到 trace 里 unsupported 的分布，确认误报率可接受再开（先量、再改）。
GROUNDEDNESS_PROMPT = """你是一个医学回答的事实核查员。

【医学资料】（编号即回答中可引用的 [n]）
{context}

【待核查的回答】
{answer}

【任务】
逐句检查【待核查的回答】中的**医学论断**（病因、症状、检查、用药、注意事项等）
是否能在【医学资料】中找到直接依据。寒暄、承接、免责声明不计入。

【输出】只输出一个 JSON 数组，元素是**不被支持**的句子序号（从 1 开始，按句子在
回答中出现的顺序）。全部都有依据时输出 []。不要输出解释、不要输出代码块。
示例：[2, 5]
"""


def build_groundedness_chain():
    """忠实性核查链：{context, answer} -> JSON 数组字符串（L2，非流式）"""
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_PROMPT)
    return prompt | llm | StrOutputParser()


_groundedness_chain = None


def get_groundedness_chain():
    global _groundedness_chain
    if _groundedness_chain is None:
        _groundedness_chain = build_groundedness_chain()
    return _groundedness_chain


def parse_unsupported(raw: str) -> list[int]:
    """把 L2 的 JSON 数组输出解析成句子序号；任何异常一律返回 []（fail-open）。

    fail-open 是刻意的：校验层是兜底，绝不能自己变成故障点——解析不了就当作
    "全部有依据"，只打日志。宁可漏报，不可误报（误报会给好回答挂上"部分内容
    未被支持"的提示，损害信任）。
    """
    import json
    try:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.strip("` \n")
            text = text.split("\n", 1)[-1] if text[:4].lower() == "json" else text
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return sorted({int(x) for x in data if str(x).strip().isdigit() and int(x) > 0})
    except Exception as e:
        print(f"[忠实性校验] 解析失败，按全部有依据处理: {type(e).__name__}: {e}")
        return []


async def stream_answer(
    question: str,
    history: list[dict] | None = None,
    config: dict | None = None,
) -> AsyncIterator[str]:
    """流式问答生成器：全程走 LangGraph（意图分流 + 检索子图 + 生成节点）

    实现在 app.graph.astream_answer()，这里只做转发，保持 main.py 的调用方式不变。
    config 为 LangGraph thread 配置（conversation_id → thread_id），checkpointer
    开启时必传，由 main.py 组装。

    yield 顺序:
      1. {"type": "rewrite", "data": "..."}  改写后的检索词（仅医学路径有）
      2. {"type": "trace",   "data": [...]}  各步（累计数组）
      3. 多个 {"type": "token", "data": "..."} 流式文本块
         （末了可能再补一个 token：节点在流式内容之外追加的确定性文本，
           如兜底路径的免责声明，见 graph.node_answer_insufficient 的 answer_suffix）
      4. {"type": "sources", "data": [...]}  引用来源（最后一条；闲聊与兜底路径为空数组）
      3.5 {"type": "clarification_request", "data": {...}}  ← 阶段二：证据不足中断，
          出现即本轮结束（无 token / sources），等下一轮恢复
      4.5 {"type": "correction", "data": {answer, invalid_citations, verdict}}
          ← 阶段四：忠实性校验改了文本才出现（剥除越界引用 / 追加提示），
          前端整段替换回答，main.py 覆盖落库文本
    """
    from .graph import astream_answer   # 延迟导入，避免与 graph 循环依赖

    async for payload in astream_answer(question, history, config=config):
        yield payload
