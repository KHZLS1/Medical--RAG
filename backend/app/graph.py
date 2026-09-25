"""LangGraph 状态图：把线性 RAG 链路重构为可观测的节点图

结构
----
  完整图   build_graph():
      intent ─┬─ medical → 检索子图
              │              rewrite ─┬─ act=ack/chitchat →（子图结束）─→ chat
              │                       └─ 其余 → route → retrieve → rerank
              │                                                     └─┬─ strong → generate（MEDICAL_PROMPT）
              │                                                       └─ none   → insufficient（禁引用编号）
              └─ chat    → 直接回应，不检索

  检索子图 build_retrieval_graph():  rewrite →(条件)→ route → retrieve → rerank

  dialogue_act 的分流判据全部取自 app/dialogue_policy.py 的策略表，
  本文件不散落字符串比较。

每步把中间态写入 GraphState（含 trace），供前端"检索过程"面板展示（配合 B1）。

设计约定
--------
  - **单一事实来源**：检索子图是唯一的检索实现，rag_chain.retrieve() 直接调用它，
    不再保留第二份手写检索逻辑，避免"图"与"旧函数"两处漂移。
    意图分流**不在**子图里——eval_rag.py 直接调子图做检索评估，不希望被
    "这题算不算医学问题"干扰。
  - **会话语义分流（方案 B）**：真实对话里用户常只回一句"好的"。若照走检索链，
    上下文改写会把它补成病症查询、无条件召回 Top-K 疾病资料，而 MEDICAL_PROMPT
    又只准用资料作答，于是把上一个病再讲一遍。故在最前面加一道纯规则判定。
  - **相关性闸门（方案 C）**：node_rerank 里，rerank 最高分低于阈值即判定库中
    无相关内容，清空 context 并令 evidence_state=none。
  - **证据二档 + 独立兜底（方案 ②）**：evidence_state 只有 strong / none 两档
    （实测分数双峰，中间是空档，按区间切 partial 永远不触发；字段留位待标定数据）。
    evidence=none 时**不再走 node_generate**——MEDICAL_PROMPT 规则 9/10 无条件要求
    "标注引用编号 + 免责声明"，context 为空时模型只能硬编一个 [1]（案例 B 的成因）。
    改走 node_answer_insufficient：独立 Prompt + 明确禁止引用编号 + 代码确定性追加
    免责声明。**MEDICAL_PROMPT 一个字没动**，它的规则 9/10 从此只在真有资料时生效。
  - **对话行为（阶段一）**：`dialogue_act` 由改写那一次 LLM 调用顺带产出
    （见 `query_rewriter._parse_rewrite_output`），零额外延迟。`ack` / `chitchat`
    在检索子图内提前结束、落到 chat；`followup` 用 `enhanced_query` 打分。
    这两条**必须同时成立** —— 否则「不了」被改写补全成病症查询后，
    "改用改写打分"会让它高分命中病症资料，原始 bug 比现在更隐蔽。
    详见 `dialogue_policy` 模块 docstring。
  - **⚠️ 「意图判定」每轮恰好出现 1 次**：`node_rewrite` 与 `node_chat_generate`
    都写这条 trace，而 ack 掉头后两条路径都会落到 chat。所以 `node_rewrite`
    **只在真走检索时**才写 trace（ack/chitchat 时不写），由 chat 节点补写。
    `_intent_step()` 同时看 `intent` 与 `dialogue_act`，两条来源任一为会话语即算。
  - **⚠️ trace 与子图节点**：检索子图是嵌在父图里的子图节点，它返回时会把自己
    拿到的 trace（含父图在进入子图之前写入的部分）连新增一起回传，而父图 trace
    用的是 `Annotated[list, add]` 归约器 → 父图先写的内容会被重复追加一次。
    因此规则是：**父图里排在检索子图之前的节点，一律不要写 trace**。
    node_classify_intent 遵守这条，改由分支后的第一个节点补写该步（见 _intent_step）。
    node_answer_insufficient 排在子图**之后**，写 trace 不受此限（不会重复）。
    以后若在 intent 与子图之间再插节点，同样要遵守。
  - **循环导入**：对 rag_chain 的依赖一律放在函数内部延迟导入
    （rag_chain 在模块层引用本模块时不会回环）。
  - **可序列化**：routing 以 dict 形式存 state（而非 RoutingResult 对象），
    以便未来接 checkpointer 做人工介入（A3）。
"""
import json
from functools import lru_cache
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from .config import settings
from .query_rewriter import rewrite_for_retrieval, format_history
from .vectorstore import get_retriever
from .reranker import rerank_documents
from .agents.source_router import route_knowledge_source, filter_by_source, RoutingResult
from .dialogue_policy import NON_RETRIEVAL_ACTS, policy_for

# 与 vectorstore.DEFAULT_RECALL_K 保持一致：两路检索各召回条数
DEFAULT_RECALL_K = 20


class GraphState(TypedDict, total=False):
    # ---- 输入 ----
    question: str
    history: list[dict]
    # ---- 各步中间态（B1 前端展示用）----
    trace: Annotated[list[dict], add]   # 累积的步骤记录（多节点写入时自动追加）
    intent: str                          # "medical" | "chat"（方案 B，入口规则门）
    dialogue_act: str                    # 阶段一：new_question | followup | ack | chitchat
    enhanced_query: str
    routing: dict
    docs: list
    context: str
    sources: list[dict]
    # ---- 证据判定（方案 ②）----
    top_score: float                     # rerank 最高分（sigmoid 后，落在 (0,1)）
    evidence_state: str                  # "strong" | "none"（"partial" 留位待标定）
    answer: str
    answer_suffix: str                   # 节点在流式内容之后追加的确定性文本（免责声明）


def _make_step(name: str, detail: dict) -> dict:
    """生成一条 trace 记录，供前端时间线渲染"""
    return {"step": name, **detail}


def _intent_step(state: GraphState) -> dict:
    """生成「意图判定」这条 trace。

    由**分支后的第一个节点**（node_rewrite / node_chat_generate）负责写入，
    而不是由 node_classify_intent 自己写——原因见模块 docstring 里
    "trace 与子图节点" 那一段：父图在子图之前写的 trace 会被重复追加一次。

    判为"会话语义"有两个来源，任一成立即算：
      1. 入口规则门（`intent == "chat"`）—— 零成本，只拦明显的；
      2. 改写节点的对话行为分类（`dialogue_act` 属于不检索的那几类）—— 兜住漏词。
    两者结果是同一件事，所以展示上合并成一条，避免面板里出现两个"意图判定"。
    """
    is_chat = (
        state.get("intent") == "chat"
        or state.get("dialogue_act") in NON_RETRIEVAL_ACTS
    )
    return _make_step("意图判定", {
        "intent": "会话语义（跳过检索）" if is_chat else "医学问诊",
    })


# ============================================================================
# 节点
# ============================================================================
def node_classify_intent(state: GraphState) -> GraphState:
    """判定这轮发言是「医学问诊」还是「会话语义」，决定要不要检索（方案 B）

    放在链路最前面，纯规则、零 LLM 调用（多一次调用就是每轮 +1~2s）。
    判不准时一律按 medical 走——宁可多检索一次，也不要漏答真问题。

    ⚠️ 本节点**刻意不写 trace**（否则面板里会重复出现一次），见 _intent_step。
    """
    from .intent import is_conversational   # 延迟导入，避免 import 期耦合

    if settings.intent_gate_enabled and is_conversational(state["question"]):
        return {"intent": "chat"}
    return {"intent": "medical"}


def _route_by_intent(state: GraphState) -> str:
    """条件边路由：intent -> 目标节点名"""
    return "chat" if state.get("intent") == "chat" else "medical"


def node_rewrite(state: GraphState) -> GraphState:
    """查询改写 + 对话行为判定：多轮时结合历史补全省略主语

    走 `rewrite_for_retrieval` 拿**契约对象**，而不是裸字符串——LLM 那天返回
    「（用户过于简短，无法判断具体意图）」这种说明文字时，被当检索词送进 Milvus
    的事故就是这么来的。契约保证 enhanced_query 一定能拿去检索；退化时它等于
    原问题，并把 degraded/reason 写进 trace 供观测（D1 选 A，不再拼历史主题）。

    阶段一起，**同一次调用**还产出 `dialogue_act`（见 query_rewriter）。
    act 属于 ack/chitchat 时本节点之后会掉头、不走检索，此时**刻意不写 trace**：
    那两种情况最终落到 node_chat_generate，由它写 trace，否则「意图判定」会重复。
    """
    result = rewrite_for_retrieval(state["question"], state.get("history"))

    # 总开关关掉 → 恒按最安全的 new_question 处理（行为退回本阶段之前）
    act = result.dialogue_act if settings.dialogue_act_enabled else "new_question"

    detail: dict = {"query": result.query, "act": act}
    if result.degraded:
        # 只在退化时才带这两个字段：前端 trace 面板会把所有非空键铺成
        # "key: value"，正常情况多两个 false/空值只是噪音。
        detail["degraded"] = True
        detail["reason"] = result.reason

    if act in NON_RETRIEVAL_ACTS:
        # 掉头路径：不写 trace（留给 chat 节点）、**不产出 enhanced_query**。
        # 后者是刻意的：astream_answer 见 enhanced_query 就推一个 rewrite 事件，
        # 而"这轮压根没检索"却报一条"改写后的检索词"是自相矛盾的观测。
        # 行为标签本身由 chat 节点写进 trace，信息不丢。
        # 也不产生 context/sources（下游 chat 节点会显式置空）。
        return {"dialogue_act": act}

    return {
        "enhanced_query": result.query,
        "dialogue_act": act,
        "trace": [
            _intent_step(state),     # 补写意图判定（本节点是医学路径的 trace 起点）
            _make_step("查询改写", detail),
        ],
    }


def node_route(state: GraphState) -> GraphState:
    """知识来源路由：判定查通用库还是上传文档

    注意：filename 必须一并存入 routing，否则下游 filter_by_source 无法
    按文件名二次收窄（会把其它上传文档的召回混进来）。
    """
    routing = route_knowledge_source(state["question"])
    return {
        "routing": {
            "source": routing.source,
            "confidence": routing.confidence,
            "filename": routing.filename,
        },
        "trace": [_make_step("来源路由", {
            "source": routing.source,
            "confidence": routing.confidence,
            "filename": routing.filename or "（未指定）",
            "reason": routing.reason,
        })],
    }


def node_retrieve(state: GraphState) -> GraphState:
    """混合检索：服务端 BM25 + 稠密向量，召回落入 docs"""
    retriever = get_retriever(k=DEFAULT_RECALL_K)
    docs = retriever.invoke(state["enhanced_query"])
    return {
        "docs": docs,
        "trace": [_make_step("混合检索", {"recalled": len(docs)})],
    }


def node_rerank(state: GraphState) -> GraphState:
    """来源过滤 + Reranker 精排 + 相关性闸门 + 证据判定 + 格式化 context/sources"""
    routing = RoutingResult(**state["routing"])

    filtered = filter_by_source(state["docs"], routing)
    # 安全阀：过滤后为空且路由高置信时回退为不过滤，避免误判把文档滤光
    if not filtered and routing.source in ("uploaded", "general") and routing.confidence >= 0.6:
        filtered = state["docs"]

    # Top-K 读配置（settings.reranker_top_k），不再硬编码，改 .env 即生效
    top_k = settings.reranker_top_k

    # 打分用的 query 按对话行为分流（策略表见 dialogue_policy）：
    #   followup → 用 enhanced_query：指代被补全后才与文档可比。
    #              这是修「它有什么副作用」top_score=0.355 漏答的关键。
    #   其余     → 用原始 question：这是「不了」没有回归的**真正原因** ——
    #              改写 LLM 自己会用历史把「不了」补成病症查询，
    #              若换成改写 query 打分就会高分命中「小儿发热」。不能动。
    is_followup = (
        settings.followup_score_with_enhanced
        and state.get("dialogue_act") == "followup"
    )
    score_query = state["enhanced_query"] if is_followup else state["question"]
    docs = rerank_documents(score_query, filtered, top_k=top_k)

    # ---- 方案 C：相关性闸门 ----
    # rerank_score 来自 reranker 里 compute_score(normalize=True)，已过 sigmoid、
    # 落在 (0,1)。最高分都低于阈值 → 判定"库里确实没有相关内容"，清空 context。
    top_score = max((d.metadata.get("rerank_score", 0.0) for d in docs), default=0.0)
    gated = (
        settings.relevance_gate_enabled
        and len(docs) > 0
        and top_score < settings.rerank_score_threshold
    )
    if gated:
        docs = []

    # ---- 方案 ②：证据二档 ----
    # 闸门清空后 docs 为空（含"本来就没召回"的情况）→ none；否则 strong。
    # 只做两档是因为实测分数是双峰（0.999 / 0.022 / 0.005 / 0.000），中间是空档，
    # 按区间切的 partial 永远不触发。真要做 partial 得改成"覆盖度"定义
    # （Top-K 中过阈值文档占比），且需要一批"医学相关但库中无精确匹配"的标定数据。
    # "partial" 的位置预留在本字段，等数据齐了再加。
    evidence_state = "strong" if docs else "none"

    from .rag_chain import _format_docs_with_sources   # 延迟导入，避免循环依赖
    context, sources = _format_docs_with_sources(docs)
    return {
        "docs": docs,
        "context": context,
        "sources": sources,
        "top_score": top_score,
        "evidence_state": evidence_state,
        "trace": [_make_step("精排", {
            "candidates": len(filtered),
            "top_k": len(docs),
            # top_score 写进 trace 是刻意的：前端"检索过程"面板会显示它，
            # 不必另写脚本就能拿到分数分布来标定 C 的阈值。
            "top_score": round(top_score, 3),
            # scored_by 同样给观测用：A/B「followup 是否改用改写打分」时，
            # 面板上一眼就能看出这一轮用的是哪个 query。
            "scored_by": "改写 query" if is_followup else "原句",
            "gate": "低于阈值 → 置空" if gated else "通过",
            "evidence": evidence_state,
        })],
    }


def _build_user_statement(history: list[dict] | None) -> str:
    """方案 D：把历史 user 发言拼成【用户情况】，但剔除确认/寒暄类。

    原实现把所有历史 user 消息一并拼入，于是"好的""嗯"也被当成主诉，
    导致 MEDICAL_PROMPT 规则 2 的前提（【用户情况】为空则不得搬运症状）
    永远不成立，模型会以为用户一直在描述同一个病、于是反复复述。
    """
    from .intent import is_conversational   # 延迟导入，复用 B 的词表

    users = [
        (m.get("content") or "").strip()
        for m in (history or [])
        if m.get("role") == "user"
    ]
    complaints = [u for u in users if u and not is_conversational(u)]
    return "\n".join(complaints)[:800] or "（无）"


def node_generate(state: GraphState) -> GraphState:
    """基于 context 生成回答（非流式；流式见 astream_answer）"""
    from .rag_chain import get_generation_chain   # 延迟导入，避免循环依赖

    chain = get_generation_chain()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"],
        "user_statement": _build_user_statement(state.get("history")),
    })
    return {"answer": answer}


def node_chat_generate(state: GraphState) -> GraphState:
    """会话语义路径（方案 B）：不检索、不精排，直接基于历史自然回应"""
    from .rag_chain import get_chat_chain   # 延迟导入，避免循环依赖

    chain = get_chat_chain()
    answer = chain.invoke({
        "history": format_history(state.get("history") or []),
        "question": state["question"],
    })
    return {
        "answer": answer,
        "context": "",
        "sources": [],     # 显式置空：闲聊回答不该带引用来源
        "trace": [
            # 补写意图判定（本节点是**所有** chat 汇聚路径的 trace 起点：
            # 入口规则门直连、以及 ack/chitchat 从检索子图掉头回来）
            _intent_step(state),
            _make_step("对话式回答", {
                "act": state.get("dialogue_act", "—"),
                "retrieval": "已跳过",
            }),
        ],
    }


# ============================================================================
# 条件边路由（判据一律取自 dialogue_policy 的策略表）
# ============================================================================
def _route_after_rewrite(state: GraphState) -> str:
    """改写之后：ack / chitchat 在**检索前**掉头。

    子图走到 END 即结束，控制权交回父图，由 _route_after_retrieval 接住送往 chat。
    （已实测：子图内条件边直接 END 后，父图能正常按状态继续路由。）
    """
    return "skip" if state.get("dialogue_act") in NON_RETRIEVAL_ACTS else "continue"


def _route_after_retrieval(state: GraphState) -> str:
    """策略矩阵的后半张：先看对话行为（含掉头回来的），再看证据状态。

    ack / chitchat 的 answer 策略是 "chat" → 送往对话节点；
    否则按证据二档：有资料 generate，无资料 insufficient。
    """
    if policy_for(state.get("dialogue_act"))["answer"] == "chat":
        return "chat"
    return "insufficient" if state.get("evidence_state") == "none" else "generate"


def node_answer_insufficient(state: GraphState) -> GraphState:
    """医学问诊但库中无证据 → 独立兜底，**禁止引用编号**

    为什么不复用 node_generate：MEDICAL_PROMPT 规则 9/10 是无条件的
    （必须标引用编号、必须加免责声明）。context 为空时模型被逼着标编号，
    只能硬编一个 [1] —— 案例 B 那串假引用就是这么来的。
    把这条路径整条挪出医学 Prompt，规则 9/10 就只在真有资料时生效。

    `situation` 由 Python 侧拼好传给 Prompt，不让 LLM 自己判断该走哪条；
    `state["intent"]` 目前只可能是 "medical"（chat 早就被入口分流走了），
    但保留另一分支是为阶段一的分类器预留——那时会有会话误判进医学路径。

    免责声明**由代码确定性追加**（不交给 LLM），这样"有没有声明"是可断言的。
    """
    from .rag_chain import get_insufficient_chain, DISCLAIMER   # 延迟导入

    is_medical = state.get("intent") != "chat"
    situation = (
        "用户提出了一个医学问题，但知识库中没有检索到任何可支撑回答的资料。"
        if is_medical
        else "用户这句话并不构成医学问诊（属于确认、寒暄、感谢或收尾）。"
    )

    chain = get_insufficient_chain()
    answer = chain.invoke({
        "situation": situation,
        "history": format_history(state.get("history") or []),
        "question": state["question"],
    })
    answer = answer.strip()

    # 免责声明由代码确定性拼上（不让 LLM 写，这样"有没有声明"是可断言的）。
    #
    # ⚠️ 但**流式路径必须单独补发它**：前端拿到的是 messages 模式流出的 LLM token，
    # 而这句话是在 LLM 流完之后才拼的，不在 token 流里 —— 只写进 answer 的话，
    # graph.invoke 看得到、前端看不到。所以这里额外声明 answer_suffix，
    # 由 astream_answer 在 token 流结束后补一个 token 事件（见那里的注释）。
    suffix = ""
    if is_medical:
        suffix = "\n\n" + DISCLAIMER
        answer = answer.rstrip() + suffix

    return {
        "answer": answer,
        "answer_suffix": suffix,   # 供流式层补发，不落前端展示
        "context": "",        # 显式置空（精排那步给的是"（无相关资料）"占位符）
        "sources": [],        # 没有资料就没有来源，前端不该显示引用卡片
        "trace": [_make_step("证据不足兜底", {
            "top_score": round(state.get("top_score", 0.0), 3),
            "retrieval": "无可用资料 → 不作答医学内容",
        })],
    }


# ============================================================================
# 图构建
# ============================================================================
def build_retrieval_graph():
    """检索子图：rewrite →(条件)→ route → retrieve → rerank

    这是检索链路的**唯一实现**，rag_chain.retrieve() 也调用它。
    ack / chitchat 会让子图在 rewrite 之后直接结束（不检索），
    此时 docs / context / sources 都未产出 —— 调用方需容忍缺省值。
    """
    g = StateGraph(GraphState)
    g.add_node("rewrite", node_rewrite)
    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)

    g.set_entry_point("rewrite")
    # 阶段一：ack / chitchat 在检索前掉头（子图直接结束，由父图接住送往 chat）
    g.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {"skip": END, "continue": "route"},
    )
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", END)
    return g.compile()


def build_graph():
    """完整图：intent 分流 → 医学走检索子图，再按证据二档分流；会话走 chat"""
    g = StateGraph(GraphState)
    g.add_node("intent", node_classify_intent)
    g.add_node("retrieval", build_retrieval_graph())
    g.add_node("generate", node_generate)
    g.add_node("insufficient", node_answer_insufficient)
    g.add_node("chat", node_chat_generate)

    g.set_entry_point("intent")
    g.add_conditional_edges(
        "intent",
        _route_by_intent,
        {"medical": "retrieval", "chat": "chat"},
    )
    # 检索完**不再**直连 generate：无资料时必须换掉 Prompt，否则规则 9/10 会逼出假引用。
    # 阶段一起多一个出口：ack / chitchat 在子图内提前结束时，这里接住送往 chat。
    g.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {"chat": "chat", "generate": "generate", "insufficient": "insufficient"},
    )
    g.add_edge("generate", END)
    g.add_edge("insufficient", END)
    g.add_edge("chat", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_retrieval_graph():
    return build_retrieval_graph()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


# ============================================================================
# 流式问答（生产入口）
# ============================================================================
async def astream_answer(question: str, history: list[dict] | None = None):
    """流式问答：全程走 LangGraph，yield 与旧 stream_answer 兼容的 SSE 载荷，并多推 trace。

    yield 顺序：
      1. {"type": "rewrite", "data": "改写后的检索词"}   ← 仅医学路径有
      2. {"type": "trace",   "data": [累计的步骤数组]}
      3. {"type": "token",   "data": "..."}          × N
      3.5 {"type": "token",  "data": "代码追加的声明"}  ← 仅兜底路径（answer_suffix）
      4. {"type": "sources", "data": [...]}          最后一条

    trace 每次推**累计**数组，前端直接整体替换即可，无需自行拼接。
    会话语义走 chat 节点：不产生 rewrite 事件，sources 为空数组。
    医学问诊但无证据走 insufficient 节点：同样 sources 为空数组。
    """
    graph = get_graph()
    initial = {"question": question, "history": history or [], "trace": []}

    trace_acc: list[dict] = []
    final_sources: list[dict] | None = None
    final_answer = ""
    final_suffix = ""
    emitted_tokens = False

    # 会出字的三个节点。**新增节点务必同步加进来**——漏掉的后果是它的 token 全被
    # 过滤掉，只剩下面 final_answer 兜底补发：能看到完整回答、但没有流式效果，
    # 极易误判成别的问题。（"chat" 就踩过一次，insufficient 是第二次。）
    STREAMING_NODES = ("generate", "chat", "insufficient")

    async for mode, chunk in graph.astream(initial, stream_mode=["updates", "messages"]):
        # ---- 模型 token 流 ----
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") not in STREAMING_NODES:
                continue
            text = getattr(msg, "content", "") or ""
            if text:
                emitted_tokens = True
                yield json.dumps({"type": "token", "data": text}, ensure_ascii=False)
            continue

        # ---- 节点状态更新 ----
        if not isinstance(chunk, dict):
            continue
        for _node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            if "enhanced_query" in update:
                yield json.dumps(
                    {"type": "rewrite", "data": update["enhanced_query"]},
                    ensure_ascii=False,
                )
            if "trace" in update:
                trace_acc.extend(update["trace"])
                yield json.dumps({"type": "trace", "data": trace_acc}, ensure_ascii=False)
            if "sources" in update:
                final_sources = update["sources"]        # 缓存，最后统一推送
            if "answer" in update:
                final_answer = update["answer"]
            if "answer_suffix" in update:
                final_suffix = update["answer_suffix"]

    # 节点在 token 流**之后**追加的确定性文本（兜底路径的免责声明）不在 messages 流里，
    # 必须在这里补一个 token 事件，否则前端只看到 LLM 正文、看不到那句话。
    # 只在已流过 token 时补：没流过时走下面的整段兜底，而整段 answer 里已含这份文本。
    if emitted_tokens and final_suffix:
        yield json.dumps({"type": "token", "data": final_suffix}, ensure_ascii=False)

    # 兜底：若 messages 模式没捕获到 token（模型未按流式返回），一次性补发完整回答，
    # 保证前端一定能拿到内容，不会出现"空回答"。
    if not emitted_tokens and final_answer:
        yield json.dumps({"type": "token", "data": final_answer}, ensure_ascii=False)

    yield json.dumps({"type": "sources", "data": final_sources or []}, ensure_ascii=False)
