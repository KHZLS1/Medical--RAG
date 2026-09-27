"""LangGraph 状态图：把线性 RAG 链路重构为可观测的节点图

结构
----
  完整图   build_graph():
      intent ─┬─ medical → 检索子图
              │              rewrite ─┬─ act=ack/chitchat →（子图结束）─→ chat
              │                       └─ 其余 → route → retrieve → rerank
              │                                                     └─┬─ strong → generate → ground_check（忠实性校验）
              │                                                       └─ none ─┬─ 可追问且非急症 → human_review（interrupt 暂停）
              │                                                                  │   ├─ 给了补充 → requery →(仍有证据/再问)…
              │                                                                  │   └─ 收尾/放弃 → insufficient
              │                                                                  └─ 否则 → insufficient（禁引用编号）
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
  - **指代消解（阶段三）**：`focus_entity` 是**跨轮显式状态**（与 `dialogue_act`
    同源产出、靠 checkpointer 保留），回答"我们正在聊哪个病/哪个药"。
    更新规则取自策略表 `focus` 列：仅医学类 act（new_question/followup）覆盖，
    `ack`/`chitchat` 保持（否则一句「好的」就把焦点清空 → 下一句真追问没得补）。
    使用规则极窄：**只在 `followup` 且改写退化时**才用焦点补全检索词
    （`"{焦点} {原问题}"`），两层优先级 = 显式状态 → `_focus_from_history(history)`。
    第二层是为评测路径准备的（`eval_*` 调 build_graph 不带 checkpointer，状态从空起）。
    `node_requery` 是**程序化调用**子图，必须显式把焦点传进去，否则澄清后重检索丢焦点。
  - **忠实性校验（阶段四）**：`generate → ground_check → END`。L1 用正则校验回答里的
    `[n]` 是否都能在 sources 里找到，越界编号由程序**确定性剥除**（不靠 Prompt 求模型
    守规矩）；L2（默认关）逐句核查论断是否有 context 依据，未支持则追加确定性提示。
    校验结果放 `grounding`，文本真的变了才推 `correction` 事件让前端整段替换
    （token 事件只能追加，这是前端唯一能反映"剥除"的入口）。
    ⚠️ `ground_check` **不得**加进 `_astream_run` 的 STREAMING_NODES —— L2 会调 LLM，
    放行就会把核查输出当成回答推给前端（`chat` / `insufficient` 已各踩过一次）。
  - **检查点与人工澄清（阶段二 / A3）**：evidence=none 且非急症且轮次未用完时，
    `node_human_review` 调 `interrupt()` 暂停整张图（状态由 AsyncSqliteSaver 落盘），
    SSE 以 `clarification_request` 事件收尾；下一轮请求由 `astream_resume` 以
    `Command(resume=...)` 恢复。用户的回复**一律融合**进原问题（不判"新话题还是
    补充"——改写节点会对融合后的问题重新改写，新话题信号自然占主导）。
    `node_requery` 刻意**程序化调用**检索子图（同 rag_chain.retrieve 形态），
    而不是把边连回父图 "retrieval" 子图节点——后者会触发下一条的 trace 重复坑。
  - **⚠️ 「意图判定」每轮恰好出现 1 次**：`node_rewrite` 与 `node_chat_generate`
    都写这条 trace，而 ack 掉头后两条路径都会落到 chat。所以 `node_rewrite`
    **只在真走检索时**才写 trace（ack/chitchat 时不写），由 chat 节点补写。
    `_intent_step()` 同时看 `intent` 与 `dialogue_act`，两条来源任一为会话语即算。
  - **⚠️ trace 与子图节点**：检索子图是嵌在父图里的子图节点，它返回时会把自己
    拿到的 trace（含父图在进入子图之前写入的部分）连新增一起回传，而父图 trace
    的归约器是**追加**语义 → 父图先写的内容会被重复追加一次。
    因此规则是：**父图里排在检索子图之前的节点，一律不要写 trace**。
    node_classify_intent 遵守这条，改由分支后的第一个节点补写该步（见 _intent_step）。
    node_answer_insufficient / human_review / requery 排在子图**之后**，写 trace
    不受此限（不会重复）。
    以后若在 intent 与子图之间再插节点，同样要遵守。
  - **⚠️ trace 必须按轮重置**：开了 checkpointer 后状态跨轮保留，"继承 + 追加"
    会让 trace 每轮滚雪球（第 5 轮能涨到 75 条）。由 node_classify_intent 在
    新一轮入口发 `TRACE_RESET` 哨兵清空，详见 `_TraceReset`。
  - **循环导入**：对 rag_chain 的依赖一律放在函数内部延迟导入
    （rag_chain 在模块层引用本模块时不会回环）。
  - **可序列化**：routing 以 dict 形式存 state（而非 RoutingResult 对象），
    以便未来接 checkpointer 做人工介入（A3）。
"""
import asyncio
import json
from functools import lru_cache, partial
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from .config import settings
from .query_rewriter import rewrite_for_retrieval, format_history
from .vectorstore import get_retriever
from .reranker import rerank_documents
from .agents.source_router import route_knowledge_source, filter_by_source, RoutingResult
from .dialogue_policy import NON_RETRIEVAL_ACTS, policy_for, should_update_focus

# 与 vectorstore.DEFAULT_RECALL_K 保持一致：两路检索各召回条数
DEFAULT_RECALL_K = 20


def thread_config(conversation_id: int) -> dict:
    """conversation_id → LangGraph thread 配置。同一会话多轮共用一个 thread。"""
    return {"configurable": {"thread_id": f"conv-{conversation_id}"}}


# trace 重置哨兵：新一轮入口（node_classify_intent）发它，归约器见即清空。
#
# 为什么需要它
# ------------
# trace 原本用 `operator.add` 归约器。不开 checkpointer 时每轮状态从空开始，
# "只增不减"没有副作用；**开了 checkpointer 之后**同一 thread 的 trace 跨轮
# 保留，而检索子图是父图的一个节点、会**继承**父图状态，返回时把"继承到的
# 历史 trace + 本轮新增"一起回传，追加归约器再追加一次 → 每轮滚雪球
# （实测：第 3 轮 trace 20 条，第 5 轮 75 条，前端"检索过程"面板整屏都是
# 上一轮的旧步骤）。所以新一轮开始必须显式清空。
#
# **resume 路径不经过 node_classify_intent**（图从 human_review 重入），
# 所以上一轮的「人工澄清」步骤得以保留 —— 这正是想要的：澄清与补充后的
# 重检索同属一次问答，面板该连起来看。
#
# 为什么是字符串常量而不是自定义哨兵对象
# ------------------------------------
# 节点返回值在**归约之前**会作为 pending write 交给 checkpointer 序列化
# （langgraph 的 aput_writes），自定义对象会直接报
# "Type is not msgpack serializable"（实测踩过）。字符串是 msgpack 原生类型，
# 且 trace 的元素恒为 dict，`==` 比较不可能误命中。
TRACE_RESET = "__trace_reset__"


def _merge_trace(left: list | None, right) -> list:
    """trace 通道归约器：见哨兵清空，否则追加。"""
    if right == TRACE_RESET:
        return []
    if right is None:
        return list(left or [])
    return list(left or []) + list(right)


class GraphState(TypedDict, total=False):
    # ---- 输入 ----
    question: str
    history: list[dict]
    # ---- 各步中间态（B1 前端展示用）----
    # 归约器是 _merge_trace 而非 operator.add：多了"重置"能力，见 _TraceReset
    trace: Annotated[list[dict], _merge_trace]   # 累积的步骤记录（多节点写入时自动追加）
    intent: str                          # "medical" | "chat"（方案 B，入口规则门）
    dialogue_act: str                    # 阶段一：new_question | followup | ack | chitchat
    focus_entity: str                    # 阶段三：当前讨论的医学实体（跨轮保留，靠 checkpointer）
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
    # ---- 人工澄清（阶段二 / A3）----
    clarify_rounds: int                  # 本轮已追问次数（防死循环）
    clarify_reply: str                   # 用户对追问的回复（resume 后写入）
    review_outcome: str                  # "requery"（重新检索）| "giveup"（放弃收尾）
    # ---- 忠实性校验（阶段四）----
    grounding: dict                      # 校验结论（verdict/cited/invalid/unsupported/corrected_answer）


def _make_step(name: str, detail: dict) -> dict:
    """生成一条 trace 记录，供前端时间线渲染"""
    return {"step": name, **detail}


# 急症关键词：命悬一线的问题不允许被「请补充信息」卡住，直接走 insufficient
# 引导就医（急症且有资料时根本不会走到 none 分支，此处只兜"恰好无资料"的边角）。
_EMERGENCY_HINTS: tuple[str, ...] = (
    "胸痛", "呼吸困难", "喘不上气", "气短", "昏迷", "意识丧失", "晕厥",
    "大出血", "抽搐", "惊厥", "服毒", "自杀", "药物过量", "误食",
)


def _should_ask_user(state: GraphState, clarify_ok: bool = True) -> bool:
    """证据不足时，是否中断向用户追问（三个条件都满足才问）。

    `clarify_ok` 是**这张编译好的图有没有 checkpointer**，由 build_graph 绑定。
    为什么不能只看配置：评测脚本调 build_graph() 不带 checkpointer，
    而 `graph_checkpointer_enabled` 仍是默认 True——若按配置放行，
    node_human_review 会在没有 checkpointer 的图上调 interrupt() 直接抛错，
    5 条 out_of_corpus 样本全崩。中断能力必须按实际传入的 checkpointer 判定。
    """
    return (
        clarify_ok
        and settings.human_review_enabled
        and state.get("clarify_rounds", 0) < settings.human_review_max_rounds
        and not any(h in state["question"] for h in _EMERGENCY_HINTS)
    )


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

    但它是**新一轮的唯一入口**，因此负责发 trace 重置信号：开了 checkpointer
    后状态跨轮保留，不清空就会滚雪球（详见 _TraceReset）。resume 路径从
    human_review 重入、不经过这里，上一轮的步骤得以保留。
    """
    from .intent import is_conversational   # 延迟导入，避免 import 期耦合

    reset = {"trace": TRACE_RESET}
    if settings.intent_gate_enabled and is_conversational(state["question"]):
        return {**reset, "intent": "chat"}
    return {**reset, "intent": "medical"}


def _route_by_intent(state: GraphState) -> str:
    """条件边路由：intent -> 目标节点名"""
    return "chat" if state.get("intent") == "chat" else "medical"


def _focus_from_history(history: list[dict] | None) -> str:
    """兜底焦点：最近一条**非会话语**的用户发言（截断 40 字）。

    为什么需要它（阶段三 D4 第二层）：focus_entity 是跨轮状态，靠 checkpointer 保留；
    而 eval_rag / eval_dialogue 调 build_graph() 不带 checkpointer，每轮状态从空
    开始 —— 没有这层，指代追问在评测里永远拿不到焦点，新逻辑等于测不到。
    它同时是"显式状态丢失"时的安全网。

    与 `_build_user_statement` 的关键差别：只取**最近一条**（焦点要的是"当前在聊
    什么"，不是"用户说过哪些症状"），且截断到 40 字（拼进检索词的长度预算）。
    """
    from .intent import is_conversational   # 延迟导入，复用同一套词表

    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if text and not is_conversational(text):
            return text[:40]
    return ""


def node_rewrite(state: GraphState) -> GraphState:
    """查询改写 + 对话行为判定：多轮时结合历史补全省略主语

    走 `rewrite_for_retrieval` 拿**契约对象**，而不是裸字符串——LLM 那天返回
    「（用户过于简短，无法判断具体意图）」这种说明文字时，被当检索词送进 Milvus
    的事故就是这么来的。契约保证 enhanced_query 一定能拿去检索；退化时它等于
    原问题，并把 degraded/reason 写进 trace 供观测（D1 选 A，不再拼历史主题）。

    阶段一起，**同一次调用**还产出 `dialogue_act`（见 query_rewriter）。
    act 属于 ack/chitchat 时本节点之后会掉头、不走检索，此时**刻意不写 trace**：
    那两种情况最终落到 node_chat_generate，由它写 trace，否则「意图判定」会重复。

    阶段一 §13.9 的"接线"：**自足问题（new_question）用原句检索，丢弃改写产物**。
    实测改写对自足问题是净负收益（无改写 hit_rate 0.92 vs 有改写 0.70~0.82），
    改写只服务于 followup 的指代消解 / 省略补全。改写那次 LLM 调用照做——
    act 本身就是它的副产品，跳不掉；只是产物不再送进 Milvus，故零额外延迟。

    阶段三起再顺带维护 `focus_entity`（跨轮显式状态），并在**指代追问 + 改写退化**
    时用它兜底补全检索词（见下方 focus_fallback 段）。
    """
    result = rewrite_for_retrieval(state["question"], state.get("history"))

    # 总开关关掉 → 恒按最安全的 new_question 处理（行为退回本阶段之前）
    act = result.dialogue_act if settings.dialogue_act_enabled else "new_question"

    # ---- 阶段三：焦点实体（跨轮显式状态）----
    # 只在医学类 act 上更新（策略表 focus=update），ack/chitchat 保持原值 ——
    # 否则一句「好的」就把焦点清成空，下一句真追问就没得补（D2 状态污染）。
    focus_ok = settings.focus_entity_enabled and settings.dialogue_act_enabled
    prev_focus = state.get("focus_entity") or ""
    next_focus = prev_focus
    if focus_ok and result.focus_entity and should_update_focus(act):
        next_focus = result.focus_entity

    # 检索用哪个 query 由策略表定（dialogue_policy._RETRIEVE_POLICY["retrieve_with"]）。
    # dialogue_act_enabled=False 时不启用本闸：此时 act 是恒定值而非真实分类，
    # 不能据此断言"这题自足"，须保持本阶段之前的"改写一律生效"行为。
    gate = (
        settings.rewrite_gate_on_act
        and settings.dialogue_act_enabled
        and policy_for(act).get("retrieve_with") == "question"
    )
    retrieval_query = state["question"] if gate else result.query

    # ---- 阶段三：指代追问 + 改写退化 → 用焦点补全（D4）----
    # 改写退化时 result.query 已回退成原问题，而 followup 的指代词（"它""这个病"）
    # 本身不可检索 → 召回必崩。用焦点补全是**确定性**的，不赌 LLM 当次是否听话。
    # 改写成功时不用它：实测 followup 用改写 query 检索+打分更优（策略表不等式 1）。
    # ack/chitchat 永不进入本分支：act == "followup" 这道门挡住「不了」型回归。
    focus_fallback = False
    if focus_ok and act == "followup" and result.degraded:
        focus_for_fallback = next_focus or _focus_from_history(state.get("history"))
        if focus_for_fallback:
            retrieval_query = f"{focus_for_fallback} {state['question']}"
            focus_fallback = True

    detail: dict = {"query": result.query, "act": act}
    if gate:
        # 只在闸门生效（即改写产物被丢弃）时才写这两项：默认路径下
        # "query 就是检索词"，再写一遍只是噪音。前端 trace 面板据此显示
        # 这一轮实际送进 Milvus 的是哪一句。
        detail["used"] = retrieval_query
        detail["used_from"] = "原句（自足问题，丢弃改写）"
    elif focus_fallback:
        detail["used"] = retrieval_query
        detail["used_from"] = "焦点实体 + 原问题（改写退化兜底）"
    if next_focus:
        # 空则不写，避免 trace 面板里出现一行 `focus: `（见前端 traceDetail 的铺法）
        detail["focus"] = next_focus
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
        # 更不返回 focus_entity：LangGraph 只合并返回的键，不返回即"保持原值"，
        # 这正是 D2/D3 要的语义（不要为了"显式"而写 focus_entity: prev_focus）。
        return {"dialogue_act": act}

    return {
        "enhanced_query": retrieval_query,
        "dialogue_act": act,
        "focus_entity": next_focus,
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
# 忠实性校验（阶段四）
# ============================================================================
def node_ground_check(state: GraphState) -> GraphState:
    """忠实性校验：L1 引用可校验（确定性）+ L2 逐句核查（可选）

    为什么需要 L1：MEDICAL_PROMPT 规则 9 无条件要求标引用编号。方案 ② 已把"无资料"
    整条路挪出医学 Prompt，但**有资料**时模型仍可能标出越界编号（5 条资料写 [6]）——
    那就是伪造出处。这里用程序把它确定性剥掉，让"引用可校验"成为可断言的性质。

    为什么 L2 默认关：每轮 +1~2s，且"未被支持"的判准本身有误报。先拿 L1 的 trace
    数据说话，再决定开不开（先量、再改）。

    ⚠️ 本节点**不得**加入 _astream_run 的 STREAMING_NODES：L2 会调 LLM，
    若被放行，核查输出会被当成回答推给前端（该项目已踩过两次，见模块 docstring）。

    T59 改道（自省后处理）：`top_score` 切不开"可回答/答不上来"（实测重叠），
    改为在生成侧判"模型自己有没有自认答不上来"，命中则剥掉**全部**编号 ——
    见 rag_chain.detect_unanswerable 的说明。它排在 L1 之后：L1 剥"越界"，
    本层处理"编号合法但整篇答不上来"。

    节点不写 answer_suffix：L1 的剥除与 L2 的追加提示**共用 corrected_answer
    这一个出口**，避免"correction 替换掉 suffix"的互相打架。
    """
    from .rag_chain import (   # 延迟导入，避免循环依赖
        check_citations, detect_unanswerable, strip_all_citations,
        GROUNDING_HEDGE, get_groundedness_chain, parse_unsupported,
    )

    answer = state.get("answer") or ""
    final = answer
    invalid: list[int] = []
    cited: list[int] = []
    stripped: list[int] = []
    unanswerable = ""

    if settings.groundedness_citation_check:
        # L1：越界编号（引用了不存在的资料）
        final, invalid, cited = check_citations(answer, state.get("sources"))
        # T59：生成侧自省 —— 回答自认"资料答不了"却挂着编号，那是给"我不知道"
        # 配的出处。先 L1 剥越界，本层再剥全部。
        unanswerable = detect_unanswerable(final)
        if unanswerable:
            final, stripped = strip_all_citations(final)

    if stripped:
        verdict = "unanswerable_stripped"
    elif invalid:
        verdict = "citation_fixed"
    else:
        verdict = "pass"

    unsupported: list[int] = []
    if settings.groundedness_llm_enabled and final:
        try:
            raw = get_groundedness_chain().invoke({
                "context": state.get("context") or "",
                "answer": final,
            })
            unsupported = parse_unsupported(str(raw))
        except Exception as e:
            # fail-open：校验层不参与故障传导
            print(f"[忠实性校验] L2 调用失败，跳过: {type(e).__name__}: {e}")
        if unsupported:
            final = final.rstrip() + "\n\n" + GROUNDING_HEDGE
            verdict = "unsupported" if verdict == "pass" else f"{verdict}+unsupported"

    grounding = {
        "verdict": verdict,
        "cited": cited,
        "invalid_citations": invalid,
        # T59：因"模型自认答不上来"被整篇剥除的编号（含原本合法的）
        "stripped_citations": stripped,
        "unanswerable_hit": unanswerable,
        "unsupported_sentences": unsupported,
        # 仅在**文本真的变了**时才给 corrected_answer：流式层据此决定要不要推
        # correction 事件。没变就不推 —— 保证"默认无行为变化"。
        "corrected_answer": final if final != answer else None,
    }
    return {
        "answer": final,
        "grounding": grounding,
        "trace": [_make_step("忠实性校验", {
            "verdict": verdict,
            "cited": cited or "（无）",
            "invalid": invalid or "（无）",
            # 只在真的剥了才写，避免绝大多数正常回答的 trace 多两行噪音
            **({"stripped": stripped, "hit": unanswerable} if stripped else {}),
            "unsupported": unsupported or "（无）",
        })],
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


def _route_after_retrieval(state: GraphState, clarify_ok: bool = True) -> str:
    """策略矩阵的后半张：先看对话行为（含掉头回来的），再看证据状态。

    ack / chitchat 的 answer 策略是 "chat" → 送往对话节点；
    否则按证据二档：有资料 generate，无资料时——阶段二起先过人工澄清闸
    （可追问且非急症且轮次未用完 → 中断向用户要补充信息），不满足才 insufficient。
    """
    if policy_for(state.get("dialogue_act"))["answer"] == "chat":
        return "chat"
    if state.get("evidence_state") == "none":
        return "human_review" if _should_ask_user(state, clarify_ok) else "insufficient"
    return "generate"


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
# 人工澄清（阶段二 / A3）
# ============================================================================
def node_human_review(state: GraphState) -> GraphState:
    """证据不足 → 中断图执行，向用户追问补充信息

    interrupt() 的语义：第一次执行到这里时图**暂停**（本轮 SSE 以
    clarification_request 事件收尾，状态已由 checkpointer 落盘），等下一轮
    请求以 Command(resume=...) 重入；重入时本节点**从头重跑**，interrupt()
    直接返回 resume 值。因此 interrupt() 之前不得有任何副作用。

    写 trace 安全性：本节点排在检索子图之后，且后继（requery 是程序化调用、
    insufficient 是普通节点）都不会再进入父图的 "retrieval" 子图节点，
    不触发「子图返回时父图 trace 重复追加」的坑（见模块 docstring）。
    """
    from langgraph.types import interrupt   # 延迟导入，保持 import 期轻

    fresh = interrupt({
        "type": "clarification_request",
        "message": (
            "现有医学资料中暂时没有找到与您的问题直接相关的内容。"
            "为了更准确地帮您查询，能否补充一些信息？例如：症状出现的部位、"
            "持续时间、是否伴随其他不适、既往病史。若不方便补充，回复「算了」即可。"
        ),
        "top_score": round(state.get("top_score", 0.0), 3),
    })

    # ---- 以下只在 resume 之后执行 ----
    # resume 值由 main.py 组装成 dict；脚本手工恢复时可能是裸字符串，两者都接。
    if isinstance(fresh, dict):
        text = (fresh.get("reply") or "").strip()
        fresh_history = fresh.get("history") or []
    else:
        text, fresh_history = str(fresh or "").strip(), []

    rounds = state.get("clarify_rounds", 0) + 1
    base: dict = {"clarify_rounds": rounds, "clarify_reply": text or "（未补充）"}
    if fresh_history:
        # 恢复轮刷新历史：requery 里改写节点才能看到最近几轮（含澄清气泡）
        base["history"] = fresh_history

    # 收尾/拒答类回复（「算了」「不了」…）→ 复用意图门词表，口径一致
    from .intent import is_conversational
    giveup = not text or is_conversational(text)

    if giveup:
        return {**base, "review_outcome": "giveup",
                "trace": [_make_step("人工澄清", {"reply": text or "（未补充）",
                                                  "outcome": "放弃"})]}

    # 有实质内容 → 融合进原问题重新检索（D3：融合而非替换，理由见实施计划 §1）
    return {
        **base,
        "question": f'{state["question"]}（用户补充：{text}）',
        "review_outcome": "requery",
        "trace": [_make_step("人工澄清", {"reply": text[:50], "outcome": "重新检索"})],
    }


async def node_requery(state: GraphState) -> GraphState:
    """带着用户补充重新检索。

    ⚠️ 刻意**程序化调用**检索子图（与 rag_chain.retrieve 同一形态），而不是把
    边连回父图的 "retrieval" 子图节点——后者会把父图已积累的 trace 连同子图
    新增一起回传、再被父图 add 归约器重复追加一次（见模块 docstring
    「trace 与子图节点」）。程序化调用以空 trace 起步，子图返回值即本轮全部
    新步骤，父图 extend 不重复。检索实现仍只有一份。
    """
    result = await get_retrieval_graph().ainvoke({
        "question": state["question"],
        "history": state.get("history") or [],
        "trace": [],
        # 程序化调用子图：传什么、子图就看到什么。不传 → 澄清后重检索那一轮焦点丢失。
        # （rag_chain.retrieve() 只服务 eval_rag，刻意不传，故子图里读焦点一律用 .get）
        "focus_entity": state.get("focus_entity") or "",
    })
    return {
        "docs": result.get("docs", []),
        "context": result.get("context", "（无相关资料）"),
        "sources": result.get("sources", []),
        "top_score": result.get("top_score", 0.0),
        "evidence_state": result.get("evidence_state", "none"),
        # 焦点可能在本轮被改写更新（话题切换），带回父图状态，供下一轮继续沿用
        "focus_entity": result.get("focus_entity", state.get("focus_entity") or ""),
        "trace": result.get("trace", []),
    }


def _route_after_review(state: GraphState) -> str:
    return "requery" if state.get("review_outcome") == "requery" else "insufficient"


def _route_after_requery(state: GraphState, clarify_ok: bool = True) -> str:
    """澄清后再检索：有证据就答；仍无证据时，还想再问一轮（max_rounds>1）才回
    human_review，否则到头，直接 insufficient。clarify_rounds 保证循环有界。"""
    if state.get("evidence_state") == "strong":
        return "generate"
    return "human_review" if _should_ask_user(state, clarify_ok) else "insufficient"


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


def build_graph(checkpointer=None):
    """完整图：intent 分流 → 医学走检索子图，再按证据二档分流；会话走 chat。

    阶段二新增：human_review（interrupt 追问）与 requery（带补充重检索）。
    checkpointer 由 get_graph() 按配置注入，build 本身不读配置。

    澄清闸的可用性按**实际传入的 checkpointer** 绑定到条件边上（`clarify_ok`），
    而不是读 `graph_checkpointer_enabled`：评测脚本走 build_graph() 不带
    checkpointer，若按配置放行就会在没有 checkpointer 的图上调 interrupt() 抛错。
    """
    clarify_ok = checkpointer is not None
    g = StateGraph(GraphState)
    g.add_node("intent", node_classify_intent)
    g.add_node("retrieval", build_retrieval_graph())
    g.add_node("human_review", node_human_review)
    g.add_node("requery", node_requery)
    g.add_node("generate", node_generate)
    g.add_node("ground_check", node_ground_check)
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
    # 阶段二再多一个出口：无资料但可追问且非急症 → human_review 中断向用户要补充。
    g.add_conditional_edges(
        "retrieval",
        partial(_route_after_retrieval, clarify_ok=clarify_ok),
        {
            "chat": "chat",
            "generate": "generate",
            "insufficient": "insufficient",
            "human_review": "human_review",
        },
    )
    g.add_conditional_edges(
        "human_review", _route_after_review,
        {"requery": "requery", "insufficient": "insufficient"},
    )
    g.add_conditional_edges(
        "requery",
        partial(_route_after_requery, clarify_ok=clarify_ok),
        {"generate": "generate", "human_review": "human_review",
         "insufficient": "insufficient"},
    )
    # 阶段四：generate 之后必过忠实性校验（L1 剥除越界引用 / L2 追加提示），
    # 由它统一收尾。chat / insufficient 保持直连 END：它们的 Prompt 本就禁止引用编号，
    # 接上去只是白跑一次正则（且 insufficient 的"无引用"是 eval_dialogue 的硬断言）。
    g.add_edge("generate", "ground_check")
    g.add_edge("ground_check", END)
    g.add_edge("insufficient", END)
    g.add_edge("chat", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_retrieval_graph():
    return build_retrieval_graph()


_compiled_graph = None
_graph_init_lock: asyncio.Lock | None = None


async def get_graph():
    """完整图单例（异步）。checkpointer 开启时带 AsyncSqliteSaver 编译。

    为什么是 async：AsyncSqliteSaver 构造时必须已在事件循环里（内部调
    get_running_loop），而生产链路全程走 astream——同步 SqliteSaver 的
    async 方法会直接抛 NotImplementedError（SSE 只剩 error 事件的根因）。
    lru_cache 用不了（对协程函数缓存的是 coroutine，二次 await 会炸），
    改手工单例 + 事件循环内锁。

    ⚠️ 图实例按配置缓存：改 GRAPH_CHECKPOINTER_ENABLED 等配置后必须重启
    后端（与 tuned_weights / get_retriever 同款注意事项）。
    """
    global _compiled_graph, _graph_init_lock
    if _compiled_graph is not None:
        return _compiled_graph
    if _graph_init_lock is None:
        # 检查与赋值之间无 await，事件循环内不会交错，懒建锁是安全的
        _graph_init_lock = asyncio.Lock()
    async with _graph_init_lock:
        if _compiled_graph is None:
            if settings.graph_checkpointer_enabled:
                from .checkpointer import get_checkpointer   # 延迟导入，避免 import 期耦合
                _compiled_graph = build_graph(checkpointer=await get_checkpointer())
            else:
                _compiled_graph = build_graph()
    return _compiled_graph


# ============================================================================
# 流式问答（生产入口）
# ============================================================================
async def _astream_run(graph_input, config: dict | None):
    """astream_answer / astream_resume 共用的消费循环（阶段二抽出）。

    graph_input 二选一：
      - 新一轮提问：{"question":..., "history":..., "trace":[]}
      - 恢复中断：  Command(resume={"reply":..., "history":...})
    """
    graph = await get_graph()

    trace_acc: list[dict] = []
    final_sources: list[dict] | None = None
    final_answer = ""
    final_suffix = ""
    final_grounding: dict | None = None
    emitted_tokens = False
    interrupted = False

    # 会出字的三个节点。**新增节点务必同步加进来**——漏掉的后果是它的 token
    # 全被过滤掉，只剩下面 final_answer 兜底补发：能看到完整回答、但没有流式效果，
    # 极易误判成别的问题。（"chat" 就踩过一次，insufficient 是第二次。）
    STREAMING_NODES = ("generate", "chat", "insufficient")

    async for mode, chunk in graph.astream(
        graph_input, config=config, stream_mode=["updates", "messages"]
    ):
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

        # ---- 中断：证据不足，图暂停等用户补充（阶段二 / A3）----
        # astream 在 updates 流里以 "__interrupt__" 键抛出 Interrupt 对象；
        # 此时图已暂停、状态已落盘，本轮 SSE 以 clarification_request 收尾。
        if "__interrupt__" in chunk:
            intr = chunk["__interrupt__"][0]
            yield json.dumps(
                {"type": "clarification_request", "data": intr.value},
                ensure_ascii=False,
            )
            interrupted = True
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
                # updates 模式推的是节点的**原始返回值**，入口节点发的是重置哨兵
                # （不是 list）——跳过它：那一步本来就不写 trace，也避免 extend 报错。
                steps = update["trace"]
                if isinstance(steps, list):
                    trace_acc.extend(steps)
                    yield json.dumps({"type": "trace", "data": trace_acc}, ensure_ascii=False)
            if "sources" in update:
                final_sources = update["sources"]        # 缓存，最后统一推送
            if "answer" in update:
                final_answer = update["answer"]
            if "answer_suffix" in update:
                final_suffix = update["answer_suffix"]
            if "grounding" in update:
                # 忠实性校验结论（阶段四）。注意 ground_check **不在** STREAMING_NODES：
                # 它若调了 L2，那些 token 会被过滤掉，不会混进回答。
                final_grounding = update["grounding"]

    if interrupted:
        # 图已暂停：不推 sources、不做整段兜底。
        # 前端把 clarification_request 当作本轮终点（见 Chat.tsx 的 gotClarify），
        # 若照常推 sources=[] 会与「流中断检测」逻辑打架。
        return

    # 节点在 token 流**之后**追加的确定性文本（兜底路径的免责声明）不在 messages 流里，
    # 必须在这里补一个 token 事件，否则前端只看到 LLM 正文、看不到那句话。
    # 只在已流过 token 时补：没流过时走下面的整段兜底，而整段 answer 里已含这份文本。
    if emitted_tokens and final_suffix:
        yield json.dumps({"type": "token", "data": final_suffix}, ensure_ascii=False)

    # 兜底：若 messages 模式没捕获到 token（模型未按流式返回），一次性补发完整回答，
    # 保证前端一定能拿到内容，不会出现"空回答"。
    if not emitted_tokens and final_answer:
        yield json.dumps({"type": "token", "data": final_answer}, ensure_ascii=False)

    # 引用编号被剥除 / 追加了忠实性提示 → 推 correction，前端据此替换整段回答。
    # token 事件只能追加，所以这是前端**唯一**能反映"剥除"的入口。
    # 位置在 sources 之前：它是**内容**事件，与 token 流归为一组。
    # ⚠️ main.py 必须消费它并覆盖 full_answer，否则页面显示修正版、库里存原始版。
    if final_grounding and final_grounding.get("corrected_answer"):
        yield json.dumps({"type": "correction", "data": {
            "answer": final_grounding["corrected_answer"],
            "invalid_citations": final_grounding.get("invalid_citations", []),
            "verdict": final_grounding.get("verdict", ""),
        }}, ensure_ascii=False)

    yield json.dumps({"type": "sources", "data": final_sources or []}, ensure_ascii=False)


async def astream_answer(
    question: str,
    history: list[dict] | None = None,
    config: dict | None = None,
):
    """流式问答：全程走 LangGraph（意图分流 + 检索子图 + 生成节点）

    yield 顺序：
      1. {"type": "rewrite", "data": "改写后的检索词"}   ← 仅医学路径有
      2. {"type": "trace",   "data": [累计的步骤数组]}
      3. {"type": "token",   "data": "..."}          × N
         （末了可能再补一个 token：节点在流式内容之外追加的确定性文本，
           如兜底路径的免责声明，见 graph.node_answer_insufficient 的 answer_suffix）
      4. {"type": "sources", "data": [...]}          最后一条
      3.5 {"type": "clarification_request", "data": {...}}  ← 阶段二：证据不足中断，
          出现即本轮结束（无 token / sources），等下一轮 astream_resume 恢复。
      4.5 {"type": "correction", "data": {answer, invalid_citations, verdict}}
          ← 阶段四：仅当忠实性校验**真的改了文本**（剥除越界引用 / 追加提示）才出现。
          前端据此整段替换已显示的回答；main.py 据此覆盖落库文本。
          它排在 sources 之前，但整段回答都在这条流里，顺序对落库无影响。

    trace 每次推**累计**数组，前端直接整体替换即可，无需自行拼接。
    会话语义走 chat 节点：不产生 rewrite 事件，sources 为空数组。
    医学问诊但无证据走 insufficient 节点：同样 sources 为空数组。

    checkpointer 开启时 config 必传（langgraph 要求 thread_id），
    main.py 已按 conversation_id 组装；不传会在 astream 处直接报错。
    """
    initial = {"question": question, "history": history or [], "trace": []}
    async for payload in _astream_run(initial, config):
        yield payload


async def astream_resume(user_reply: str, history: list[dict] | None, config: dict):
    """恢复被 interrupt 暂停的图：把用户这一轮的话作为 resume 值传回 human_review。

    resume 值组装成 dict 而非裸字符串：顺带把最新 history 刷进图状态，
    requery 里改写节点才能看到最近几轮（含上一轮的澄清气泡）。
    """
    from langgraph.types import Command

    cmd = Command(resume={"reply": user_reply, "history": history or []})
    async for payload in _astream_run(cmd, config):
        yield payload
