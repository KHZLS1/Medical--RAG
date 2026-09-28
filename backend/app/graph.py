"""LangGraph 状态图：把线性 RAG 链路重构为可观测的节点图

结构
----
  完整图   build_graph():
      intent ─┬─ medical → 检索子图 → tool（确定性计算，⑤ 默认关）→ grade（证据分级）
              │                                  ├─ chat（子图内 ack/chitchat 掉头回来的）
              │                                  ├─ strong（或有工具结果）→ generate ─┬─ 正常 → ground_check
              │                                  │                                    └─ 降级 → END（见 ⑧）
              │                                  └─ none ─┬─ 可追问且非急症 → human_review（interrupt）
              │                                           │     ├─ 给了补充 → 回边到检索子图 ⟲
              │                                           │     └─ 收尾/放弃 → insufficient
              │                                           └─ 否则 → insufficient（禁引用编号）
              └─ chat    → 直接回应，不检索

  ⚠️ ④ 的证据分级**没有**回边（2026-09-28 实测删除：分级 Prompt 看不到上一次的检索词，
     重检必然撞词、纯浪费）。澄清重入那条 `human_review → 检索子图` 与它无关，照旧保留。

  检索子图 build_retrieval_graph():  rewrite →(条件)→ route → retrieve → rerank
     ⚠️ 带 `input_schema` / `output_schema`，进出口是契约，见 RetrievalInput。

  dialogue_act 的分流判据全部取自 app/dialogue_policy.py 的策略表，
  本文件不散落字符串比较。

每步把中间态写入 GraphState（含 trace），供前端"检索过程"面板展示（配合 B1）。

设计约定
--------
  - **单一事实来源**：检索子图是唯一的检索实现，rag_chain.retrieve() 直接调用它，
    不再保留第二份手写检索逻辑，避免"图"与"旧函数"两处漂移。
    意图分流**不在**子图里——eval_rag.py 直接调子图做检索评估，不希望被
    "这题算不算医学问题"干扰。同理，**证据分级也不在子图里**：它是对话层的判断，
    而 eval_rag 要的是纯检索指标。
  - **会话语义分流（方案 B）**：真实对话里用户常只回一句"好的"。若照走检索链，
    上下文改写会把它补成病症查询、无条件召回 Top-K 疾病资料，而 MEDICAL_PROMPT
    又只准用资料作答，于是把上一个病再讲一遍。故在最前面加一道纯规则判定。
  - **相关性闸门（方案 C）**：node_rerank 里，rerank 最高分低于阈值即判定库中
    无相关内容，清空 context 并令 evidence_state=none。
  - **证据二档 + 独立兜底（方案 ②）**：evidence_state 只有 strong / none 两档
    （实测分数双峰，中间是空档，按区间切 partial 永远不触发）。
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
    `focus_entity` 在 `RetrievalInput` 里，子图直接继承父图状态即可拿到，
    不再需要"程序化调用时手工传焦点"那种写法。
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
    恢复后经一条普通边重入检索子图；`clarify_rounds` 已 +1，所以仍无证据时
    `_should_ask_user` 会因轮次用尽而不再追问 ⇒ 循环天然有界。
  - **① 子图进出口契约**：检索子图声明了 `input_schema` / `output_schema`，
    进出口都是显式映射（见 `RetrievalInput` / `RetrievalOutput`）。这**解除了**
    原先那条"父图里排在检索子图之前的节点一律不许写 trace"的约束 —— 根因是
    子图会继承父图状态、返回时连继承到的 trace 一起交回、被父图的追加归约器
    再追加一次。把 trace 挡在 `input_schema` 外，子图内部恒从空起步，问题消失。
    ⚠️ **别再把 trace 加回 `RetrievalInput`**：加回去就立刻退回重复追加。
  - **④ 证据分级**：`grade` 在检索子图之后，判"这批资料够不够回答"。判不足时
    **把 `evidence_state` 改写成 `none`**，让既有的无资料路径原样接管（先追问、
    再兜底）—— 判据见下文 `node_grade`。
    ⚠️ 这里**刻意没有回边**（即"换个检索词回到检索子图再检一次"）。2026-09-28 实测
    把它删了，理由写在下文 `node_grade` 的 docstring 里（一句话：那把分级 Prompt
    看不到上一次用的检索词 ⇒ 必然与改写节点撞词 ⇒ 回边拿不到新词，纯浪费一轮）。
    总开关默认关（每轮 +1 次 LLM 调用，而 provider 尾延迟是本项目最大噪声源）。
  - **⑧ 节点级可靠性**：每个节点按类别挂 `retry_policy`，按**节点名**挂
    `error_handler`，划分与理由见文件下半部分 `_NODE_KINDS` 那一段。三个要点：
    ① **流式节点不挂重试**（重试会把失败那次已经吐出去的 token 和重试的拼在一起）；
    ② **降级 handler 若要续跑，必须返回 `Command(goto=...)`** —— 它被调度成 PUSH
    任务、不触发失败节点的出边，返回普通 dict 会让流程静默停在那里；
    ③ **不挂 `timeout`**：langgraph 的节点级超时只支持 async 节点，本项目全是
    sync ⇒ 一挂上 `build_graph()` 就抛 ValueError、后端起不来。要压单轮墙钟得用
    **客户端级**超时 `LLM_TIMEOUT_SEC`（见 config / llm.py）。
    有了它，任何单点失败都不再让整轮变成空响应。
  - **⚠️ 「意图判定」每轮恰好出现 1 次**：`node_rewrite` 与 `node_chat_generate`
    都写这条 trace，而 ack 掉头后两条路径都会落到 chat。所以 `node_rewrite`
    **只在真走检索时**才写 trace（ack/chitchat 时不写），由 chat 节点补写。
    `_intent_step()` 同时看 `intent` 与 `dialogue_act`，两条来源任一为会话语即算。
    ⚠️ 澄清重入子图的那一轮（`review_outcome="requery"`）也**不写**这条 ——
    入口判定一轮只做一次。
    `eval_dialogue.py` 把它当硬断言，改 trace 相关逻辑务必跑一遍。
  - **⚠️ trace 必须按轮重置**：开了 checkpointer 后状态跨轮保留，"继承 + 追加"
    会让 trace 每轮滚雪球（第 5 轮能涨到 75 条）。由 node_classify_intent 在
    新一轮入口发 `TRACE_RESET` 哨兵清空，详见 `_merge_trace`。
    ⚠️ 同一个入口还负责清 `grade` / `answer_degraded` 这两个"本轮内有效"的字段。
    **以后往 GraphState 加任何只在单轮内有意义的字段，都要回到那里加一行**：
    这类字段残留会导致下一轮静默走错分支，是静默错误里最贵的一类。
  - **循环导入**：对 rag_chain 的依赖一律放在函数内部延迟导入
    （rag_chain 在模块层引用本模块时不会回环）。
  - **可序列化**：routing 以 dict 形式存 state（而非 RoutingResult 对象）。
"""
import asyncio
import json
from functools import lru_cache, partial
from typing import TypedDict, Annotated

from langgraph.errors import NodeError
from langgraph.graph import StateGraph, END
from langgraph.types import Command, RetryPolicy

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
    # ---- 证据分级（T62-④）----
    # ⚠️ 没有 `grade_retry`：④ 的回边已于 2026-09-28 实测删除（拿不到新词，见 node_grade）。
    grade: dict                          # {"verdict": sufficient|insufficient, "reason": str}
    # ---- 节点级降级（T62-⑧）----
    # ⚠️ 语义要**精确到"回答是降级产出的"**，不能写成"本轮有节点降级"。
    # 后者会让 `_route_after_generate` 在"重试轮 rewrite 降级了一下、生成其实完全
    # 正常"的时候误跳过忠实性校验 —— 静默少做一层校验，最难发现的那类回归。
    # 至于"这一轮到底降级过没有"给观测用的信息，trace 里的「节点降级」步骤已经有了，
    # 不必再存一份状态（存了反而要维护两处一致性）。
    answer_degraded: bool               # 本轮回答由降级 handler 给出（仅流式 LLM 节点会置位）
    # ---- 确定性医学计算（T62-⑤）----
    # {"tool", "args", "summary", "data", "text"}；没算就是 {}。
    # ⚠️ **必须按轮重置**（见 node_classify_intent）—— 否则上一轮算出的 eGFR 会让
    # 这一轮"库里没资料但有工具结果"，路由直接放行到 generate，答非所问。
    tool_result: dict
    # ---- 跨会话长期记忆（T62-⑦）----
    # long_term_memory：注入生成层【用户情况】的长期病史段落（本轮读到的）。
    # memory_written：本轮**新写入**的记忆条目（已存在的会被 key 去重掉）。
    # ⚠️ 两者都必须按轮重置，理由同 tool_result。
    long_term_memory: str
    memory_written: list


class RetrievalInput(TypedDict, total=False):
    """检索子图的**入口契约**（T62-①）。

    ⚠️ 刻意**不含 `trace`** —— 这是本契约存在的主要理由，不是漏写。
    子图是父图的一个节点，它会**继承**父图状态，返回时把自己最终状态整体交回，
    而父图 trace 的归约器是追加语义 ⇒ 父图在进入子图**之前**写的步骤被追加两次
    （实测：PRE 出现 2 次）。把 trace 挡在门外，子图内部的 trace 就恒从空起步，
    交回的就恰好是"本轮新增"，追加一次正好。
    ⚠️ 改这里之前先读 `_merge_trace` 与 `_intent_step` 的注释，它们是同一件事的
    另外两个切面；三者必须一起理解，否则很容易"修好一处、坏掉另一处"。

    `enhanced_query` 在**入**口也要有：澄清重入（`review_outcome="requery"`）会重入本子图，
    且它在子图外的入口节点不写 trace，所以检索词得由外面带进来。
    （原先还有个 `grade_retry`，是给 ④ 回边用的；回边已删除，字段一并去掉。）
    """

    question: str
    history: list[dict]
    focus_entity: str
    enhanced_query: str


class RetrievalOutput(TypedDict, total=False):
    """检索子图的**出口契约**（T62-①）。

    出口不做显式映射也能跑（子图会把继承到的字段一起交回），显式声明是为了
    **让漏写变成可见的**：以后往子图加字段时，如果忘了加在这里，父图拿到的就是
    缺失值而不是旧值，问题会立刻暴露；不声明则表现为"悄悄用了上一轮的值"。

    对照下游读取点逐项核对过：`build_graph` 的路由（evidence_state）、
    `astream_answer` 的事件（enhanced_query / sources / trace）、`node_generate`
    （context）、`rag_chain.retrieve()`（docs / context / sources / enhanced_query）。
    `routing` **刻意不在出口**：它只服务子图内部的 `node_rerank`，
    父图与任何调用方都不读它（已全仓核对）。
    """

    docs: list
    context: str
    sources: list
    top_score: float
    evidence_state: str
    enhanced_query: str
    dialogue_act: str
    focus_entity: str
    trace: list


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
# 节点级可靠性与预算（T62-⑧）
# ============================================================================
# 要解决的问题（README「已知限制」里那条）：provider 抽一下，整轮就变成空响应
# —— 前端「⚠️ 响应流已中断，未收到任何内容」，后端「回答为空，跳过落库」。
# 根因是重试压在 llm.py 的 client 层：它只知道"这次 HTTP 调用失败"，不知道
# "这个节点在图里干什么"，所以只能一律重试、也一律把异常抛出去，图没有接管的
# 余地。上移到节点层后，每个节点有自己的预算与**确定的降级动作**，单点失败不再
# 等于整轮失败 —— 这也是"图"相对"一条链"最容易讲清楚的一处收益。
#
# 节点分三类，划法来自实测而非照搬文档：
#   llm_stream  流式 LLM 节点（generate / chat / insufficient）
#               **不挂 retry_policy**：stream_mode="messages" 会把失败那次尝试
#               已经吐出的 token 一并推给前端，重试再吐一遍 ⇒ 用户看到两段拼接
#               的残句。实测确认：异常发生在首个 token 之前时安全，吐了一半再
#               失败就拼接。故只挂超时 + error_handler。
#   llm_plain   非流式 LLM 节点（rewrite / ground_check / grade）
#               不产出面向用户的 token，重试对外不可见 ⇒ 可以重试。
#   tool        确定性 / IO 节点（route / retrieve / rerank）
#               同上可重试；Milvus 与 reranker 的瞬时失败重试一次通常就好了。
#
# ⚠️ `intent` / `human_review` **刻意不在表里**：前者是纯规则、重试没有意义；
# 后者是 interrupt 节点，重试会让"暂停"的语义变得无法解释。
#
# ⚠️ 降级 handler 按**节点名**分派（见 _DEGRADE_HANDLERS），不按这里的类别：
# 同一个 llm_plain 里 rewrite / grade / ground_check 的降级目标各不相同。
# 另外 error_handler 的返回值有讲究 —— 它被调度成 PUSH 任务、**不触发失败节点的
# 出边**，需要续跑就得返回 `Command(goto=...)`（实测见 _degrade_rewrite 的注释）。
_KIND_LLM_STREAM = "llm_stream"
_KIND_LLM_PLAIN = "llm_plain"
_KIND_TOOL = "tool"

_NODE_KINDS: dict[str, str] = {
    "rewrite": _KIND_LLM_PLAIN,
    "route": _KIND_TOOL,
    "retrieve": _KIND_TOOL,
    "rerank": _KIND_TOOL,
    "tool": _KIND_TOOL,
    "recall": _KIND_TOOL,
    "grade": _KIND_LLM_PLAIN,
    "generate": _KIND_LLM_STREAM,
    "chat": _KIND_LLM_STREAM,
    "insufficient": _KIND_LLM_STREAM,
    "ground_check": _KIND_LLM_PLAIN,
    "remember": _KIND_LLM_PLAIN,
}

# 降级文案由**代码确定性给出**，不让 LLM 写 —— 与 DISCLAIMER / GROUNDING_HEDGE
# 同一哲学：这样"降级时用户看到了什么"是可断言的。
# ⚠️ 措辞刻意避开 UNANSWERABLE_HINTS 里的任何短语（"资料未提供""无法回答"…）：
# 降级 ≠ 资料答不了。必须让用户知道"是服务出了问题、可以重试"，否则兜底文案会被
# ground_check 的自省层误判成"模型自认答不上来"、把编号再剥一遍，语义就拧了。
DEGRADED_LLM_ANSWER = (
    "抱歉，本次回答没有成功生成（服务端暂时不可用或响应超时）。请稍后重试；"
    "若情况紧急，请立即拨打 120 或前往急诊。"
)
DEGRADED_CHAT_ANSWER = "抱歉，我这边刚才没能回应过来，请再说一次。"
DEGRADED_INSUFFICIENT_ANSWER = (
    "抱歉，本次查询没有完成（服务端暂时不可用或响应超时），暂时无法判断资料中"
    "是否有相关内容。请稍后重试。"
)


def _retry_policy_for(kind: str) -> RetryPolicy | None:
    """按节点类别给出重试策略。流式 LLM 节点恒为 None（见上方实测说明）。"""
    if not settings.node_retry_enabled or kind == _KIND_LLM_STREAM:
        return None
    return RetryPolicy(
        max_attempts=settings.node_retry_max_attempts,
        initial_interval=settings.node_retry_initial_interval,
        backoff_factor=settings.node_retry_backoff_factor,
        max_interval=settings.node_retry_max_interval,
    )


# ⚠️ 这里**刻意没有** `_timeout_for()`：langgraph 1.2.11 的节点级 `TimeoutPolicy`
# 只支持 async 节点 —— `validate_timeout_supported` 在 `compile()` 阶段直接抛
# ValueError："sync Python execution cannot be safely cancelled in-process"，
# 而本项目所有节点都是 `def`（sync）⇒ 一旦挂上，`build_graph()` 抛错、**后端整个起不来**。
# 2026-09-28 实测踩到（打开旋钮即崩，默认值 0 恰好掩盖）。
# 能真的掐断请求的是**客户端级**超时：`settings.llm_timeout_sec`（见 llm.py）。
# 别把它加回来，除非先把节点改成 async。


def _degrade_step(error: NodeError, action: str) -> dict:
    """降级也要留痕：前端「检索过程」面板要能看出这轮是降级跑完的。"""
    return _make_step("节点降级", {
        "node": error.node,
        "error": type(error.error).__name__,
        "action": action,
    })


def _degrade_llm_stream(state: GraphState, error: NodeError) -> dict:
    """流式 LLM 节点失败后兜底：给一段确定性文案，不让用户拿到空回答。

    ⚠️ 为什么要写 `grounding.corrected_answer`：若失败发生在吐了一半 token
    **之后**，那些 token 前端已经显示了，而 `answer` 走的是"没捕获到 token 才
    整段补发"的兜底分支 ⇒ 用户会看到半句残文，降级文案永远到不了屏幕上。
    复用 `correction` 事件（前端整段替换、main.py 覆盖落库文本）是唯一已经打通
    的出口，故这里制造一次"修正"。配套地 `_route_after_generate` 会把 generate
    的后继直接指向 END，避免 ground_check 用自己的 grounding 覆盖掉这份修正。
    """
    from .rag_chain import DISCLAIMER      # 延迟导入，避免循环依赖

    node = error.node
    if node == "chat":
        answer, action = DEGRADED_CHAT_ANSWER, "确定性告知（不检索、无引用）"
    elif node == "insufficient":
        answer = DEGRADED_INSUFFICIENT_ANSWER + "\n\n" + DISCLAIMER
        action = "确定性告知 + 免责声明"
    else:
        answer, action = DEGRADED_LLM_ANSWER, "确定性告知"

    print(f"[节点降级] {node} 失败: {type(error.error).__name__}: {error.error}")
    return {
        "answer": answer,
        "answer_suffix": "",     # 整段走 correction，不要再叠一层 token 补发
        "context": "",
        "sources": [],
        "answer_degraded": True,
        "grounding": {
            "verdict": "node_degraded",
            "cited": [],
            "invalid_citations": [],
            "unsupported_sentences": [],
            "corrected_answer": answer,
        },
        "trace": [_degrade_step(error, action)],
    }


def _degrade_rewrite(state: GraphState, error: NodeError) -> Command:
    """改写失败 → 退化成"用原句检索"，并**继续**走完检索链路。

    有损但安全：改写只服务于把省略/指代补全，退化后仍能检索到东西 ——
    `RewriteResult` 的 degraded 路径本来就是这么设计的，这里只是让它多覆盖
    "连 LLM 都调不通"这一种情形。act 一并退回 `new_question`
    （与 `dialogue_act_enabled=False` 同款最保守档）。

    ⚠️ 必须返回 `Command(goto="route")` 而不是普通 dict。实测（langgraph 1.2.11）：
    error_handler 被调度成一个 **PUSH 任务**（`prepare_node_error_handler_task` 里
    `langgraph_triggers: PUSH_TRIGGER`），**不会**触发失败节点的出边 ——
    返回普通 dict 的话子图到此为止，`route/retrieve/rerank` 全都不跑。
    后果不是报错而是**静默错**：`context` 缺失，于是 `node_generate` 抛 KeyError，
    又被它自己的降级 handler 接住，用户看到"服务暂时不可用"，而真正的原因是
    改写挂了一下下。所以这里必须显式指明续跑目标。

    `enhanced_query` 必须显式写：下游 `node_retrieve` 是按 `state["enhanced_query"]`
    取值（方括号访问），不写就是 KeyError。
    同时补写「意图判定」：本 handler 取代了 `node_rewrite` 的返回值，而那条 trace
    本来该由它写；漏写会让 `eval_dialogue` 的"每轮恰好 1 次"不变量变成 0 次。
    """
    print(f"[节点降级] rewrite 失败: {type(error.error).__name__}: {error.error}")
    return Command(goto="route", update={
        "dialogue_act": "new_question",
        "enhanced_query": state.get("question") or "",
        "trace": [_intent_step(state), _degrade_step(error, "退回原句检索（不做改写）")],
    })


def _degrade_route(state: GraphState, error: NodeError) -> Command:
    """来源路由失败 → 退化成"不做来源过滤"，继续检索。

    `routing` 必须在 `node_rerank` 之前补齐（那里是 `RoutingResult(**state["routing"])`）。
    auto + 0.5 恰好命中 `filter_by_source` 的"不过滤"分支，语义上正是
    "没判出来就别过滤" —— 比强行猜一个来源安全。
    """
    print(f"[节点降级] route 失败: {type(error.error).__name__}: {error.error}")
    return Command(goto="retrieve", update={
        "routing": {
            "source": "auto", "confidence": 0.5, "filename": None,
            "reason": "节点降级：来源路由失败，不做来源过滤",
        },
        "trace": [_degrade_step(error, "跳过来源过滤，继续检索")],
    })


def _degrade_recall(state: GraphState, error: NodeError) -> dict:
    """召回/精排失败 → 退化成"没有资料"，交给既有的无资料路径收尾。

    为什么敢这样退：`evidence_state="none"` 是本项目**已经打磨过的一条完整路径**
    （独立 Prompt、禁引用编号、代码确定性追加免责声明、且有硬断言覆盖）。
    检索挂掉与"库里没有"对用户是同一件事：现在拿不出有资料支撑的回答。
    复用它的好处是降级路径也是被测试走过的那条，不必新造一条没人走过的分支。

    刻意返回普通 dict（**不走 `Command(goto=)`**）：子图到此结束正是想要的 ——
    后面没有"再试一次"的余地，让父图接住去走无资料分支。
    """
    print(f"[节点降级] {error.node} 失败: {type(error.error).__name__}: {error.error}")
    return {
        "docs": [],
        "context": "",
        "sources": [],
        "top_score": 0.0,
        "evidence_state": "none",
        "trace": [_degrade_step(error, "退化为『无可用资料』")],
    }


def _degrade_grade(state: GraphState, error: NodeError) -> Command:
    """证据分级失败 → fail-open：当作"资料够用"，**直接去生成**。

    为什么目标可以写死成 `generate`：`node_grade` 的两道守卫保证"真干了活"的
    前提是 act 为医学类且 `evidence_state == "strong"`，而 strong 在
    `_route_after_retrieval` 里的下一步恒为 generate。所以这里不需要重算路由，
    也不该重算 —— 重算就得把 `clarify_ok` 带进来，而 handler 拿不到它。
    在守卫之前抛异常是不可能的（`policy_for` 是纯查表）。

    必须用 `Command(goto=)`：否则分级节点一挂，整轮直接结束、连回答都没有。
    """
    print(f"[节点降级] grade 失败: {type(error.error).__name__}: {error.error}")
    return Command(goto="generate", update={
        "grade": {"verdict": "sufficient", "reason": "分级层异常，按资料够用处理"},
        "trace": [_degrade_step(error, "按『资料够用』处理，继续生成")],
    })


def _degrade_ground_check(state: GraphState, error: NodeError) -> dict:
    """忠实性校验失败 → 跳过校验，直接收尾。

    返回空 state 的普通 dict 即可：`ground_check → END`，不收尾也是收到同一个地方。
    刻意**不改 answer**，也**不补 grounding** —— 校验层是最后一道兜底，
    它挂掉时正确做法是"放原始回答过去"，而不是把回答换成别的。
    """
    print(f"[节点降级] ground_check 失败: {type(error.error).__name__}: {error.error}")
    return {"trace": [_degrade_step(error, "跳过忠实性校验")]}


# 按**节点名**分派，不是按类别：同一个类别（llm_plain）里 rewrite / grade /
# ground_check 的降级目标各不相同，按类别分派只能三选一，必错两个。
_DEGRADE_HANDLERS = {
    "rewrite": _degrade_rewrite,
    "route": _degrade_route,
    "retrieve": _degrade_recall,
    "rerank": _degrade_recall,
    "grade": _degrade_grade,
    "generate": _degrade_llm_stream,
    "chat": _degrade_llm_stream,
    "insufficient": _degrade_llm_stream,
    "ground_check": _degrade_ground_check,
}


def _node_opts(name: str) -> dict:
    """节点名 → `add_node` 的可靠性参数。表里没有的节点一个参数都不加。

    ⚠️ 是"没登记就没有保护"，不是"默认给一份保守保护"：`human_review` 这类
    节点一旦被套上重试或降级，暂停语义就不可解释了。新增节点时请显式登记意图。
    """
    kind = _NODE_KINDS.get(name)
    if kind is None:
        return {}
    opts: dict = {}
    policy = _retry_policy_for(kind)
    if policy is not None:
        opts["retry_policy"] = policy
    handler = _DEGRADE_HANDLERS.get(name)
    if handler is not None:
        opts["error_handler"] = handler
    return opts


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

    ⚠️ 它同时负责重置**其它跨轮状态**（④⑧⑤ 引入的 grade / answer_degraded / tool_result）。
    这三个都必须按轮清零，理由各不同但后果一样 —— 上一轮的值活到下一轮：
      · `grade` 残留 ⇒ 诊断面板会把上一轮的分级结论当成这一轮的；
      · `answer_degraded` 残留 ⇒ `_route_after_generate` 会跳过忠实性校验；
      · `tool_result` 残留 ⇒ 这轮明明没算，路由却按"有工具结果"放行 generate，答非所问。
    这是"跨轮状态必须显式重置"的第四次踩点（前三次见 _merge_trace / focus_entity / ④）。
    **以后往 GraphState 加任何"本轮内有效"的字段，都要回到这里加一行。**
    """
    from .intent import is_conversational   # 延迟导入，避免 import 期耦合

    reset = {
        "trace": TRACE_RESET,
        "grade": {},
        "answer_degraded": False,
        "tool_result": {},
        "long_term_memory": "",
        "memory_written": [],
    }
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

    （原有一个"④ 重检索轮直通"的分支，随 ④ 回边一起删于 2026-09-28。）
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


def _build_user_statement(history: list[dict] | None, memory: str = "") -> str:
    """方案 D：把历史 user 发言拼成【用户情况】，但剔除确认/寒暄类。

    原实现把所有历史 user 消息一并拼入，于是"好的""嗯"也被当成主诉，
    导致 MEDICAL_PROMPT 规则 2 的前提（【用户情况】为空则不得搬运症状）
    永远不成立，模型会以为用户一直在描述同一个病、于是反复复述。

    T62-⑦ 追加 `memory`（跨会话长期病史），**只在开启长期记忆时非空**：
    它带一个显式标题 + "除非与本次提问直接相关，否则不要提及"的约束 ——
    【用户情况】一旦不为空，规则 2 那条"空则不得搬运症状"的前提就变了，
    必须让模型能区分"长期背景"与"本次主诉"，否则会把病史当成主诉反复展开。
    ⚠️ MEDICAL_PROMPT 本身**一个字没改**（规则 9/10 无条件生效那条纪律）。
    """
    from .intent import is_conversational   # 延迟导入，复用 B 的词表

    users = [
        (m.get("content") or "").strip()
        for m in (history or [])
        if m.get("role") == "user"
    ]
    complaints = [u for u in users if u and not is_conversational(u)]
    body = "\n".join(complaints)[:800] or "（无）"
    if not memory:
        return body
    return (f"{memory}\n"
            f"（以上为长期背景，除非与本次提问直接相关，否则不要提及）\n"
            f"{body}")


def node_generate(state: GraphState) -> GraphState:
    """基于 context 生成回答（非流式；流式见 astream_answer）"""
    from .rag_chain import get_generation_chain   # 延迟导入，避免循环依赖

    chain = get_generation_chain()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"],
        "user_statement": _build_user_statement(state.get("history"),
                                                state.get("long_term_memory") or ""),
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
        # ⑤：库里没资料，但确定性计算给出了答案 —— 那同样是一种"可回答的证据"。
        # 少了这一条，eGFR 这类问题会被判成"资料不足"去追问或拒答，工具就白算了。
        # ⚠️ 只在**本轮的** tool_result 上生效（它在 node_classify_intent 里按轮清零）。
        if state.get("tool_result"):
            return "generate"
        return "human_review" if _should_ask_user(state, clarify_ok) else "insufficient"
    return "generate"


def _route_after_generate(state: GraphState) -> str:
    """生成之后：正常走忠实性校验；**降级**的那一轮直接收尾。

    为什么降级要跳过校验，而不是"顺手也校验一下"：
      `_degrade_llm_stream` 把降级文案写进了 `grounding.corrected_answer`
      （那是唯一能让前端整段替换、把半句残文换成降级文案的出口）。而
      `_astream_run` 对 grounding 是**后写覆盖**：`ground_check` 一定会在自己的
      update 里带上一个新的 grounding，且它的 corrected_answer 是 None（降级
      文案里没有引用编号，校验不会改文本）⇒ 前端那份修正会被覆盖掉、请求退回
      "用户看到半句残文"。这里断开它，是让两份 grounding 不打架的唯一位置。

    ⚠️ 判据必须是 `answer_degraded`（"回答由降级产出"）而**不是**"本轮有节点降级"。
    差别在重试轮：`grade` 回边时 `rewrite` 可能降级过一次，而生成完全正常 ——
    用"本轮有降级"作判据会把这一轮的正常回答**跳过忠实性校验**，
    静默少做一层，是最难发现的那类回归。
    """
    return "skip" if state.get("answer_degraded") else "check"


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
# 证据分级（T62-④）
# ============================================================================
def node_grade(state: GraphState) -> GraphState:
    """判"这批资料够不够回答"，不够就把这轮推进无资料路径。

    它替代的是那条**已被实测证伪**的方案：按 top_score 区间切 `partial` 三档。
    实测结论（README「已知限制」第一条）：真·可回答 0.658~1.000 与"沾边但答不了"
    0.576 / 0.914 / 0.647 完全重叠，分数这一维切不开。于是把判断从"检索侧的一个
    数字"换成"生成侧的一次自省" —— 不问分数，直接问拿这批资料能不能答。
    实测（2026-09-28，靶样本 ooc-04/05/08/09，top_score 0.330 / 0.576 / 0.647 /
    0.914）：④ 开之前四条全走 generate，开之后**四条全部翻成 insufficient** ✅
    ⇒ 它确实做到了 top_score 做不到的事。

    ⚠️ **这里刻意没有回边**（"换个检索词回到检索子图再检一次"）。2026-09-28 实测后删除，
    依据是三条硬事实：
      1. 靶样本 4/4 的 `retry_query` 与当前检索词**逐字相同**；
      2. 根因在设计里——分级 Prompt 只喂 `{question}` + `{context}`，**看不到上一次
         用的是什么检索词** ⇒ 它只能从同一个 question 再抽一遍关键词，而改写节点
         也是从同一个 question 抽 ⇒ 两边必然收敛到同一组词；
      3. 检索是确定性的，同一个词重跑结果不可能变 ⇒ 那一轮 = 一次完整检索 +
         一次 LLM 分级的**纯浪费**（实测单条墙钟因此从分钟级涨到 10 分钟级）。
    ⇒ 净收益 100% 来自"判不足 → 走无资料路径"这半边，回边只贡献成本。
    要复活它，必须先解决第 2 条：把 `enhanced_query` 喂进分级 Prompt 并要求"避开
    已用过的词"，再重新标定（否则还是撞词）。

    三条"不改行为"的短路（自上而下）：
      1. 总开关关（**默认**）→ 返回空 update。是空 update，不是空节点：
         不写 trace、不动任何字段，行为与本阶段之前逐字一致。
      2. act 的 answer 策略不是 medical（ack / chitchat）→ 这轮压根没检索。
      3. `evidence_state != strong` → 本来就走无资料路径，分级没有意义。
      外加一条 fail-open：LLM 调用或解析失败 → 当作"资料够用"。
      宁可少一次拦，也不能因为分级层自己出问题把**本来答得上来**的问题推进
      "无资料"路径 —— 那是拒答，代价远大于多答。

    判定为不足时只有一条出路：**把 `evidence_state` 改成 none**，让既有的
    `_route_after_retrieval` 原样接管（先追问、再兜底）。刻意复用而非新造分支：
    无资料那条路已被离线测试与三条硬断言覆盖过，走一条走熟的路比新开一条安全得多。

    ⚠️ 本节点**不要**加进 `_astream_run` 的 STREAMING_NODES：它内部会调 LLM，
    放行就会把分级判定的输出当成回答推给前端（`chat` / `insufficient` 已各踩过一次）。
    """
    if not settings.evidence_grade_enabled:
        return {}

    if policy_for(state.get("dialogue_act"))["answer"] != "medical":
        return {}
    if state.get("evidence_state") != "strong":
        return {}

    from .rag_chain import get_evidence_grade_chain, parse_grade_output  # 延迟导入

    try:
        raw = get_evidence_grade_chain().invoke({
            "question": state["question"],
            "context": state.get("context") or "",
        })
        verdict, reason = parse_grade_output(str(raw))
    except Exception as e:
        # fail-open：分级层绝不能自己变成故障点（与 parse_unsupported 同纪律）
        print(f"[证据分级] 调用失败，按『资料够用』处理: {type(e).__name__}: {e}")
        return {}

    if verdict == "sufficient":
        return {
            "grade": {"verdict": "sufficient", "reason": reason},
            "trace": [_make_step("证据分级", {
                "verdict": "资料足以回答", "reason": reason,
            })],
        }

    return {
        "grade": {"verdict": "insufficient", "reason": reason},
        # 关键一步：把结论落回既有通道，让原来的路由原样接管
        "evidence_state": "none",
        "trace": [_make_step("证据分级", {
            "verdict": "资料沾边但答不了 → 走无资料路径",
            "reason": reason,
            "top_score": round(state.get("top_score", 0.0), 3),
        })],
    }


# ============================================================================
# 确定性医学计算（T62-⑤）
# ============================================================================
def node_tool(state: GraphState) -> GraphState:
    """⑤ 确定性医学计算：把"该算的"交给代码，而不是让模型心算。

    为什么要有这一步
        eGFR 这类量有确定公式，但指数项 + 单位换算（μmol/L → mg/dL）让模型经常算错；
        而"去检索"也检索不出"你这个数值对应的答案"（语料里不会有 42.6 这个数）。
        确定性的事交给代码 —— 这是工具节点存在的全部理由。

    位置：`retrieval → tool → grade`。放在检索之后**不是**因为依赖检索结果，而是因为
        工具与检索是并列的两种"拿证据"手段；且它必须在 `_route_after_retrieval`
        判落点**之前**跑完 —— 否则"库里没资料、但这个算得出来"会先被判成 insufficient，
        工具就白算了（见那条路由里的 tool_result 分支）。

    结果怎么进回答：追加到 `context` 尾部。生成节点只读 context，所以"算出来的结论"
        与"检索到的资料"对它就是同一种输入，不必另开一条 Prompt 通路，也就不会
        再引入一份需要同步维护的模板。

    失败纪律（与 ④ 一致）：开关关 / 非医学轮 / 不需要算 / 解析失败 / 参数不全
        ⇒ 一律返回空 update，本轮照原路径走。工具层**绝不会**成为新的故障点。
    """
    if not settings.tool_node_enabled:
        return {}
    if policy_for(state.get("dialogue_act"))["answer"] != "medical":
        return {}
    from .medical_tools import DISCLAIMER as TOOL_DISCLAIMER
    from .medical_tools import compute_for_question

    result = compute_for_question(state.get("question") or "")
    if not result:
        return {}
    text = (f"【确定性计算结果 · {result['tool']}】\n"
            f"{result['summary']}\n（{TOOL_DISCLAIMER}）")
    context = state.get("context") or ""
    return {
        "tool_result": {**result, "text": text},
        "context": f"{context}\n\n{text}" if context else text,
        "trace": [_make_step("计算工具", {
            "tool": result["tool"],
            "args": result["args"],
            "summary": result["summary"],
        })],
    }


# ============================================================================
# 跨会话长期记忆（T62-⑦）
# ============================================================================
def node_recall(state: GraphState) -> GraphState:
    """⑦ 读跨会话长期记忆（慢病史 / 过敏史 / 长期用药），供生成层作背景。

    与 `focus_entity` 的分工（别混）：focus_entity 是 **thread 内**的，换会话就没了；
    这里读的是 **跨会话** 的。两者生命周期不同，塞一起必然互相污染。

    位置：`retrieval → recall → tool → grade`。它不依赖检索结果，但必须在
    `node_generate` 之前读到；与 tool 并列，同属"给生成层补料"。

    开销：只读本地 sqlite（微秒级），**不产生 LLM 调用** —— 这个开关的成本
    全在写入侧（见 node_remember）。
    """
    if not settings.long_term_memory_enabled:
        return {}
    if policy_for(state.get("dialogue_act"))["answer"] != "medical":
        return {}
    from .long_term_memory import recall

    try:
        items, text = recall()
    except Exception as e:
        print(f"[长期记忆] 读取失败，跳过: {type(e).__name__}: {e}")
        return {}
    if not items:
        return {}
    return {
        "long_term_memory": text,
        "trace": [_make_step("长期记忆", {
            "count": len(items),
            "items": [f"{i['kind']}：{i['text']}" for i in items],
        })],
    }


def node_remember(state: GraphState) -> GraphState:
    """⑦ 收尾时把这一轮里**长期成立**的事实写进跨会话记忆。

    位置：`ground_check → remember → END`。**只在正常答完的轮次执行** ——
    降级轮（generate 直接 END）、insufficient、chat 都不经过这里，这是刻意的：
    抽取要基于"一轮完整问答"，残句和兜底文案里没有什么可记的。

    代价：每轮 **+1 次 LLM 调用**（抽取长期事实）。失败一律当"这轮没记的"，
    不影响回答（fail-open，与 ④⑤ 同一纪律）。
    """
    if not settings.long_term_memory_enabled:
        return {}
    from .long_term_memory import KIND_LABELS, learn_from_turn

    try:
        written = learn_from_turn(state.get("question") or "", state.get("answer") or "")
    except Exception as e:
        print(f"[长期记忆] 写入失败，跳过: {type(e).__name__}: {e}")
        return {}
    if not written:
        return {}
    return {
        "memory_written": written,
        "trace": [_make_step("记忆写入", {
            "count": len(written),
            "items": [f"{KIND_LABELS[w['kind']]}：{w['text']}" for w in written],
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

    写 trace 安全性：本节点排在检索子图之后，写 trace 不经过子图边界，不会重复。
    （① 之前这里需要额外论证"后继是程序化调用而非连回子图"，现在子图有独立的
    进出口契约，那条论证连同它防的坑一起消失了。）
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
        # 恢复轮刷新历史：重入检索子图后，改写节点才能看到最近几轮（含澄清气泡）
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


def _route_after_review(state: GraphState) -> str:
    return "requery" if state.get("review_outcome") == "requery" else "insufficient"


# ============================================================================
# 图构建
# ============================================================================
def build_retrieval_graph():
    """检索子图：rewrite →(条件)→ route → retrieve → rerank

    这是检索链路的**唯一实现**，rag_chain.retrieve() 也调用它。
    ack / chitchat 会让子图在 rewrite 之后直接结束（不检索），
    此时 docs / context / sources 都未产出 —— 调用方需容忍缺省值。

    ① 起带 `input_schema` / `output_schema`：进出口按契约显式声明，
    子图**不再继承父图的 trace**，因此"父图里排在检索子图之前的节点一律不许写
    trace"这条约束随之解除（成因与实测证据见 `RetrievalInput` 的 docstring）。
    """
    g = StateGraph(GraphState, input_schema=RetrievalInput, output_schema=RetrievalOutput)
    g.add_node("rewrite", node_rewrite, **_node_opts("rewrite"))
    g.add_node("route", node_route, **_node_opts("route"))
    g.add_node("retrieve", node_retrieve, **_node_opts("retrieve"))
    g.add_node("rerank", node_rerank, **_node_opts("rerank"))

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
    """完整图：intent 分流 → 医学走检索子图 → 读长期记忆 → 确定性计算 → 证据分级 → 二档分流。

    阶段二新增 human_review（interrupt 追问）；
    ④ 新增 grade（证据分级；**回边已实测删除**，理由见 node_grade）；
    ⑤ 新增 tool（确定性医学计算，默认关）；
    ⑦ 新增 recall / remember（跨会话长期记忆，默认关）；
    ⑧ 给节点挂上重试 / 降级（节点级 TimeoutPolicy 不可用，见 `_timeout_for` 位置那段说明）。
    checkpointer 由 get_graph() 按配置注入，build 本身不读配置。

    澄清闸的可用性按**实际传入的 checkpointer** 绑定到条件边上（`clarify_ok`），
    而不是读 `graph_checkpointer_enabled`：评测脚本走 build_graph() 不带
    checkpointer，若按配置放行就会在没有 checkpointer 的图上调 interrupt() 抛错。

    ⚠️ 澄清后的重检索**不再另起一个 requery 节点**（① 的连带清理）：
    那个节点当初是"程序化调用子图"的形态，纯粹为了绕开 trace 重复追加的坑。
    schema 分离之后坑没了，它就是一条普通的 `human_review → retrieval` 边 ——
    图能一眼看懂，也少一个节点要维护。**检索实现仍然只有一份。**
    """
    clarify_ok = checkpointer is not None
    g = StateGraph(GraphState)
    g.add_node("intent", node_classify_intent)
    g.add_node("retrieval", build_retrieval_graph())
    g.add_node("recall", node_recall, **_node_opts("recall"))
    g.add_node("tool", node_tool, **_node_opts("tool"))
    g.add_node("grade", node_grade, **_node_opts("grade"))
    g.add_node("human_review", node_human_review)   # interrupt 节点：刻意不挂可靠性参数
    g.add_node("generate", node_generate, **_node_opts("generate"))
    g.add_node("ground_check", node_ground_check, **_node_opts("ground_check"))
    g.add_node("remember", node_remember, **_node_opts("remember"))
    g.add_node("insufficient", node_answer_insufficient, **_node_opts("insufficient"))
    g.add_node("chat", node_chat_generate, **_node_opts("chat"))

    g.set_entry_point("intent")
    g.add_conditional_edges(
        "intent",
        _route_by_intent,
        {"medical": "retrieval", "chat": "chat"},
    )
    # 检索完**不再**直连 generate：无资料时必须换掉 Prompt，否则规则 9/10 会逼出假引用。
    # ① 起中间多一跳 grade（证据分级）。它是**空节点**（总开关默认关，直接返回空 update），
    # 所以"多一跳"不改变任何行为，只是给判不足留一个统一的改写点。
    # ⑤ 在它前面再插一跳 tool（确定性计算）。同样是**空节点**（默认关），
    # 且它**必须排在判落点之前** —— 路由要读 tool_result 决定"没资料也能答"。
    # ⑦ 再前面插一跳 recall（读跨会话长期记忆）。也是空节点（默认关），
    # 只读本地 sqlite、不产生 LLM 调用；读到的背景由 node_generate 注入【用户情况】。
    g.add_edge("retrieval", "recall")
    g.add_edge("recall", "tool")
    g.add_edge("tool", "grade")
    # 出口表**与检索子图那条完全共用同一个函数** —— 刻意不复制一份：
    # grade 判不足时只把 evidence_state 改成 none，之后该往哪走由原来的策略决定，
    # 判据一个字都不该变。各写一份的话，以后改证据策略就得改两处，而"漏改一处"
    # 正是本项目最初那个 bug 的成因。
    # （2026-09-28 之前这里还有一个 "retry": "retrieval" 的回边，已实测删除。）
    g.add_conditional_edges(
        "grade",
        partial(_route_after_retrieval, clarify_ok=clarify_ok),
        {
            "chat": "chat",
            "generate": "generate",
            "insufficient": "insufficient",
            "human_review": "human_review",
        },
    )
    # 澄清之后：补充信息已融合进 question，直接重入检索子图。
    # 此时 clarify_rounds 已 +1，若仍无证据，_should_ask_user 会因为轮次用尽而
    # 不再追问、落到 insufficient ⇒ 循环天然有界，不需要额外的计数。
    g.add_conditional_edges(
        "human_review", _route_after_review,
        {"requery": "retrieval", "insufficient": "insufficient"},
    )
    # 阶段四：generate 之后过忠实性校验（L1 剥除越界引用 / L2 追加提示），
    # 由它统一收尾。chat / insufficient 保持直连 END：它们的 Prompt 本就禁止引用编号，
    # 接上去只是白跑一次正则（且 insufficient 的"无引用"是 eval_dialogue 的硬断言）。
    # ⑧：降级的那一轮直接 END（理由见 _route_after_generate）。
    g.add_conditional_edges(
        "generate", _route_after_generate,
        {"check": "ground_check", "skip": END},
    )
    # ⑦ 收尾：正常答完的轮次再抽一次"长期成立的病史"写进跨会话记忆。
    # 只有 ground_check 这条正常路径接它 —— 降级 / insufficient / chat 都不记（理由见 node_remember）。
    g.add_edge("ground_check", "remember")
    g.add_edge("remember", END)
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
          ← 阶段四：忠实性校验**真的改了文本**（剥除越界引用 / 追加提示）时出现。
          ⑧ 起还多一个来源：节点降级（verdict="node_degraded"）—— 那是**流式节点
          挂掉时半句残文的唯一出路**，因为 token 事件只能追加。
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
    重入检索子图后改写节点才能看到最近几轮（含上一轮的澄清气泡）。
    """
    from langgraph.types import Command

    cmd = Command(resume={"reply": user_reply, "history": history or []})
    async for payload in _astream_run(cmd, config):
        yield payload
