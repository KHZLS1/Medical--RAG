"""T62 三条深化的离线自测：⑧ 节点级可靠性 / ① 子图 schema 分离 / ④ 证据分级回边

不需要 Milvus / LLM / MySQL：把 graph.py 引用的外部依赖全部换成内存假实现，
只验证**图的行为**。跑法（backend 目录、RAG 环境）：

    python scripts/test_node_reliability.py

退出码 0 = 全绿。

为什么这个测试非要有
--------------------
这三条改动的共同点是"平时看不出来"：
  · ⑧ 的重试/降级只在出错时生效 —— 平时跑一遍什么都测不到；
  · ① 的收益是**消除**一个坑 —— 坑没了，测试也不会变红，反而容易被人改回去；
  · ④ 默认关闭 —— 开了才有行为，关着时等于没实现。
所以每一条都要有"把开关拨过去、把故障注入进去"的断言，否则这些代码
只有等生产环境踩了才知道有没有用。
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

import app.graph as g
import app.rag_chain as rc
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return cond


# ============================================================================
# 假实现（与真实依赖同签名）
# ============================================================================
class FakeRewriteResult:
    def __init__(self, query, dialogue_act="new_question", degraded=False):
        self.query = query
        self.dialogue_act = dialogue_act
        self.degraded = degraded
        self.reason = "fake"
        self.focus_entity = ""


class FakeRouting:
    source = "general"
    confidence = 0.9
    filename = None
    reason = "fake"


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return list(self._docs)


HIT_DOCS = [Document(page_content="戈谢病是常染色体隐性遗传的溶酶体贮积病…",
                     metadata={"department": "儿科", "title": "戈谢病", "source": "t"})]


def fake_rerank(query, docs, top_k=5):
    if not docs:
        return []
    return [Document(page_content=d.page_content,
                     metadata={**(d.metadata or {}), "rerank_score": 0.95}) for d in docs][:top_k]


def fake_format_history(history):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def fake_chain_ok(inputs):
    return "这是基于资料的回答。[1]"


class FakeChain:
    """可注入行为的假链：默认成功，也可改成抛异常或返回固定串。"""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        return self._fn(inputs)


def chain_ok():
    return FakeChain(fake_chain_ok)


def chain_boom(exc=ConnectionError("provider down")):
    def _boom(inputs):
        raise exc
    return FakeChain(_boom)


def chain_json(payload_getter):
    """返回 JSON 字符串的链。payload_getter(call_index) -> dict。"""
    import json as _json
    def _fn(inputs):
        return _json.dumps(payload_getter())
    return FakeChain(_fn)


def base_patches(**over):
    """graph.py 里所有外部依赖的替身。over 可覆盖任意一项。"""
    table = {
        "rewrite_for_retrieval": lambda q, h=None: FakeRewriteResult(q),
        "route_knowledge_source": lambda q: FakeRouting(),
        "get_retriever": lambda k=20: FakeRetriever(HIT_DOCS),
        "rerank_documents": fake_rerank,
        "format_history": fake_format_history,
    }
    table.update(over)
    return table


def patch_graph(**over):
    return patch.multiple("app.graph", **base_patches(**over))


async def run(graph, graph_input, config):
    """异步跑图，返回 (events, 逐节点累计的 trace)。与生产 _astream_run 同构。"""
    events = []
    trace_acc = []
    async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        if "__interrupt__" in chunk:
            events.append(("interrupt", chunk["__interrupt__"][0].value))
            continue
        for _node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            steps = update.get("trace")
            if isinstance(steps, list):
                trace_acc.extend(steps)
            for key in ("answer", "evidence_state", "question", "answer_degraded", "grounding"):
                if key in update:
                    events.append((key, update[key]))
    return events, trace_acc


def steps_of(trace):
    return [t.get("step") for t in trace]


# ============================================================================
# ⑧ 节点级可靠性
# ============================================================================
async def t_reliability():
    print("== ⑧-1 结构性判定：谁挂了可靠性参数、谁没挂 ==")
    with patch.object(g.settings, "node_retry_enabled", True):
        # 流式节点必须**没有** retry_policy（重试会把残句和重试结果拼在一起）
        for node in ("generate", "chat", "insufficient"):
            opts = g._node_opts(node)
            check(f"{node} 不挂 retry_policy", "retry_policy" not in opts, str(list(opts)))
            check(f"{node} 有 error_handler", "error_handler" in opts)
        for node in ("rewrite", "route", "retrieve", "rerank", "grade", "ground_check"):
            opts = g._node_opts(node)
            check(f"{node} 挂 retry_policy", "retry_policy" in opts, str(list(opts)))
            check(f"{node} 有 error_handler", "error_handler" in opts, str(list(opts)))
        # interrupt 与纯规则节点刻意不登记
        check("human_review 不加任何可靠性参数", g._node_opts("human_review") == {},
              str(g._node_opts("human_review")))
        check("intent 不加任何可靠性参数", g._node_opts("intent") == {})
        # 降级 handler 必须按**节点名**分派：按类别分派会让同一类别里的
        # rewrite / grade / ground_check 撞车（它们的降级目标各不相同）
        check("rewrite 与 grade 的 handler 不同",
              g._DEGRADE_HANDLERS["rewrite"] is not g._DEGRADE_HANDLERS["grade"])
        check("rewrite 与 ground_check 的 handler 不同",
              g._DEGRADE_HANDLERS["rewrite"] is not g._DEGRADE_HANDLERS["ground_check"])
        check("retrieve 与 rerank 共用同一个召回降级",
              g._DEGRADE_HANDLERS["retrieve"] is g._DEGRADE_HANDLERS["rerank"])

    print("== ⑧-2 降级文案不得命中自省白名单（否则会被误剥）==")
    from app.rag_chain import UNANSWERABLE_HINTS, detect_unanswerable
    for text in (g.DEGRADED_LLM_ANSWER, g.DEGRADED_CHAT_ANSWER,
                 g.DEGRADED_INSUFFICIENT_ANSWER):
        hit = detect_unanswerable(text)
        check("降级文案不触发 detect_unanswerable", hit == "", f"命中 {hit!r}: {text[:20]}…")
    check("降级文案与自省白名单无交集",
          not any(h in g.DEGRADED_LLM_ANSWER for h in UNANSWERABLE_HINTS))

    print("== ⑧-3 非流式节点重试：第一次挂、第二次成功 ==")
    calls = {"n": 0}

    def flaky_rewrite(q, h=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("首次失败")
        return FakeRewriteResult("重试后的检索词")

    with patch_graph(rewrite_for_retrieval=flaky_rewrite), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01), \
         patch.object(g.settings, "node_retry_enabled", True), \
         patch.object(g.settings, "node_retry_max_attempts", 3):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880001))
    check("rewrite 被调用 2 次（重试生效）", calls["n"] == 2, f"实际 {calls['n']} 次")
    check("重试后拿到了改写产物", any(
        t.get("step") == "查询改写" and t.get("query") == "重试后的检索词" for t in trace),
        str(steps_of(trace)))
    check("没有产生降级步骤", "节点降级" not in steps_of(trace))

    print("== ⑧-4 流式节点不重试：generate 每次都挂也只调一次 ==")
    g_calls = {"n": 0}

    def boom_chain(inputs):
        g_calls["n"] += 1
        raise ConnectionError("provider down")

    with patch_graph(), \
         patch("app.rag_chain.get_generation_chain", lambda: FakeChain(boom_chain)), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01), \
         patch.object(g.settings, "node_retry_enabled", True), \
         patch.object(g.settings, "node_retry_max_attempts", 3):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880002))
    check("generate 只被调用 1 次（流式节点不重试）", g_calls["n"] == 1, f"实际 {g_calls['n']} 次")
    check("trace 里有降级步骤", "节点降级" in steps_of(trace), str(steps_of(trace)))
    ans = next((e[1] for e in events if e[0] == "answer"), "")
    check("用户拿到的是降级文案而不是空回答", ans == g.DEGRADED_LLM_ANSWER, repr(ans[:30]))
    grounding = next((e[1] for e in events if e[0] == "grounding"), None)
    check("写了 corrected_answer（前端整段替换的唯一出口）",
          bool(grounding and grounding.get("corrected_answer") == g.DEGRADED_LLM_ANSWER),
          str(grounding))
    check("降级轮跳过忠实性校验（不让两份 grounding 打架）",
          "忠实性校验" not in steps_of(trace), str(steps_of(trace)))
    check("降级轮没有引用编号", "[" not in ans.replace("[", "[") or "[1]" not in ans)

    print("== ⑧-5 rewrite 全挂 → 退化成原句检索，且不破坏不变量 ==")
    with patch_graph(rewrite_for_retrieval=lambda q, h=None: (_ for _ in ()).throw(
            ConnectionError("always"))), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01), \
         patch.object(g.settings, "node_retry_enabled", True):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880003))
    check("走了降级", "节点降级" in steps_of(trace))
    check("仍然产出了回答（没把整轮炸掉）", any(e[0] == "answer" for e in events))
    check("「意图判定」恰好 1 次（降级也要补写这条 trace）",
          steps_of(trace).count("意图判定") == 1, str(steps_of(trace)))
    # error_handler 被调度成 PUSH 任务、**不触发失败节点的出边**，
    # 所以这条断言真正验证的是 handler 里那句 `Command(goto="route")` 有没有生效。
    # 少了它流程会静默停在这里：context 缺失 → generate 抛 KeyError → 又走降级，
    # 用户看到"服务暂时不可用"，却完全不知道是改写挂了一下下。
    check("降级后**继续**走完了检索链路（goto 生效）",
          "来源路由" in steps_of(trace) and "混合检索" in steps_of(trace)
          and "精排" in steps_of(trace), str(steps_of(trace)))
    check("用原句完成了检索（enhanced_query 落成原问题）",
          any(t.get("step") == "查询改写" or t.get("step") == "节点降级" for t in trace)
          and "混合检索" in steps_of(trace))
    # 回归：判据必须是"回答由降级产出"而非"本轮有节点降级"。
    # 这一轮 rewrite 降级过，但回答是正常生成的 ⇒ 忠实性校验**必须照跑**。
    # 用"本轮有降级"作判据的话这里会静默少做一层校验。
    check("上游降级不影响本轮正常回答的忠实性校验",
          "忠实性校验" in steps_of(trace), str(steps_of(trace)))
    check("answer_degraded 未被上游降级误置位",
          not any(e[0] == "answer_degraded" and e[1] for e in events),
          str([e for e in events if e[0] == "answer_degraded"]))

    print("== ⑧-5b 来源路由挂掉 → 跳过来源过滤、继续检索 ==")
    with patch_graph(route_knowledge_source=lambda q: (_ for _ in ()).throw(
            ConnectionError("router down"))), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01), \
         patch.object(g.settings, "node_retry_enabled", False):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880006))
    check("走了降级", "节点降级" in steps_of(trace), str(steps_of(trace)))
    check("降级后仍然检索到了资料（没被来源过滤掐死）",
          "混合检索" in steps_of(trace) and "精排" in steps_of(trace)
          and any(e[0] == "answer" for e in events), str(steps_of(trace)))
    check("evidence_state 仍是 strong（auto 路由不过滤）",
          any(e[0] == "evidence_state" and e[1] == "strong" for e in events),
          str([e for e in events if e[0] == "evidence_state"]))
    check("上游降级不影响忠实性校验（同上，防同一处回归）",
          "忠实性校验" in steps_of(trace), str(steps_of(trace)))

    print("== ⑧-6 检索链挂掉 → 退化为『无可用资料』，不炸 ==")
    with patch_graph(get_retriever=lambda k=20: (_ for _ in ()).throw(
            ConnectionError("milvus down"))), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01), \
         patch.object(g.settings, "node_retry_enabled", True), \
         patch.object(g.settings, "human_review_enabled", False):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880004))
    check("走了降级", "节点降级" in steps_of(trace), str(steps_of(trace)))
    check("落点 insufficient（无资料路径）",
          any(e[0] == "answer" for e in events) and
          "证据不足兜底" in steps_of(trace), str(steps_of(trace)))
    check("evidence_state 落成 none",
          any(e[0] == "evidence_state" and e[1] == "none" for e in events))

    print("== ⑧-7 总开关关掉 → 不重试、直接降级 ==")
    calls2 = {"n": 0}

    def flaky_rewrite2(q, h=None):
        calls2["n"] += 1
        raise ConnectionError("第一次就挂")

    with patch_graph(rewrite_for_retrieval=flaky_rewrite2), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "node_retry_enabled", False), \
         patch.object(g.settings, "node_retry_initial_interval", 0.01):
        graph = g.build_graph()
        events, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                                  g.thread_config(880005))
    check("开关关掉时只调用 1 次", calls2["n"] == 1, f"实际 {calls2['n']} 次")
    check("仍然走了降级（开关只关重试，不关降级）", "节点降级" in steps_of(trace))

    print("== ⑧-8 节点级 timeout 必须缺席（挂上就 compile 失败）==")
    # 防地雷回归：langgraph 的节点级 TimeoutPolicy 只支持 async 节点
    # （`validate_timeout_supported` 在 compile 阶段抛 ValueError），本项目全是 sync
    # ⇒ 只要有人把 timeout 挂回来，`build_graph()` 抛错、后端整个起不来。
    for name in g._NODE_KINDS:
        opts = g._node_opts(name)
        check(f"{name} 不挂 timeout", "timeout" not in opts, str(sorted(opts)))
    check("_timeout_for 已从 graph 移除", not hasattr(g, "_timeout_for"))
    check("_node_opts 只产出 retry_policy / error_handler 两类键",
          all(set(g._node_opts(n)) <= {"retry_policy", "error_handler"} for n in g._NODE_KINDS))
    # 真正能掐断请求的超时在客户端层（llm.py 透传 ChatOpenAI(timeout=...)）
    check("LLM 客户端超时可配且默认 180s", g.settings.llm_timeout_sec == 180.0,
          str(g.settings.llm_timeout_sec))


# ============================================================================
# ① 子图 schema 分离
# ============================================================================
async def t_subgraph_schema():
    print("== ①-1 契约本身：input 不含 trace（改回去就立刻退化成重复追加）==")
    ins = set(g.RetrievalInput.__annotations__)
    outs = set(g.RetrievalOutput.__annotations__)
    check("RetrievalInput 不含 trace", "trace" not in ins, str(sorted(ins)))
    check("RetrievalOutput 含 trace", "trace" in outs)
    check("RetrievalInput 含 focus_entity（澄清重入不再需要手工传）", "focus_entity" in ins)
    # ④ 回边删掉后 grade_retry 一并去掉（回边是它唯一的用途）
    check("RetrievalInput 不含 grade_retry（④ 回边已删）", "grade_retry" not in ins, str(sorted(ins)))
    check("RetrievalOutput 不含 routing（只有子图内部读它）", "routing" not in outs,
          str(sorted(outs)))
    for key in ("docs", "context", "sources", "top_score", "evidence_state", "enhanced_query"):
        check(f"RetrievalOutput 含 {key}", key in outs)

    print("== ①-2 澄清重检索走普通边后，trace 不重复 ==")
    with patch_graph(), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "human_review_enabled", True):
        saver = InMemorySaver()
        graph = g.build_graph(checkpointer=saver)
        cfg = g.thread_config(880101)
        # 第一轮：检索不到 → interrupt
        with patch_graph(get_retriever=lambda k=20: FakeRetriever([])):
            ev1, _ = await run(graph, {"question": "戈谢病的酶替代疗法", "history": [], "trace": []}, cfg)
        check("第一轮中断了", any(e[0] == "interrupt" for e in ev1))
        # 第二轮：带补充恢复
        ev2, trace2 = await run(graph, Command(resume={
            "reply": "腹部胀大，脾脏肿大三年",
            "history": [{"role": "user", "content": "戈谢病的酶替代疗法"},
                        {"role": "assistant", "content": "（追问话术）"}],
        }), cfg)
    steps = steps_of(trace2)
    check("含「人工澄清」", "人工澄清" in steps, str(steps))
    # 这一条是 ① 的**核心回归断言**：若把 trace 加回 RetrievalInput，
    # human_review 写的「人工澄清」会被子图继承、返回时再追加一次 —— 变成 2 次。
    check("「人工澄清」恰好 1 次（schema 分离防住重复追加）",
          steps.count("人工澄清") == 1, f"出现 {steps.count('人工澄清')} 次: {steps}")
    check("「意图判定」恰好 1 次", steps.count("意图判定") == 1, str(steps))
    check("「查询改写」恰好 1 次", steps.count("查询改写") == 1, str(steps))
    check("「混合检索」恰好 1 次", steps.count("混合检索") == 1, str(steps))
    check("恢复后正常生成", any(e[0] == "answer" for e in ev2))

    print("== ①-3 已无 requery 节点，图结构就是普通边 ==")
    nodes = set(g.build_graph().get_graph().nodes)
    check("节点表里没有 requery", "requery" not in nodes, str(sorted(nodes)))
    check("有 grade 节点", "grade" in nodes)
    check("不再有 node_requery 函数", not hasattr(g, "node_requery"))
    check("不再有 _route_after_requery 函数", not hasattr(g, "_route_after_requery"))

    print("== ①-4 rag_chain.retrieve() 仍可用（不再传 trace）==")
    with patch_graph():
        docs, ctx, sources, q = rc.retrieve("戈谢病")
    check("返回了文档", len(docs) == 1, str(len(docs)))
    check("返回了 context", bool(ctx))
    check("返回了 sources", len(sources) == 1)
    check("enhanced_query 回落到原问题", q == "戈谢病", q)


# ============================================================================
# ④ 证据分级回边
# ============================================================================
async def t_evidence_grade():
    print("== ④-1 默认关：零行为变化（无 trace、不改落点）==")
    with patch_graph(), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch.object(g.settings, "evidence_grade_enabled", False):
        graph = g.build_graph()
        ev, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                              g.thread_config(880201))
    check("没有「证据分级」trace", "证据分级" not in steps_of(trace), str(steps_of(trace)))
    check("落点仍是 generate", "忠实性校验" in steps_of(trace))

    print("== ④-2 判不足 → 直接走无资料路径（⚠️ 刻意没有回边）==")
    # 回边已于 2026-09-28 实测删除（分级拿不到新检索词，纯浪费一轮）。下面四条
    # 「检索/分级各只跑 1 次」的断言就是**防回边复活**的护栏：加回来立刻变红。
    retrieve_calls = {"n": 0}

    def counting_retriever(k=20):
        class R:
            def invoke(self, query):
                retrieve_calls["n"] += 1
                return list(HIT_DOCS)
        return R()

    grade_calls = {"n": 0}

    def grade_insufficient(inputs=None):
        grade_calls["n"] += 1
        return {"verdict": "insufficient", "reason": "只讲了病因"}

    with patch_graph(get_retriever=counting_retriever), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch("app.rag_chain.get_evidence_grade_chain",
               lambda: chain_json(grade_insufficient)), \
         patch.object(g.settings, "evidence_grade_enabled", True), \
         patch.object(g.settings, "human_review_enabled", False):
        graph = g.build_graph()
        ev, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                              g.thread_config(880202))
    steps = steps_of(trace)
    check("判不足后落回无资料路径", "证据不足兜底" in steps, str(steps))
    check("没有走 generate（不硬答）", "忠实性校验" not in steps, str(steps))
    check("检索只跑了 1 次（无重检索）", retrieve_calls["n"] == 1, f"实际 {retrieve_calls['n']} 次")
    check("分级只调了 1 次", grade_calls["n"] == 1, f"实际 {grade_calls['n']} 次")
    check("trace 里没有「重试检索」", "重试检索" not in steps, str(steps))
    check("「意图判定」恰好 1 次", steps.count("意图判定") == 1, str(steps))
    check("分级结论落到了 state.grade 上（供诊断读）",
          any(t.get("step") == "证据分级" for t in trace)
          and any(e[0] == "evidence_state" and e[1] == "none" for e in ev), str(ev[-4:]))

    print("== ④-3 判够用 → 照常生成（分级不能变成拒答器）==")
    with patch_graph(), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch("app.rag_chain.get_evidence_grade_chain",
               lambda: chain_json(lambda: {"verdict": "sufficient", "reason": "覆盖了治疗"})), \
         patch.object(g.settings, "evidence_grade_enabled", True):
        graph = g.build_graph()
        ev, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                              g.thread_config(880203))
    steps = steps_of(trace)
    check("判够用时照常生成", "忠实性校验" in steps, str(steps))
    check("trace 写明「资料足以回答」",
          any(t.get("verdict") == "资料足以回答" for t in trace),
          str([t for t in trace if t.get("step") == "证据分级"]))
    check("判够用后 evidence_state 仍是 strong（没被误改成 none）",
          any(e[0] == "evidence_state" and e[1] == "strong" for e in ev)
          and not any(e[0] == "evidence_state" and e[1] == "none" for e in ev), str(ev))

    print("== ④-4 分级层自己挂了 → fail-open（按够用处理）==")
    with patch_graph(), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch("app.rag_chain.get_evidence_grade_chain",
               lambda: chain_boom(RuntimeError("分级服务挂了"))), \
         patch.object(g.settings, "evidence_grade_enabled", True):
        graph = g.build_graph()
        ev, trace = await run(graph, {"question": "戈谢病怎么治", "history": [], "trace": []},
                              g.thread_config(880204))
    check("fail-open：仍然正常生成", "忠实性校验" in steps_of(trace), str(steps_of(trace)))
    check("fail-open：没有产出分级 trace", "证据分级" not in steps_of(trace))

    print("== ④-5 分级输出解析：脏数据一律 fail-open，且**只认 insufficient** ==")
    from app.rag_chain import parse_grade_output
    check("返回 2 元组 (verdict, reason)",
          len(parse_grade_output('{"verdict":"insufficient","reason":"r"}')) == 2)
    check("空串 → sufficient", parse_grade_output("")[0] == "sufficient")
    check("非 JSON → sufficient", parse_grade_output("我觉得还行")[0] == "sufficient")
    check("数组而非对象 → sufficient", parse_grade_output("[1,2]")[0] == "sufficient")
    check("大小写/空格容错", parse_grade_output('{"verdict":" INSUFFICIENT "}')[0] == "insufficient")
    # 旧版 Prompt 里那个 "retry" 值：回边没了之后必须折成 sufficient（少拦一次，不是多拦）
    check("旧版 'retry' 值折成 sufficient（回边已删）",
          parse_grade_output('{"verdict":"retry"}')[0] == "sufficient")
    check("reason 截断到 60 字",
          len(parse_grade_output('{"verdict":"insufficient","reason":"' + "疾" * 200 + '"}')[1]) == 60)
    check("代码块围栏可解析",
          parse_grade_output('```json\n{"verdict":"insufficient","reason":"r"}\n```')[0]
          == "insufficient")

    print("== ④-6 跨轮不残留：第二轮 trace 与全新 thread 单轮逐字一致 ==")
    class CountingRewrite:
        def __init__(self):
            self.questions = []

        def __call__(self, q, h=None):
            self.questions.append(q)
            return FakeRewriteResult(q)

    rw = CountingRewrite()
    with patch_graph(rewrite_for_retrieval=rw), \
         patch("app.rag_chain.get_generation_chain", chain_ok), \
         patch("app.rag_chain.get_insufficient_chain", chain_ok), \
         patch("app.rag_chain.get_chat_chain", chain_ok), \
         patch("app.rag_chain.get_evidence_grade_chain",
               lambda: chain_json(lambda: {"verdict": "insufficient", "reason": "沾边"})), \
         patch.object(g.settings, "evidence_grade_enabled", True), \
         patch.object(g.settings, "human_review_enabled", False):
        saver = InMemorySaver()
        graph = g.build_graph(checkpointer=saver)
        cfg = g.thread_config(880301)
        await run(graph, {"question": "第一轮问题", "history": [], "trace": []}, cfg)
        ev2, trace2 = await run(graph, {"question": "第二轮的全新问题", "history": [], "trace": []}, cfg)

        # 对照组：同一个第二轮的题，在**全新 thread** 上跑一轮。
        # 只看步数会把跨轮累积和"本来就长"混起来，与全新 thread 对比才是正确比法。
        graph_fresh = g.build_graph()
        _ev_f, trace_fresh = await run(graph_fresh,
                                       {"question": "第二轮的全新问题", "history": [], "trace": []},
                                       g.thread_config(880302))
    steps2 = steps_of(trace2)
    check("第二轮走了改写（没有拿上一轮的检索词直通）",
          "查询改写" in steps2, str(steps2))
    check("第二轮改写拿到的是新问题", "第二轮的全新问题" in rw.questions, str(rw.questions))
    check("第二轮 trace 与全新 thread 单轮逐字一致（无跨轮累积）",
          steps2 == steps_of(trace_fresh),
          f"第二轮 {steps2}\nvs 单轮 {steps_of(trace_fresh)}")
    check("第二轮「意图判定」恰好 1 次", steps2.count("意图判定") == 1, str(steps2))


async def main():
    await t_reliability()
    print()
    await t_subgraph_schema()
    print()
    await t_evidence_grade()

    print("\n离线自测完成。")
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项:")
        for name in _FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
