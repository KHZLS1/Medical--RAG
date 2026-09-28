"""阶段二澄清流程的离线自测（不需要 Milvus / LLM / MySQL）

原理：把 graph.py 引用的外部依赖（改写/路由/检索/精排/生成链）全部替换成
内存假实现，只验证**新增的管线本身**：
  1. evidence=none 且可追问 → 图在 human_review 处 interrupt 暂停
  2. 暂停后 get_state(config).next 指向 human_review（main.py 的恢复判据）
  3. Command(resume=...) 恢复 → 回复融合进原问题 → 重入检索子图重新检索
  4. 补充后 evidence=strong → generate 正常收尾
  5. 回复「算了」→ insufficient 收尾
  6. 急症问题 → 不追问，直接 insufficient
  7. 澄清轮 trace 不与第一轮检索步骤重复（D4 验证点）
  8. 无 checkpointer → 不进 human_review（验证清单 #3「开关回退」的核心不变量）
  9. 删除会话 → 该 thread 的 checkpoint 快照被清理（阶段二 §4 的记账项）

跑法（backend 目录、rag 环境）：
  python scripts/test_clarify_flow.py
退出码 0 = 全绿。
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch
from types import SimpleNamespace

import app.graph as g
from app import checkpointer as cp
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

_FAILED: list[str] = []


# ---- 假实现：与真实依赖同签名 ----
class FakeRewriteResult:
    def __init__(self, query, dialogue_act="new_question"):
        self.query = query
        self.dialogue_act = dialogue_act
        self.degraded = False
        self.reason = ""
        # 阶段三：node_rewrite 会读 result.focus_entity，假实现必须同签名（空串=无焦点）
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


EMPTY_DOCS = []   # 第一轮：检索不到任何东西 → evidence=none
HIT_DOCS = [Document(page_content="戈谢病是常染色体隐性遗传的溶酶体贮积病…",
                     metadata={"department": "儿科", "title": "戈谢病", "source": "t"})]


def fake_rerank(query, docs, top_k=5):
    if not docs:
        return []
    scored = [Document(page_content=d.page_content, metadata={
        **(d.metadata or {}), "rerank_score": 0.95}) for d in docs]
    return scored[:top_k]


def fake_format_history(history):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def fake_chain_invoke(inputs):
    return "这是基于资料的回答。[1]"


def make_fake_chain():
    chain = SimpleNamespace()
    chain.invoke = fake_chain_invoke
    return chain


PATCHES = {
    "app.graph.rewrite_for_retrieval":
        lambda q, h=None: FakeRewriteResult(q),
    "app.graph.route_knowledge_source":
        lambda q: FakeRouting(),
    "app.graph.get_retriever":
        lambda k=20: FakeRetriever(EMPTY_DOCS),
    "app.graph.rerank_documents": fake_rerank,
    "app.graph.format_history": fake_format_history,
    "app.rag_chain.get_generation_chain": make_fake_chain,
    "app.rag_chain.get_insufficient_chain": make_fake_chain,
    "app.rag_chain.get_chat_chain": make_fake_chain,
}


async def run_graph(graph, graph_input, config):
    """异步跑图，收集 updates 流里的事件（与生产 _astream_run 同为异步消费）"""
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
            if "trace" in update:
                # updates 推的是节点原始返回值：入口节点发的是 TRACE_RESET 哨兵
                # （不是 list），与生产 _astream_run 同款守卫
                steps = update["trace"]
                if isinstance(steps, list):
                    trace_acc.extend(steps)
            if "answer" in update:
                events.append(("answer", update["answer"]))
            if "evidence_state" in update:
                events.append(("evidence", update["evidence_state"]))
            if "question" in update:
                events.append(("question", update["question"]))
    events.append(("trace", trace_acc))
    return events


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return cond


async def main():
    with patch.multiple("app.graph",
                        rewrite_for_retrieval=PATCHES["app.graph.rewrite_for_retrieval"],
                        route_knowledge_source=PATCHES["app.graph.route_knowledge_source"],
                        get_retriever=PATCHES["app.graph.get_retriever"],
                        rerank_documents=fake_rerank,
                        format_history=fake_format_history), \
         patch("app.rag_chain.get_generation_chain", make_fake_chain), \
         patch("app.rag_chain.get_insufficient_chain", make_fake_chain), \
         patch("app.rag_chain.get_chat_chain", make_fake_chain):
        # interrupt() 必须有 checkpointer 才能暂停/恢复；用内存版，不碰生产 sqlite
        saver = InMemorySaver()
        graph = g.build_graph(checkpointer=saver)

        print("== 1. 证据不足且可追问 → interrupt 暂停 ==")
        config = g.thread_config(999901)
        # 问题文本在这里**无关紧要**：检索/精排全是假实现，证据有无由 EMPTY_DOCS /
        # HIT_DOCS 决定。别把它当成"库外样例"——那需要真实 Milvus 标定，
        # 见 test_clarify_e2e.py 顶部的⚠️。
        events = await run_graph(graph, {
            "question": "戈谢病的酶替代疗法",
            "history": [],
            "trace": [],
        }, config)
        intr = [e for e in events if e[0] == "interrupt"]
        ok1 = check("收到 clarification_request", len(intr) == 1)
        if intr:
            payload = intr[0][1]
            check("interrupt value 带 message", "message" in payload and payload["message"])
            check("interrupt value 带 type=clarification_request",
                  payload.get("type") == "clarification_request")
        check("没有继续生成回答", not any(e[0] == "answer" for e in events))

        print("== 2. 带补充恢复 → 融合 + 重检索 + 生成 ==")
        with patch("app.graph.get_retriever", lambda k=20: FakeRetriever(HIT_DOCS)):
            events2 = await run_graph(graph, Command(resume={
                "reply": "腹部胀大，脾脏肿大三年",
                "history": [{"role": "user", "content": "戈谢病的酶替代疗法"},
                            {"role": "assistant", "content": "（追问话术）"}],
            }), config)
        fused = [e for e in events2 if e[0] == "question"]
        ok2 = check("回复融合进原问题", any("用户补充" in e[1] and "戈谢病" in e[1]
                                          for e in fused),
                    str(fused))
        check("重检索后有证据", any(e[0] == "evidence" and e[1] == "strong" for e in events2))
        check("正常生成回答", any(e[0] == "answer" for e in events2))

        print("== 3. trace 不重复（D4 验证点）==")
        trace2 = [e for e in events2 if e[0] == "trace"][0][1]
        steps = [t["step"] for t in trace2]
        check("含「人工澄清」步骤", "人工澄清" in steps, str(steps))
        check("含第二轮「意图判定」", "意图判定" in steps, str(steps))
        dup_intent = steps.count("意图判定")
        check("「意图判定」恰好 1 次（无子图重复追加）", dup_intent == 1,
              f"出现 {dup_intent} 次: {steps}")
        dup_rewrite = steps.count("查询改写")
        check("「查询改写」恰好 1 次", dup_rewrite == 1,
              f"出现 {dup_rewrite} 次: {steps}")

        print("== 4. 回复「算了」→ 放弃，insufficient 收尾 ==")
        config3 = g.thread_config(999902)
        await run_graph(graph, {"question": "某生僻问题", "history": [], "trace": []}, config3)
        with patch("app.graph.get_retriever", lambda k=20: FakeRetriever(HIT_DOCS)):
            events3 = await run_graph(graph, Command(resume={
                "reply": "算了", "history": []}), config3)
        trace3 = [e for e in events3 if e[0] == "trace"][0][1]
        review = [t for t in trace3 if t["step"] == "人工澄清"]
        check("澄清步骤标记「放弃」", review and review[0].get("outcome") == "放弃",
              str(review))
        check("走了 insufficient（生成回答为兜底文案）",
              any(e[0] == "answer" for e in events3))

        print("== 5. 急症问题 → 不追问，直接 insufficient ==")
        config4 = g.thread_config(999903)
        events4 = await run_graph(graph, {
            "question": "突发胸痛伴呼吸困难怎么办", "history": [], "trace": []}, config4)
        check("没有 interrupt", not any(e[0] == "interrupt" for e in events4))
        check("没有 clarification", not any(
            e[0] == "interrupt" and e[1].get("type") == "clarification_request"
            for e in events4))

        print("== 6. 挂起状态判据（main.py 的恢复检测）==")
        check("thread_config 生成 conv- 前缀",
              g.thread_config(42)["configurable"]["thread_id"] == "conv-42")

        # 单轮场景抓不到这个 bug：trace_acc 只收集本次 astream 的节点原始返回值，
        # 而累积发生在**跨轮**（检索子图作为父图节点继承上一轮落盘的 trace）。
        # 必须同一 thread 连问两轮才暴露。
        print("== 7. trace 按轮重置（checkpointer 跨轮不累积）==")
        config5 = g.thread_config(999904)
        with patch("app.graph.get_retriever", lambda k=20: FakeRetriever(HIT_DOCS)):
            ev_a = await run_graph(graph, {
                "question": "高血压患者日常注意什么", "history": [], "trace": []}, config5)
            ev_b = await run_graph(graph, {
                "question": "糖尿病患者饮食注意什么", "history": [], "trace": []}, config5)
        n_a = len([e for e in ev_a if e[0] == "trace"][0][1])
        n_b = len([e for e in ev_b if e[0] == "trace"][0][1])
        # 6 步 = 意图判定 / 查询改写 / 来源路由 / 混合检索 / 精排 / 忠实性校验
        # （阶段四给 generate 之后接了 ground_check，比之前多一步）
        check("首轮 trace 为单轮长度（6 步）", n_a == 6, f"{n_a} 条")
        check("次轮 trace 未累积（等于首轮）", n_b == n_a,
              f"首轮 {n_a} 条 → 次轮 {n_b} 条（累积即失败）")

        # 验证清单 #3（开关回退）的**核心不变量**：追问能力按"这张图有没有
        # checkpointer"判定，而不是读 HUMAN_REVIEW_ENABLED。评测脚本走
        # build_graph() 不带 checkpointer，若按配置放行就会在没有 checkpointer
        # 的图上调 interrupt() 直接抛错（5 条 out_of_corpus 样本全崩）。
        # 关 .env 开关是同一个不变量的人工复现，这里用"不传 checkpointer"离线覆盖。
        print("== 8. 无 checkpointer → 不进 human_review ==")
        graph_nocp = g.build_graph()          # 刻意不传 checkpointer
        ev_nocp = await run_graph(graph_nocp, {
            "question": "某库外生僻问题", "history": [], "trace": []},
            g.thread_config(999905))
        check("没有 interrupt（没有 checkpointer 就不该暂停）",
              not any(e[0] == "interrupt" for e in ev_nocp),
              str([e[0] for e in ev_nocp]))
        check("落到 insufficient（有兜底回答，而不是抛错）",
              any(e[0] == "answer" for e in ev_nocp))

        # 阶段二 §4 记账项：checkpointer 的 sqlite 独立于 MySQL，删会话不会波及它，
        # 不清理就无限堆积（每轮都写 docs/context 全文）。这里用上面的 InMemorySaver
        # 验证清理动作本身：thread 999901 在第 1 步已因 interrupt 落了快照。
        print("== 9. 删除会话 → checkpoint 快照被清理 ==")
        config_del = g.thread_config(999901)
        before = await saver.aget_tuple(config_del)
        check("删除前该 thread 有快照（前置条件成立）", before is not None)

        async def _fake_get_checkpointer():
            return saver

        with patch.object(cp, "get_checkpointer", _fake_get_checkpointer), \
             patch.object(cp.settings, "graph_checkpointer_enabled", True):
            cleaned = await cp.delete_thread(999901)
        after = await saver.aget_tuple(config_del)
        check("delete_thread 返回 True", cleaned is True)
        check("删除后快照消失", after is None, str(after))

        # 开关关掉时不该去建库、也不该调用 saver
        with patch.object(cp.settings, "graph_checkpointer_enabled", False):
            skipped = await cp.delete_thread(999902)
        check("checkpointer 关闭时不执行清理（返回 False）", skipped is False, str(skipped))

    print("\n离线管线自测完成。")
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项: {_FAILED}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
