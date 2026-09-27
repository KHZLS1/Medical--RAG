"""阶段二澄清流程的端到端自测（需要全栈：后端 + Milvus + LLM key）

对应《实施计划_阶段二》验证清单 #1~#10 中的可自动化项：
  1. 普通问答回归（有引用、无追问）
  2. 会话语回归（「好的/谢谢」不检索、不追问）
  5. 生僻问题触发追问
  6. 补充后命中（trace 无重复 = D4）
  7. 回复「算了」→ 放弃收尾（无引用编号）
  8. 急症词不追问
  9. 澄清气泡落库
 10. checkpoint sqlite 存在

不在本脚本的两项：
  #3 开关回退（HUMAN_REVIEW_ENABLED=false）需改 .env 并重启后端，属手工步骤；
     其核心不变量「没有 checkpointer 就不进 human_review」已由
     scripts/test_clarify_flow.py 的**离线用例 8** 覆盖（不需要 Milvus）。
  #4 旧接口回归：跑 scripts/test_backend_api.py（独立脚本，自带断言与退出码）。
     该脚本刻意不碰 /api/ingest（项目规约：入库只走 scripts/ingest.py）。

⚠️ 样本选取是这套测试的**前提**，且必须实测过
------------------------------------------------
之前用「戈谢病的酶替代疗法」当"库里查不到"的样例，看着生僻，其实库里 5 条资料、
top_score 0.999 —— 追问分支压根不触发，#5/#7/#9 连带失败，还误以为是产品 bug。
下面三个常量都用真实 Milvus 标定过（数值见各自注释）。要换问题：把候选填进
`OOC_QUESTION` 直接跑本脚本即可 —— `premise_ok()` 会在断言前校验"这题到底在不在
库里"，并把 top_score / 闸门一起报出来，前提不成立会明确喊话，不需要另写探针。

跑法（Milvus 与后端已启动）：
  python scripts/test_clarify_e2e.py [--base http://localhost:8000]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

BASE = "http://localhost:8000"
TIMEOUT = 180   # 单轮 SSE 总时长上限（改写+检索+生成）

# 真·库外问题：实测 top_score=0.137 < 闸门 0.30 → evidence=none → 触发追问
OOC_QUESTION = "库欣病经蝶窦手术后的皮质醇缓解判定标准是什么"
# 对上面的补充：融合后实测 top_score=0.603 → evidence=strong，可验证"补充后命中"
OOC_SUPPLEMENT = "腹部胀大，脾脏肿大三年"
# 急症 + 库外：含急症词「昏迷」，且实测 top_score=0.119 < 闸门 → 不追问、直接兜底
EMERGENCY_OOC = "库欣病术后肾上腺危象昏迷怎么抢救"


def post_chat(question: str, conversation_id: int | None = None):
    """POST /api/chat 并解析 SSE，返回 (events, conversation_id)"""
    events = []
    conv_id = conversation_id
    resp = requests.post(
        f"{BASE}/api/chat",
        json={"question": question, "conversation_id": conversation_id},
        stream=True,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        events.append(evt)
        if evt.get("type") == "conversation_id":
            conv_id = evt["data"]
    return events, conv_id


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"\n         ↳ {detail}" if detail and not cond else ""))
    return cond


def evt_types(events):
    return [e["type"] for e in events]


def last_trace(events):
    """取**最后一条** trace 事件的数据。

    ⚠️ 不能取第一条。`graph._astream_run` 在**每个节点更新后**都推一次
    **累计**数组（前端整体替换，见 astream_answer 的 yield 顺序注释），所以
    最后一条才是完整时间线。resume 路径尤其明显：
        第 1 条 trace = [人工澄清]            ← human_review 返回时推的
        第 2 条 trace = [人工澄清, 第二轮 5 步] ← requery 返回时推的
    只读第一条会得出"trace 里没有第二轮步骤"的错误结论。
    """
    traces = [e["data"] for e in events if e["type"] == "trace"]
    return traces[-1] if traces else []


def steps_of(events):
    return [t["step"] for t in last_trace(events)]


def rerank_of(events):
    """取「精排」那一步，拿 top_score / gate，用于判定样本前提是否成立。"""
    return next((t for t in last_trace(events) if t.get("step") == "精排"), {})


def answer_of(events):
    return "".join(e["data"] for e in events if e["type"] == "token")


def has_sources(events):
    return any(e["type"] == "sources" and e["data"] for e in events)


def premise_ok(events, expect: str) -> bool:
    """校验"这题库外/库内"的前提是否成立；不成立就明确说出来，别让断言失败误导方向。"""
    rr = rerank_of(events)
    gate = rr.get("gate", "")
    score = rr.get("top_score")
    gated = gate.startswith("低于阈值")
    if expect == "out_of_corpus" and not gated:
        print(f"  ⚠️ 前提失效：该问题库中其实有资料（top_score={score}，闸门通过）→ "
              f"追问分支不会触发。请换一个真·库外的问题（先跑探针确认）。")
        return False
    if expect == "in_corpus" and gated:
        print(f"  ⚠️ 前提失效：该问题库中其实没有资料（top_score={score}，被闸门置空）→ "
              f"无法验证命中路径。请换问题。")
        return False
    return True


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base

    r = requests.get(f"{BASE}/api/health", timeout=10)
    r.raise_for_status()
    print(f"后端健康: {r.json()}\n")

    # ===== #1 普通问答回归 =====
    print("== #1 普通问答回归：小孩发烧39度怎么办 ==")
    events, conv1 = post_chat("小孩发烧39度怎么办")
    types = evt_types(events)
    check("有 token 流", "token" in types)
    check("有 sources", has_sources(events))
    check("没有 clarification_request", "clarification_request" not in types, str(types))
    steps1 = steps_of(events)
    check("trace 无「人工澄清」", "人工澄清" not in steps1, str(steps1))
    check("message_id 已回传", "message_id" in types)

    # ===== #2 会话语回归 =====
    print(f"\n== #2 会话语回归（conv {conv1}）：连发「好的」「谢谢」==")
    for q in ("好的", "谢谢"):
        events, conv1 = post_chat(q, conv1)
        types = evt_types(events)
        steps = steps_of(events)
        check(f"「{q}」有回答", "token" in types, str(types))
        check(f"「{q}」走对话路径（trace 含「对话式回答」）", "对话式回答" in steps, str(steps))
        check(f"「{q}」不检索（无 sources）", not has_sources(events), str(types))
        check(f"「{q}」不追问", "clarification_request" not in types, str(types))

    # ===== #5 触发追问 =====
    print(f"\n== #5 触发追问（conv {conv1}）：{OOC_QUESTION} ==")
    events, conv1 = post_chat(OOC_QUESTION, conv1)
    types = evt_types(events)
    premise_ok(events, "out_of_corpus")
    clarify = next((e for e in events if e["type"] == "clarification_request"), None)
    check("收到 clarification_request", clarify is not None, str(types))
    check("追问话术非空", bool(clarify and clarify["data"].get("message")))
    check("本轮无 token", "token" not in types)
    check("本轮无 sources", not has_sources(events))
    check("message_id 已回传（澄清气泡可反馈）", "message_id" in types)

    # ===== #6 补充后命中 =====
    print(f"\n== #6 补充后命中（conv {conv1}）：{OOC_SUPPLEMENT} ==")
    events, conv1 = post_chat(OOC_SUPPLEMENT, conv1)
    types = evt_types(events)
    premise_ok(events, "in_corpus")
    check("恢复后有 token 流", "token" in types, str(types))
    check("有 sources", has_sources(events))
    steps6 = steps_of(events)
    check("含「人工澄清」", "人工澄清" in steps6, str(steps6))
    check("含第二轮「意图判定」", "意图判定" in steps6, str(steps6))
    n_intent = steps6.count("意图判定")
    check("「意图判定」恰好 1 次（D4：无重复）", n_intent == 1, f"{n_intent} 次: {steps6}")

    # ===== #9 澄清气泡落库 =====
    print(f"\n== #9 澄清气泡落库（conv {conv1}） ==")
    r = requests.get(f"{BASE}/api/conversations/{conv1}/messages", timeout=30)
    msgs = r.json()["messages"]
    clarify_saved = any(
        m["role"] == "assistant" and "暂时没有找到" in m["content"] for m in msgs
    )
    check("追问话术已作为助手消息落库", clarify_saved,
          str([f'{m["role"]}:{m["content"][:20]}' for m in msgs]))

    # ===== #7 放弃追问 =====
    print("\n== #7 放弃追问（新会话） ==")
    events, conv2 = post_chat(OOC_QUESTION)
    types = evt_types(events)
    premise_ok(events, "out_of_corpus")
    check("先收到追问", "clarification_request" in types, str(types))
    events, conv2 = post_chat("算了", conv2)
    types = evt_types(events)
    answer = answer_of(events)
    check("放弃后有兜底回答", bool(answer.strip()))
    check("兜底回答无引用编号 [n]", "[" not in answer, answer[:120])
    trace7 = last_trace(events)
    review7 = [t for t in trace7 if t["step"] == "人工澄清"]
    check("澄清步骤标记「放弃」", review7 and review7[0].get("outcome") == "放弃",
          str(trace7))

    # ===== #8 急症不追问 =====
    print(f"\n== #8 急症不追问（库外急症）：{EMERGENCY_OOC} ==")
    events, conv3 = post_chat(EMERGENCY_OOC)
    types = evt_types(events)
    answer8 = answer_of(events)
    # 急症前提：必须"库外"（闸门置空）才测得到"急症覆盖追问"这一条；
    # 若库里有资料，走的是正常作答，测不到急症分支。
    if premise_ok(events, "out_of_corpus"):
        check("急症且无资料 → 不追问", "clarification_request" not in types, str(types))
        check("直接走兜底（有回答）", bool(answer8.strip()))
        check("兜底回答无引用编号 [n]", "[" not in answer8, answer8[:120])

    # ===== #10 checkpoint 落盘 =====
    print("\n== #10 checkpoint 落盘 ==")
    db = Path(__file__).resolve().parent.parent / "data/cache/graph_checkpoints.sqlite"
    check("graph_checkpoints.sqlite 存在", db.is_file(), str(db))

    # ===== 未覆盖项提示 =====
    print("\n== 未在本脚本覆盖的两项 ==")
    print("  #3 开关回退：改 .env 的 HUMAN_REVIEW_ENABLED=false 并重启后端后手工复跑；")
    print("     核心不变量（无 checkpointer 不进 human_review）见 test_clarify_flow.py")
    print("  #4 旧接口回归：另跑 python scripts/test_backend_api.py")

    print("\n端到端自测完成。")


if __name__ == "__main__":
    main()