# -*- coding: utf-8 -*-
"""对话行为分流评测（阶段一）：对抗样本集上的混淆矩阵 + 落点正确率

为什么需要它
------------
阶段一引入 `dialogue_act` 之后，"分类准不准"无法从既有指标看出来：
`hit_rate` / `mrr` 只测检索，而 ack/chitchat 根本不该检索 —— 混在一起互相稀释。
所以另开一份**对抗样本集**（`data/eval/adversarial_set.json`，64 条手工编写，
**不取自语料**），逐条走完整图，直接看分类结果与最终落点。

六组样本，各查一个方向（分组会随实测结论调整，见各条 note）：
    ack 10 / chitchat 10            漏词（「用不着」这类，关键词表结构性补不完）
    followup 10                     指代追问（含 top_score=0.355 的 F 案例）
    topic_switch 10                 自足新问题不得被并进旧话题
    short_medical 10                短真问题不得被规则门误伤（gate 的假阳性面）
    out_of_corpus 5 / rare_but_covered 2    库外必须兜底 / 罕见但库里有必须作答
    topic_match_no_answer 3         **主题高分命中但答不上来**（observe=true，见下）
    gate_misfire 4                  **已知缺陷留样**：规则门会误伤「好+症状」，见下

用法:
    cd backend
    python scripts/eval_dialogue.py --dry-run       # 校验样本集 + 规则门预判，不加载模型（秒级）
    python scripts/eval_dialogue.py --limit 5       # 先跑 5 条冒烟
    python scripts/eval_dialogue.py                 # 全量（会调 LLM，十几分钟）
    python scripts/eval_dialogue.py --group gate_misfire          # 只跑某一组
    python scripts/eval_dialogue.py --diagnose fu-08,ooc-08       # 定点诊断：把中间态全摊开
    python scripts/eval_dialogue.py --json data/eval/dialogue_results.json

三条硬断言（不依赖人眼看报告）
------------------------------
    ① 每轮 trace 里「意图判定」恰好 1 次（最容易踩的重复 bug）
    ② **无资料路径（insufficient / chat）的回答里不得出现引用编号**
       —— 直接对应最初那个 bug：context 为空时 MEDICAL_PROMPT 规则 9 会逼模型硬编 [1]
    ③ insufficient 路径必须带免责声明（代码确定性追加，不该丢）

`observe: true` 是什么
----------------------
「主题高分命中但答不上来」这一类的**期望落点尚未定义**（现状走 generate、理想该走
insufficient），硬塞一个假期望进分母只会掩盖问题。所以标 observe 的条目：
act 照常判（act 期望是明确的），**落点只观测不判定**，在报告里单列出来看事实。

⚠️ 已知缺陷：`gate_misfire` 那 4 条**当前会失败**，属预期
----------------------------------------------------------
`app/intent.py` 的 `_ACK_WORDS` 含单字「好」，而匹配用的是 `w in q`（子串），
于是任何「好 + 症状」且该症状不在 `_MEDICAL_HINTS` 里的输入都会被判成会话语，
**直接拒答**，且阶段一的 LLM 分类器**没有机会纠正**（第一层就拦掉了）。
2026-09-25 实测确认 3 条（好恶心 / 好困 / 好难受），第 4 条（好胀）是同一模式的
泛化探针。⇒ 跑全量时用 `--fail-threshold 4`，让退出码反映"除这批已知缺陷外无其它方向性错误"。

它读回四样东西
--------------
    dialogue_act  —— 改写那次 LLM 调用顺带产出的对话行为（零额外延迟）
    最终节点      —— 由 trace 的最后一步反推：chat / generate / insufficient
    top_score     —— 精排最高分（顺手收集，用来标定相关性闸门阈值）
    trace         —— 顺带校验不变量「每轮『意图判定』恰好 1 次」

⚠️ 为什么"没有 dialogue_act"不等于漏分类
----------------------------------------
分流有两层：入口**纯规则门**（`app/intent.is_conversational`）+ LLM 对话行为分类。
被规则门拦下的样本根本不会调用改写，于是没有 `dialogue_act` —— 这是设计如此
（零成本拦下明显的寒暄），不是失败。报告里单列成 `gate`：
act 准确率不计入分母，但**最终落点**照样要判对。

两类错误的代价不一样（报告里分开计数）
--------------------------------------
  医学问题 → 被判成会话：**最危险**。拒答一个真问题，用户拿不到任何医学信息。
  会话/医学问诊 → 被判成 new_question：白检索一轮，可能又复述一遍旧病（原始 bug）。

可复现性
--------
改写走零温 + 磁盘缓存，所以同一份样本集重跑得到逐字相同的 act。
若刚改过改写 Prompt（指纹变化），首跑会全量重算；想强制重算，删
`backend/data/cache/rewrite_cache.json`。
"""
import sys
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# 引用编号：原「假引用」bug 的观测指标（context 为空时 MEDICAL_PROMPT 会逼模型硬编一个 [1]）
_CITATION_RE = re.compile(r"\[\d+\]")
# 免责声明文案（与 app/rag_chain.py 同源，避免报告里出现两份不一致的常量）
from app.rag_chain import DISCLAIMER  # noqa: E402

DEFAULT_SET = BACKEND / "data" / "eval" / "adversarial_set.json"
# 逐条原始结果**默认就落盘**。曾因为不落盘，报告（stdout）和上一轮遗留的旧
# json 被当成同一次结果读，白排查了一轮；落盘成本约等于 0，别省。
DEFAULT_JSON = BACKEND / "data" / "eval" / "dialogue_results.json"

# trace 的「最后一步」→ 最终落到哪个节点。
# generate 路径没有专属步骤名，它的最后一步是「精排」。
ROUTE_BY_STEP = {
    "对话式回答": "chat",
    "证据不足兜底": "insufficient",
    "精排": "generate",
}
ROUTE_LABEL = {
    "chat": "对话回应（未检索）",
    "generate": "医学作答（有资料）",
    "insufficient": "证据不足兜底",
    "unknown": "?? 未知",
}


# ============================================================================
# 样本集校验（不依赖 Milvus / LLM / 模型权重）
# ============================================================================
def load_set(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(items: list[dict]) -> list[str]:
    from app.dialogue_policy import all_acts

    legal_acts = set(all_acts())
    legal_routes = set(ROUTE_LABEL) - {"unknown"}
    problems: list[str] = []
    seen: set[str] = set()

    for i, it in enumerate(items):
        tag = it.get("id") or f"#{i}"
        if not str(it.get("question", "")).strip():
            problems.append(f"{tag}: question 为空")
        if it.get("expect_act") not in legal_acts:
            problems.append(f"{tag}: expect_act={it.get('expect_act')!r} 不在 {sorted(legal_acts)}")
        if it.get("expect_route") not in legal_routes:
            problems.append(f"{tag}: expect_route={it.get('expect_route')!r} 非法")
        if tag in seen:
            problems.append(f"{tag}: id 重复")
        seen.add(tag)
        for m in it.get("history") or []:
            if m.get("role") not in ("user", "assistant") or not str(m.get("content", "")).strip():
                problems.append(f"{tag}: history 条目非法 {m!r}")
    return problems


def describe_set(items: list[dict]) -> None:
    by_group = Counter(it.get("group", "?") for it in items)
    print(f"样本集：{len(items)} 条")
    for g, n in by_group.most_common():
        acts = Counter(it.get("expect_act") for it in items if it.get("group") == g)
        routes = Counter(it.get("expect_route") for it in items if it.get("group") == g)
        print(f"  {g:<14} {n:>3} 条   expect_act={dict(acts)}  expect_route={dict(routes)}")
    n_multi = sum(1 for it in items if it.get("history"))
    print(f"  其中带多轮历史 {n_multi} 条 / 单轮 {len(items) - n_multi} 条")


def gate_preview(items: list[dict]) -> list[str]:
    """用规则门（`app.intent.is_conversational`，纯规则、零 LLM）预判拦截结果。

    为什么值得单列一步：分流是两层，第一层是**纯规则**，所以它的表现可以在
    跑全量之前就完全算出来。而它最容易犯的错是**误伤**——把一条短的真医学问题
    （「头痛」「好痛」）当成附和对白拦下，直接拒答。这是本项目最贵的错误类型，
    且一旦发生，第二层的 LLM 分类器**根本没机会纠正**（压根不会被调用）。

    返回"期望非会话却被拦下"的 id 列表。
    """
    from app.intent import is_conversational
    from app.dialogue_policy import NON_RETRIEVAL_ACTS
    from app.config import settings

    print()
    print("规则门预判（app/intent.is_conversational，纯规则、不调 LLM）")
    print(f"  判据：归一化后长度 ≤ {settings.intent_chat_max_chars}"
          f" → 先查医学线索词（命中即放行）→ 再匹配确认词表；门开关 intent_gate_enabled="
          f"{settings.intent_gate_enabled}")

    gated, risky = [], []
    for it in items:
        hit = is_conversational(it["question"])
        expect_conversational = it["expect_act"] in NON_RETRIEVAL_ACTS
        if hit:
            gated.append(it["id"])
        if hit and not expect_conversational:
            risky.append(it)

    print(f"  预计被拦下 {len(gated)}/{len(items)} 条：{', '.join(gated) or '（无）'}")
    if risky:
        print(f"  ⚠️ 其中 {len(risky)} 条**期望不是会话**却被拦下 —— 上线后必然误判，"
              f"且 LLM 分类器没有机会补救：")
        for it in risky:
            print(f"     {it['id']} 「{it['question']}」 期望 {it['expect_act']}")
    else:
        print("  ✅ 零误伤：期望非会话的样本全部通过规则门，第二层 LLM 分类器能拿到它们")
    if not settings.intent_gate_enabled:
        print("  注：intent_gate_enabled=False，规则门当前关闭，上述预判不会生效。")
    return [it["id"] for it in risky]


# ============================================================================
# 逐条执行
# ============================================================================
def route_of(trace: list[dict]) -> str:
    """由 trace 的最后一步反推最终节点。"""
    for step in reversed(trace):
        r = ROUTE_BY_STEP.get(step.get("step"))
        if r:
            return r
    return "unknown"


def run_item(graph, item: dict) -> dict:
    state = graph.invoke({
        "question": item["question"],
        "history": item.get("history") or [],
        "trace": [],
    })
    trace = state.get("trace") or []
    act = state.get("dialogue_act")
    sources = state.get("sources") or []
    return {
        "act": act,
        "route": route_of(trace),
        # act 缺失有两种来源：入口规则门拦下（没走改写）。这里据此区分"未分类"。
        "gated": act is None,
        "top_score": state.get("top_score"),
        "evidence": state.get("evidence_state"),
        "n_sources": len(sources),
        # ⚠️ 下面三个字段是**诊断失败案例的必需品**，不要为了"报告干净"删掉：
        #   没有 enhanced_query 就分不清"分低"是改写没补全、还是库里真没有；
        #   没有 sources 就分不清"高分命中"是真相关、还是 rerank 误判。
        "enhanced_query": state.get("enhanced_query"),
        "scored_by": next((s.get("scored_by") for s in trace if s.get("step") == "精排"), None),
        "sources_head": [s.get("title") for s in sources[:3]],
        "answer_head": (state.get("answer") or "").replace("\n", " ")[:48],
        "answer_tail": (state.get("answer") or "").replace("\n", " ")[-100:],
        # 引用一致性：这两项是原「假引用」bug 的直接观测指标。
        # 无资料路径**必须** has_citation=False（Prompt 明令禁止 + 代码保证）；
        # 有资料路径应当 has_disclaimer=True（代码确定性追加）。
        "has_citation": bool(_CITATION_RE.search(state.get("answer") or "")),
        "has_disclaimer": DISCLAIMER in (state.get("answer") or ""),
        "steps": [s.get("step") for s in trace],
        "n_intent_step": sum(1 for s in trace if s.get("step") == "意图判定"),
    }


# ============================================================================
# 报告
# ============================================================================
def diagnose(graph, items: list[dict], ids: list[str]) -> None:
    """定点诊断：只跑指定的几条，把中间态全摊开。

    用途是"改完一处后再看那几条"——比每次重跑 64 条快一个数量级（改写缓存命中后
    每条只剩检索 + 生成）。**判断"分低"是改写没补全还是库里真没有，只能靠这里。**
    """
    want = {i.strip() for i in ids if i.strip()}
    picked = [it for it in items if it["id"] in want]
    missing = want - {it["id"] for it in picked}
    if missing:
        print(f"⚠️ 样本集里找不到这些 id：{sorted(missing)}")
    if not picked:
        return

    print()
    print("=" * 78)
    print(f"定点诊断：{len(picked)} 条")
    print("=" * 78)
    for it in picked:
        res = run_item(graph, it)
        print(f"\n── {it['id']}  [{it.get('group')}]")
        print(f"   输入      : {it['question']}")
        if it.get("history"):
            print(f"   历史      : " + " | ".join(
                f"{m['role'][:1]}:{m['content'][:28]}" for m in it["history"]))
        print(f"   act       : 期望 {it['expect_act']} → 实测 {res['act'] or ('gate' if res['gated'] else '—')}"
              f"{'   ✅' if (res['gated'] or res['act'] == it['expect_act']) else '   ❌'}")
        print(f"   改写 query: {res['enhanced_query'] or '（未走改写 · 被规则门拦下）'}")
        print(f"   打分依据  : {res['scored_by'] or '—（未检索）'}")
        print(f"   top_score : {res['top_score'] if res['top_score'] is not None else '—'}")
        print(f"   证据      : {res['evidence'] or '—'}   来源 {res['n_sources']} 条 {res['sources_head']}")
        print(f"   路径      : {' → '.join(res['steps'])}")
        print(f"   回答开头  : {res['answer_head']}")
        print(f"   回答结尾  : …{res['answer_tail']}")
        print(f"   引用一致性: 含引用编号={res['has_citation']}  含免责声明={res['has_disclaimer']}"
              + ("   ← ⚠️ 无资料却带引用编号" if (res["route"] in ("insufficient", "chat")
                                                and res["has_citation"]) else ""))


def report(items, results, elapsed, cache_stats, fail_threshold):
    from app.dialogue_policy import NON_RETRIEVAL_ACTS

    n = len(items)
    act_total = act_hit = 0
    route_hit = 0
    n_gated = 0
    wrong_act = []          # (item, res, kind)
    bad_route = []
    bad_invariant = []
    bad_citation = []       # 无资料路径却带引用编号 = 原始假引用 bug 的形态
    bad_disclaimer = []     # 有资料路径缺免责声明
    observed = []           # observe=true：只观测，不判落点（期望本身还没定义）
    group_stat = defaultdict(lambda: {"n": 0, "act_ok": 0, "act_judged": 0,
                                      "route_ok": 0, "route_n": 0})

    for it, res in zip(items, results):
        g = it.get("group", "?")
        st = group_stat[g]
        st["n"] += 1

        if res["n_intent_step"] != 1:
            bad_invariant.append((it, res))
        # 引用一致性（硬约束）：没资料就没有可引的东西，出现编号即伪造出处
        if res["route"] in ("insufficient", "chat") and res["has_citation"]:
            bad_citation.append((it, res))
        if res["route"] == "insufficient" and not res["has_disclaimer"]:
            bad_disclaimer.append((it, res))
        if res["gated"]:
            n_gated += 1
        else:
            act_total += 1
            st["act_judged"] += 1
            if res["act"] == it["expect_act"]:
                act_hit += 1
                st["act_ok"] += 1

        # observe=true → 只观测 act（act 期望是明确的），**不进落点分母**：
        # 这一类的"期望落点"本身是个尚未定下的产品决策（现状 generate / 理想 insufficient），
        # 硬塞一个假期望进分母只会掩盖问题，不如单列出来看实测。
        if it.get("observe"):
            observed.append((it, res))
        else:
            st["route_n"] += 1
            if res["route"] == it["expect_route"]:
                route_hit += 1
                st["route_ok"] += 1
            else:
                bad_route.append((it, res))

        exp_conversational = it["expect_act"] in NON_RETRIEVAL_ACTS
        got_conversational = (res["act"] in NON_RETRIEVAL_ACTS) if not res["gated"] else True
        if exp_conversational != got_conversational:
            kind = "医学→会话（拒答风险）" if not exp_conversational else "会话→医学（复述旧病风险）"
            wrong_act.append((it, res, kind))

    print()
    print("=" * 78)
    print("逐条结果")
    print("=" * 78)
    print(f"{'id':<12}{'期望':<13}{'实测':<13}{'落点':<26}top  引用 判定")
    print("-" * 84)
    for it, res in zip(items, results):
        got = res["act"] or ("gate" if res["gated"] else "—")
        act_ok = res["gated"] or res["act"] == it["expect_act"]
        # observe=true 的条目不对落点判对错（期望本身没定），标出来即可
        is_obs = bool(it.get("observe"))
        route_ok = True if is_obs else (res["route"] == it["expect_route"])
        cite_ok = not (res["route"] in ("insufficient", "chat") and res["has_citation"])
        marks = []
        if not act_ok:
            marks.append("act✗")
        if is_obs:
            marks.append("观测")
        elif not route_ok:
            marks.append("落点✗")
        if not cite_ok:
            marks.append("引用✗")
        mark = "OK" if not marks else " ".join(marks)
        ts = res["top_score"]
        ts_s = f"{ts:.3f}" if isinstance(ts, (int, float)) else "  —  "
        cite_s = "有" if res["has_citation"] else "无"
        print(f"{it['id']:<12}{it['expect_act']:<13}{got:<13}"
              f"{ROUTE_LABEL.get(res['route'], res['route']):<26}{ts_s}  {cite_s:<4} {mark}")
        if it.get("note") and mark != "OK":
            print(f"{'':<12}↳ {it['note']}")

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print("混淆矩阵（行=期望 act，列=实测 act；gate=入口规则门拦下，未走改写）")
    cols = ["new_question", "followup", "ack", "chitchat", "gate", "—"]
    print(f"{'':<16}" + "".join(f"{c:>14}" for c in cols))
    for exp in ["new_question", "followup", "ack", "chitchat"]:
        row = Counter()
        for it, res in zip(items, results):
            if it["expect_act"] != exp:
                continue
            row[("gate" if res["gated"] else (res["act"] or "—"))] += 1
        print(f"{exp:<16}" + "".join(f"{row.get(c, 0):>14}" for c in cols))

    print()
    print(f"{'分组':<20}{'条数':>5}{'act准':>10}{'可判':>6}{'落点准':>11}")
    for g, st in sorted(group_stat.items()):
        act_s = f"{st['act_ok']}/{st['act_judged']}" if st["act_judged"] else "（全被规则门拦）"
        route_s = (f"{st['route_ok']}/{st['route_n']}" if st["route_n"] else "（仅观测）")
        print(f"{g:<20}{st['n']:>5}{act_s:>14}{st['act_judged']:>6}{route_s:>13}")

    print()
    print(f"act 准确率（不含被规则门拦下的 {n_gated} 条）："
          f"{act_hit}/{act_total}" + (f" = {act_hit / act_total:.1%}" if act_total else "  —"))
    route_n = n - len(observed)
    print(f"最终落点正确率：{route_hit}/{route_n} = {route_hit / route_n:.1%}"
          + (f"（另有 {len(observed)} 条 observe 只观测、不计入）" if observed else ""))
    print(f"入口规则门拦截：{n_gated}/{n} 条（这些不需 LLM 分类即可判为会话）")

    if wrong_act:
        print()
        print(f"⚠️ 行为方向性错误 {len(wrong_act)} 条：")
        for it, res, kind in wrong_act:
            print(f"   [{kind}] {it['id']} 「{it['question']}」 期望 {it['expect_act']} → 实测 {res['act']}")
    else:
        print()
        print("✅ 行为方向性错误 0 条（没有把医学问题判成会话，也没有把会话判成医学）")

    if bad_route:
        print(f"\n落点不符 {len(bad_route)} 条（含上文的行为错误，也可能只是闸门阈值问题）：")
        for it, res in bad_route:
            print(f"   {it['id']} 「{it['question']}」 期望 {it['expect_route']} → 实测 {res['route']}"
                  f"  top_score={res['top_score']}")

    if observed:
        print()
        print(f"观测项 {len(observed)} 条（observe=true，期望落点尚未定义，只看事实）：")
        for it, res in observed:
            print(f"   {it['id']} 「{it['question'][:34]}」")
            print(f"      top_score={res['top_score']:.3f}  落点={ROUTE_LABEL.get(res['route'])}"
                  f"  引用编号={'有' if res['has_citation'] else '无'}")
            print(f"      模型回答：{res['answer_head']}")

    print()
    if bad_invariant:
        print(f"❌ 不变量被破坏：{len(bad_invariant)} 条 trace 里『意图判定』不是恰好 1 次")
        for it, res in bad_invariant:
            print(f"   {it['id']} n={res['n_intent_step']} steps={res['steps']}")
    else:
        print("✅ 不变量通过：每条 trace 里『意图判定』恰好 1 次")

    # 引用一致性：直接对应最初那个 bug（context 为空时被 Prompt 逼着硬编 [1]）
    if bad_citation:
        print(f"❌ 引用一致性被破坏：{len(bad_citation)} 条**无资料路径却带引用编号**（伪造出处）")
        for it, res in bad_citation:
            print(f"   {it['id']} route={res['route']} 「{res['answer_tail'][-60:]}」")
    else:
        print("✅ 引用一致性通过：无资料路径（insufficient/chat）的回答里没有任何引用编号")
    if bad_disclaimer:
        print(f"❌ {len(bad_disclaimer)} 条 insufficient 路径缺免责声明（代码应确定性追加）")
        for it, res in bad_disclaimer:
            print(f"   {it['id']} 「{it['question']}」")

    ooc = [(it, r) for it, r in zip(items, results)
           if it.get("group") == "out_of_corpus" and isinstance(r["top_score"], (int, float))]
    if ooc:
        scores = sorted(r["top_score"] for _, r in ooc)
        print(f"\n库外问题的 top_score 分布（标定相关性闸门阈值用，当前阈值见 settings.rerank_score_threshold）：")
        print("   " + "  ".join(f"{s:.3f}" for s in scores))
        print(f"   最高 {scores[-1]:.3f} / 中位 {scores[len(scores) // 2]:.3f} / 最低 {scores[0]:.3f}")
        print("   若最高分仍低于阈值 → 闸门把库外问题判成『无证据』，兜底路径正确触发。")

    print(f"\n改写缓存：{cache_stats}")
    print(f"总耗时：{elapsed:.1f}s（{elapsed / n:.1f}s/条）")

    verdict_fail = (len(wrong_act) > fail_threshold or bool(bad_invariant)
                    or bool(bad_citation) or bool(bad_disclaimer))
    print()
    print("=" * 78)
    if verdict_fail:
        print(f"结论：未达门槛 —— 方向性错误 {len(wrong_act)} 条（上限 {fail_threshold}）"
              + ("；不变量被破坏" if bad_invariant else "")
              + ("；引用一致性被破坏" if bad_citation else "")
              + ("；缺免责声明" if bad_disclaimer else ""))
    else:
        print(f"结论：通过 —— 方向性错误 {len(wrong_act)} 条 ≤ 上限 {fail_threshold}，"
              f"不变量与引用一致性均成立")
    print("=" * 78)
    return 0 if not verdict_fail else 1


def main():
    ap = argparse.ArgumentParser(description="对话行为分流评测（阶段一）")
    ap.add_argument("--set", default=str(DEFAULT_SET), help="样本集路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--group", default=None, help="只跑某个分组，如 followup")
    ap.add_argument("--dry-run", action="store_true", help="只校验样本集，不加载模型")
    ap.add_argument("--diagnose", default=None,
                    help="定点诊断：逗号分隔的 id 列表，只跑这几条并把中间态全摊开")
    ap.add_argument("--json", default=str(DEFAULT_JSON),
                    help="把逐条原始结果落盘到该路径（默认就落盘，避免报告与逐条记录不同源）")
    ap.add_argument("--fail-threshold", type=int, default=0,
                    help="允许的方向性错误条数上限（默认 0，超出则退出码非 0）")
    args = ap.parse_args()

    items = load_set(Path(args.set))
    print("=" * 78)
    print(f"样本集：{args.set}")
    print("=" * 78)
    describe_set(items)

    problems = validate(items)
    if problems:
        print()
        print(f"❌ 样本集有 {len(problems)} 处问题：")
        for p in problems:
            print("   " + p)
        return 2
    print("\n✅ 样本集校验通过（id 唯一、expect_act/expect_route 合法、history 合法）")

    # 规则门预判：跑全量之前先看第一层会不会误伤（对**全量**算，不受 --group/--limit 影响）
    gate_preview(items)

    if args.group:
        items = [it for it in items if it.get("group") == args.group]
        print(f"（--group {args.group} → 只跑 {len(items)} 条）")
    if args.limit:
        items = items[:args.limit]
        print(f"（--limit {args.limit} → 只跑前 {len(items)} 条）")

    if args.dry_run:
        print("\n--dry-run：跳过图执行。")
        return 0

    # 延迟导入：dry-run 与样本集校验不需要 Milvus / reranker / 嵌入模型
    from app.graph import build_graph
    from app.query_rewriter import rewrite_cache_stats

    print("\n加载完整图（首次会初始化检索与重排组件，请稍候）…")
    graph = build_graph()

    # 定点诊断：只跑指定几条，不打印汇总报表（诊断模式与批量评测是两种用途）
    if args.diagnose:
        diagnose(graph, load_set(Path(args.set)), args.diagnose.split(","))
        return 0

    print(f"开始逐条执行 {len(items)} 条…")
    results = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        print(f"  [{i:>2}/{len(items)}] {it['id']:<12} {it['question'][:36]}", flush=True)
        results.append(run_item(graph, it))
    elapsed = time.time() - t0

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = BACKEND / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"set": args.set, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "elapsed_sec": round(elapsed, 1),
             "items": [{**it, "result": res} for it, res in zip(items, results)]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n逐条结果已落盘：{out}")

    return report(items, results, elapsed, rewrite_cache_stats(), args.fail_threshold)


if __name__ == "__main__":
    sys.exit(main())
