"""阶段四 忠实性校验（Groundedness Check）的离线自测

零外部依赖：不需要 Milvus / LLM / MySQL。对应 `实施计划_阶段四_忠实性校验.md` §3：

  A. 纯函数（§3.1，用例 1~11）—— L1 `check_citations` + L2 `parse_unsupported`
     与开关回退。零 LLM，可硬断言。
  B. 图级 / 节点级（§3.2，用例 12~16）—— 验接线与 fail-open：
       generate → ground_check → END 生效、chat / insufficient 不接校验节点、
       L2 异常时文本一字不改（校验层不得成为故障点）。
  C. 生成侧自省后处理（T59 改道，用例 17~21）—— 回答自认"资料答不了"时剥掉
     全部引用编号（partial 三档已证伪后的替代方案）。含位置约束的反例。

跑法（backend 目录）：
  python scripts/test_groundedness.py
退出码 0 = 全绿。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from app import graph as g
from app.config import settings
from app.query_rewriter import RewriteResult
from app.rag_chain import (
    GROUNDING_HEDGE,
    check_citations,
    detect_unanswerable,
    parse_unsupported,
    strip_all_citations,
)

_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return bool(cond)


def sources(n: int) -> list[dict]:
    return [{"index": i, "title": f"资料{i}", "department": "内科", "source": "t"}
            for i in range(1, n + 1)]


# ============================================================================
# A. 纯函数（§3.1 用例 1~10）
# ============================================================================
def case_1_normal():
    print("== 1. 正常：3 条资料，引 [1][2][3] ==")
    ans = "病因与遗传有关[1]，建议检查[2]，注意饮食[3]。"
    cleaned, invalid, cited = check_citations(ans, sources(3))
    check("cleaned 一字不改", cleaned == ans, cleaned)
    check("invalid 为空", invalid == [], str(invalid))
    check("cited=[1,2,3]", cited == [1, 2, 3], str(cited))


def case_2_out_of_range():
    print("== 2. 越界：3 条资料，引 [1][7] ==")
    cleaned, invalid, cited = check_citations("高血压可用党参[1]，但需遵医嘱[7]。", sources(3))
    check("invalid=[7]", invalid == [7], str(invalid))
    check("cleaned 里 [7] 已消失", "[7]" not in cleaned, cleaned)
    check("cleaned 里 [1] 保留", "[1]" in cleaned, cleaned)
    check("cited 记全部", cited == [1, 7], str(cited))


def case_3_all_out_of_range():
    print("== 3. 全越界：0 条资料，引 [1] ==")
    cleaned, invalid, cited = check_citations("某种说法[1]。", [])
    check("invalid=[1]", invalid == [1], str(invalid))
    check("cleaned 不含任何 [n]", not any(f"[{n}]" in cleaned for n in range(1, 10)), cleaned)


def case_4_duplicate():
    print("== 4. 重复引用 [2][2] ==")
    cleaned, invalid, cited = check_citations("见资料[2]，另见资料[2]。", sources(3))
    check("cited 去重 == [2]", cited == [2], str(cited))
    check("invalid 为空（不重复计入）", invalid == [], str(invalid))
    check("cleaned 不改", "[2]" in cleaned)


def case_5_two_digit():
    print("== 5. 两位数 [10] + 10 条资料 ==")
    _, invalid10, cited10 = check_citations("依据[10]。", sources(10))
    check("[10] 合法", invalid10 == [] and cited10 == [10], f"{invalid10} {cited10}")
    _, invalid11, _ = check_citations("依据[11]。", sources(10))
    check("[11] 判越界", invalid11 == [11], str(invalid11))
    # 年份不应被当成编号：[2024] 里的 "20" 有 2 位，会被 _CITE_RE 命中吗？
    # 正则是 \[(\d{1,2})\]，"[2024]" 不匹配（4 位数字不满足 1~2 位整体匹配）
    _, invalid_year, _ = check_citations("发表于[2024]年。", sources(10))
    check("[2024] 不误判为编号", invalid_year == [], str(invalid_year))


def case_6_no_citation():
    print("== 6. 无引用（漏标）→ 一字不改 ==")
    ans = "高血压患者应低盐饮食，规律服药。"
    cleaned, invalid, cited = check_citations(ans, sources(3))
    check("cleaned == answer", cleaned == ans, cleaned)
    check("cited 为空", cited == [], str(cited))
    check("invalid 为空", invalid == [], str(invalid))


def case_7_punctuation_collapse():
    print("== 7. 剥除后标点收敛 ==")
    cleaned, invalid, _ = check_citations("按时服药即可[7]。", sources(3))
    check("invalid=[7]", invalid == [7], str(invalid))
    check("无多余空格", cleaned == "按时服药即可。", repr(cleaned))


def case_8_parse_unsupported_normal():
    print("== 8. parse_unsupported 正常 ==")
    check('"[2, 5]" → [2, 5]', parse_unsupported("[2, 5]") == [2, 5],
          str(parse_unsupported("[2, 5]")))
    check('"[]" → []', parse_unsupported("[]") == [], str(parse_unsupported("[]")))
    check('"[5, 2, 2]" 去重升序 → [2, 5]',
          parse_unsupported("[5, 2, 2]") == [2, 5], str(parse_unsupported("[5, 2, 2]")))


def case_9_parse_unsupported_abnormal():
    print("== 9. parse_unsupported 异常输入（fail-open）==")
    for raw in ("", "抱歉我无法判断", "```\n不是JSON\n```", '{"a": 1}', "[0, -1]", None):
        got = parse_unsupported(raw)
        check(f"{raw!r} → []", got == [], str(got))
    # 代码块包裹的**合法** JSON 仍应被解开（这是正常输出形态之一）
    check('"```json\\n[2]\\n```" → [2]',
          parse_unsupported("```json\n[2]\n```") == [2],
          str(parse_unsupported("```json\n[2]\n```")))


def case_10_bad_sources_payload():
    print("== 10. 非 dict / 缺 index 的 sources ==")
    # 全部缺 index → 退化为 1..n，不误判整篇回答
    cleaned, invalid, cited = check_citations("见资料[2]。", [{"title": "a"}, {"title": "b"}])
    check("缺 index 时退化 1..n，[2] 合法", invalid == [] and cited == [2],
          f"{invalid} {cited}")
    # 混入非 dict 元素：不得抛异常
    cleaned2, invalid2, _ = check_citations("见资料[1]。", ["坏载荷", {"title": "a"}])
    check("非 dict 元素不抛异常", invalid2 == [], str(invalid2))


# ============================================================================
# B. 图级 / 节点级（§3.2 用例 12~16）
# ============================================================================
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


HIT_DOCS = [Document(
    page_content="高血压患者可少量食用党参，但需遵医嘱，注意与降压药的相互作用。",
    metadata={"department": "内科", "title": "高血压", "source": "t"},
)]


def fake_rerank(query, docs, top_k=5):
    scored = [Document(page_content=d.page_content, metadata={
        **(d.metadata or {}), "rerank_score": 0.95}) for d in docs]
    return scored[:top_k]


def make_chain(text):
    return SimpleNamespace(invoke=lambda inputs: text)


async def run_full_graph(answer_text, question="高血压患者能吃党参吗", docs=None,
                         history=None):
    """跑完整图（意图分流 → 检索 → 生成 → 忠实性校验），返回终态。

    不带 checkpointer：clarify 闸关闭，无资料时直接走 insufficient，不进 interrupt。
    """
    docs = HIT_DOCS if docs is None else docs
    with patch.multiple(
        "app.graph",
        rewrite_for_retrieval=lambda q, h=None: RewriteResult(q, False, "", False,
                                                              "new_question", ""),
        route_knowledge_source=lambda q: FakeRouting(),
        get_retriever=lambda k=20: FakeRetriever(docs),
        rerank_documents=fake_rerank,
        format_history=lambda h: "（无历史对话）",
    ), patch("app.rag_chain.get_generation_chain", lambda: make_chain(answer_text)), \
       patch("app.rag_chain.get_chat_chain", lambda: make_chain("好的，还有其他想了解的吗？")), \
       patch("app.rag_chain.get_insufficient_chain",
             lambda: make_chain("现有医学资料无法回答该问题。")):
        graph = g.build_graph()
        return await graph.ainvoke({
            "question": question, "history": history or [], "trace": [],
        })


def case_12_graph_strips_invalid():
    print("== 12. 图级：回答含越界编号 → 被剥除 ==")
    out = asyncio.run(run_full_graph("高血压可少量食用党参[1]，但需遵医嘱[7]。"))
    grounding = out.get("grounding") or {}
    check("corrected_answer 非空", bool(grounding.get("corrected_answer")),
          str(grounding))
    check("终态 answer 不含 [7]", "[7]" not in (out.get("answer") or ""), out.get("answer"))
    check("终态 answer 保留 [1]", "[1]" in (out.get("answer") or ""), out.get("answer"))
    check("verdict=citation_fixed", grounding.get("verdict") == "citation_fixed",
          str(grounding.get("verdict")))
    check("invalid_citations=[7]", grounding.get("invalid_citations") == [7],
          str(grounding.get("invalid_citations")))


def case_13_graph_normal_untouched():
    print("== 13. 图级：回答正常 → 不推 correction ==")
    ans = "高血压可少量食用党参[1]，但需遵医嘱。"
    out = asyncio.run(run_full_graph(ans))
    grounding = out.get("grounding") or {}
    check("corrected_answer is None", grounding.get("corrected_answer") is None,
          str(grounding.get("corrected_answer")))
    check("verdict=pass", grounding.get("verdict") == "pass", str(grounding.get("verdict")))
    check("answer 一字不改", out.get("answer") == ans, out.get("answer"))


def case_14_l2_appends_hedge():
    print("== 14. L2 开启 + 返回 [2] → 追加 GROUNDING_HEDGE ==")
    with patch.object(settings, "groundedness_citation_check", True), \
         patch.object(settings, "groundedness_llm_enabled", True), \
         patch("app.rag_chain.get_groundedness_chain", lambda: make_chain("[2]")):
        out = g.node_ground_check({
            "answer": "第一句有依据[1]。第二句是推断。",
            "sources": sources(3),
            "context": "资料正文",
        })
    grounding = out.get("grounding") or {}
    check("unsupported==[2]", grounding.get("unsupported_sentences") == [2],
          str(grounding.get("unsupported_sentences")))
    check("verdict=unsupported", grounding.get("verdict") == "unsupported",
          str(grounding.get("verdict")))
    check("corrected_answer 以 GROUNDING_HEDGE 结尾",
          (grounding.get("corrected_answer") or "").endswith(GROUNDING_HEDGE),
          repr(grounding.get("corrected_answer")))


def case_15_l2_fail_open():
    print("== 15. L2 开启 + 调用抛异常 → fail-open ==")
    def boom():
        raise RuntimeError("模拟 L2 服务不可用")

    with patch.object(settings, "groundedness_citation_check", True), \
         patch.object(settings, "groundedness_llm_enabled", True), \
         patch("app.rag_chain.get_groundedness_chain", boom):
        out = g.node_ground_check({
            "answer": "高血压可少量食用党参[1]。",
            "sources": sources(3),
            "context": "资料正文",
        })
    grounding = out.get("grounding") or {}
    check("流程未中断", isinstance(out, dict) and "grounding" in out)
    check("corrected_answer is None（不误报）",
          grounding.get("corrected_answer") is None,
          str(grounding.get("corrected_answer")))
    check("unsupported 为空", grounding.get("unsupported_sentences") == [],
          str(grounding.get("unsupported_sentences")))


def case_11_switch_off():
    print("== 11. 开关关掉（groundedness_citation_check=False）→ 文本一字不改 ==")
    ans = "高血压可少量食用党参[1]，但需遵医嘱[7]。"
    with patch.object(settings, "groundedness_citation_check", False):
        out = g.node_ground_check({"answer": ans, "sources": sources(3), "context": ""})
    grounding = out.get("grounding") or {}
    check("answer 一字不改", out.get("answer") == ans, out.get("answer"))
    check("invalid 为空", grounding.get("invalid_citations") == [],
          str(grounding.get("invalid_citations")))
    check("corrected_answer is None", grounding.get("corrected_answer") is None)


def case_16_trace_placement():
    print("== 16. trace：「忠实性校验」恰好 1 条；chat / insufficient 没有 ==")
    # 医学路径（有资料 → generate → ground_check）
    med = asyncio.run(run_full_graph("高血压可少量食用党参[1]。"))
    steps = [t["step"] for t in (med.get("trace") or [])]
    n_ground = steps.count("忠实性校验")
    check("医学路径含「忠实性校验」恰好 1 次", n_ground == 1, str(steps))

    # 会话语路径（规则门直连 chat，不经 generate/ground_check）
    chat = asyncio.run(run_full_graph("（不该被调用）", question="好的"))
    chat_steps = [t["step"] for t in (chat.get("trace") or [])]
    check("chat 路径没有「忠实性校验」",
          "忠实性校验" not in chat_steps, str(chat_steps))
    check("chat 路径走了「对话式回答」", "对话式回答" in chat_steps, str(chat_steps))

    # 兜底路径（无资料 → insufficient，不接校验节点）
    ins = asyncio.run(run_full_graph("（不该被调用）", docs=[]))
    ins_steps = [t["step"] for t in (ins.get("trace") or [])]
    check("insufficient 路径没有「忠实性校验」",
          "忠实性校验" not in ins_steps, str(ins_steps))


# ============================================================================
# C. 生成侧自省后处理（T59 改道：partial 三档的替代方案，用例 17~21）
# ============================================================================
def case_17_detect_unanswerable_hits():
    print("== 17. detect_unanswerable：开头自陈 → 命中 ==")
    for ans in (
        "现有医学资料无法回答该问题，建议咨询执业医师。",
        "您好。资料中未提供该药的具体剂量。",
        "未找到相关资料，无法给出确切回答。",
        "抱歉，资料未提及这个病的预后情况。",
    ):
        hit = detect_unanswerable(ans)
        check(f"命中：{ans[:16]}…", bool(hit), repr(ans))


def case_18_detect_unanswerable_position():
    print("== 18. detect_unanswerable：中后段提及 → 不命中（位置约束）==")
    # 主体答得上来，只是某一点资料没有 —— 剥掉整篇引用会丢掉真实出处
    ans = (
        "高血压患者可以少量食用党参，但需遵医嘱并监测血压[1]。"
        "日常应低盐饮食、规律服药、控制体重[2]。"
        "至于该药与党参的相互作用，资料未提供明确结论[3]。"
    )
    check("中段「资料未提供」不命中", detect_unanswerable(ans) == "",
          repr(detect_unanswerable(ans)))
    check("空串不命中", detect_unanswerable("") == "")
    check("None 不命中", detect_unanswerable(None) == "")


def case_19_strip_all_citations():
    print("== 19. strip_all_citations：剥全部编号 + 标点收敛 ==")
    cleaned, stripped = strip_all_citations("资料未提供该结论[1]，也无法给出确切答复[2]。")
    check("剥出 [1,2]", stripped == [1, 2], str(stripped))
    check("正文无编号", "[1]" not in cleaned and "[2]" not in cleaned, cleaned)
    check("标点收敛无多余空格",
          cleaned == "资料未提供该结论，也无法给出确切答复。", repr(cleaned))
    same, none_stripped = strip_all_citations("普通句子，不带编号。")
    check("无编号时文本不变", same == "普通句子，不带编号。", repr(same))
    check("无编号时返回 []", none_stripped == [], str(none_stripped))


def case_20_graph_strips_self_aware():
    print("== 20. 图级：自陈答不上来 + 引用编号 → 整篇剥除 ==")
    # 1 条资料 → [1] 是**合法**编号，L1 不动它；由自省层整篇剥掉
    ans = "现有医学资料无法回答该问题[1]，建议您咨询执业医师。"
    out = asyncio.run(run_full_graph(ans))
    grounding = out.get("grounding") or {}
    final = out.get("answer") or ""
    check("终态不含 [1]", "[1]" not in final, final)
    check("verdict=unanswerable_stripped",
          grounding.get("verdict") == "unanswerable_stripped",
          str(grounding.get("verdict")))
    check("stripped_citations==[1]", grounding.get("stripped_citations") == [1],
          str(grounding.get("stripped_citations")))
    check("invalid 为空（[1] 本身合法）", grounding.get("invalid_citations") == [],
          str(grounding.get("invalid_citations")))
    check("corrected_answer 非空（据此推 correction）",
          bool(grounding.get("corrected_answer")))
    check("回答正文保留（只剥编号，不删内容）", "建议您咨询执业医师" in final, final)


def case_21_self_aware_switch_off():
    print("== 21. 开关关掉 → 自省后处理也不生效 ==")
    ans = "现有医学资料无法回答该问题[1]。"
    with patch.object(settings, "groundedness_citation_check", False):
        out = g.node_ground_check({"answer": ans, "sources": sources(3), "context": ""})
    check("answer 一字不改", out.get("answer") == ans, out.get("answer"))
    check("stripped 为空",
          (out.get("grounding") or {}).get("stripped_citations") == [],
          str((out.get("grounding") or {}).get("stripped_citations")))


def main():
    case_1_normal()
    case_2_out_of_range()
    case_3_all_out_of_range()
    case_4_duplicate()
    case_5_two_digit()
    case_6_no_citation()
    case_7_punctuation_collapse()
    case_8_parse_unsupported_normal()
    case_9_parse_unsupported_abnormal()
    case_10_bad_sources_payload()
    case_11_switch_off()
    case_12_graph_strips_invalid()
    case_13_graph_normal_untouched()
    case_14_l2_appends_hedge()
    case_15_l2_fail_open()
    case_16_trace_placement()
    case_17_detect_unanswerable_hits()
    case_18_detect_unanswerable_position()
    case_19_strip_all_citations()
    case_20_graph_strips_self_aware()
    case_21_self_aware_switch_off()

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项: {_FAILED}")
        return 1
    print("全绿：忠实性校验 L1/L2 + 生成侧自省 + 接线与 fail-open 均通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())