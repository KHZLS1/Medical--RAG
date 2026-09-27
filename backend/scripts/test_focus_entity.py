"""阶段三 focus_entity（指代消解）的离线自测

零外部依赖：不需要 Milvus / LLM / MySQL。对应 `实施计划_阶段三_指代消解.md` §3：

  A. 解析层（§3.1，用例 1~11）—— 纯函数，零 LLM。含两个"一行未改"的回归套件：
       `_sanitize_rewrite` 11 例、`_parse_rewrite_output` 11 例 —— 它们是阶段三
       "只加不改"的证据，任一变红说明旧路径被动过。
  B. 图级兜底（§3.2，用例 12~17）—— mock 掉改写/检索/精排，验 D4 的确定性：
       只有 followup + 改写退化才用焦点补全，且 ack 不得污染跨轮焦点。

跑法（backend 目录）：
  python scripts/test_focus_entity.py
退出码 0 = 全绿。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

from langchain_core.documents import Document

from app import graph as g
from app.query_rewriter import (
    RewriteResult,
    _FOCUS_MAX_CHARS,
    _clean_focus,
    _parse_rewrite_output,
    _sanitize_rewrite,
)
from app.dialogue_policy import should_update_focus
from app.rewrite_cache import RewriteCache

_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return bool(cond)


# ============================================================================
# A. 解析层（§3.1 用例 1~11）
# ============================================================================
def case_1_three_lines():
    print("== 1. 三行输出正常解析 ==")
    r = _parse_rewrite_output(
        "行为：followup\n焦点：高血压 党参\n查询：党参 副作用 禁忌",
        "它有什么副作用",
    )
    check("act=followup", r.dialogue_act == "followup", r.dialogue_act)
    check("focus 解析", r.focus_entity == "高血压 党参", r.focus_entity)
    check("query 解析", r.query == "党参 副作用 禁忌", r.query)
    check("未退化", not r.degraded, r.reason)


def case_2_bold_labels():
    print("== 2. 加粗标签 **焦点**：高血压 ==")
    r = _parse_rewrite_output(
        "**行为**：new_question\n**焦点**：高血压\n**查询**：高血压 党参 禁忌",
        "高血压患者能吃党参吗",
    )
    check("焦点仍被识别", r.focus_entity == "高血压", r.focus_entity)
    check("焦点不进 query", r.query == "高血压 党参 禁忌", r.query)
    check("query 里不含「焦点」二字", "焦点" not in r.query, r.query)


def case_3_missing_focus_line():
    print("== 3. 缺焦点行（旧格式两行）==")
    r = _parse_rewrite_output(
        "行为：new_question\n查询：高血压 党参",
        "高血压患者能吃党参吗",
    )
    check("focus_entity 为空", r.focus_entity == "", r.focus_entity)
    check("act 正常", r.dialogue_act == "new_question", r.dialogue_act)
    check("query 正常", r.query == "高血压 党参", r.query)
    # 向后兼容：完全没有标签行时，query 与加焦点之前的路径**逐字相同**
    old = _sanitize_rewrite("高血压 党参", "高血压患者能吃党参吗")
    r2 = _parse_rewrite_output("高血压 党参", "高血压患者能吃党参吗")
    check("旧格式 query 与 _sanitize_rewrite 逐字相同",
          r2.query == old.query and r2.degraded == old.degraded,
          f"{r2.query!r} vs {old.query!r}")


def case_4_focus_empty_values():
    print("== 4. 焦点空值 ==")
    for v in ("无", "N/A", "（无）", "none", "-", "  "):
        check(f"「{v}」→ 空", _clean_focus(v) == "", repr(_clean_focus(v)))
    # 精确匹配：不得误伤「无痛」这种以「无」开头的实体
    check("「无痛」保留（精确匹配）", _clean_focus("无痛") == "无痛", _clean_focus("无痛"))


def case_5_focus_too_long():
    print("== 5. 焦点超长 ==")
    long = "高" * (_FOCUS_MAX_CHARS + 1)
    check(f"{_FOCUS_MAX_CHARS + 1} 字 → 空（宁缺勿滥）", _clean_focus(long) == "")
    ok = "高" * _FOCUS_MAX_CHARS
    check(f"{_FOCUS_MAX_CHARS} 字 → 保留", _clean_focus(ok) == ok)


def case_6_focus_without_colon():
    print("== 6. 焦点行未加冒号（焦点 高血压）==")
    r = _parse_rewrite_output("焦点 高血压", "它有什么副作用")
    check("不被识别为焦点", r.focus_entity == "", r.focus_entity)
    # 不污染 query：走既有 _sanitize_rewrite 路径，结果与旧逻辑一致
    old = _sanitize_rewrite("焦点 高血压", "它有什么副作用")
    check("query 与既有路径一致", r.query == old.query, f"{r.query!r} vs {old.query!r}")


def case_7_degraded_with_focus():
    print("== 7. 改写退化 + 焦点有效（解耦性）==")
    r = _parse_rewrite_output(
        "行为：followup\n焦点：高血压 党参\n查询：无法判断用户意图",
        "它有什么副作用",
    )
    check("degraded=True", r.degraded)
    check("query 回退原问题", r.query == "它有什么副作用", r.query)
    check("焦点照样提取（与退化解耦）", r.focus_entity == "高血压 党参", r.focus_entity)


def case_8_should_update_focus():
    print("== 8. should_update_focus ==")
    for act in ("new_question", "followup"):
        check(f"{act} → True", should_update_focus(act) is True)
    for act in ("ack", "chitchat", None, "", "unknown", "xyz"):
        check(f"{act!r} → False", should_update_focus(act) is False)


def case_9_cache_roundtrip():
    print("== 9. 缓存往返（focus 载荷）==")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cache.json"
        c = RewriteCache(path, fingerprint="fp")
        c.put("context", "hist", "q", "党参 副作用", False, "", "followup", "高血压 党参")
        hit = c.get("context", "hist", "q")
        check("focus 往返", hit and hit.get("focus") == "高血压 党参",
              str(hit.get("focus") if hit else None))
        check("act 往返", hit and hit.get("act") == "followup")

        # 旧条目（阶段三之前的缓存）没有 focus 键 → 读取方回退空串，不报错
        key = RewriteCache.make_key("single", "", "旧问题")
        path.write_text(json.dumps({
            "meta": {"fingerprint": "fp"},
            "entries": {key: {"query": "旧查询", "degraded": False,
                              "reason": "", "act": "new_question"}},
        }, ensure_ascii=False), encoding="utf-8")
        c2 = RewriteCache(path, fingerprint="fp")
        old = c2.get("single", "", "旧问题")
        check("旧条目 focus 回退空串", old is not None and old.get("focus", "") == "",
              str(old))


# _sanitize_rewrite 11 例回归（阶段三不改它 → 必须逐字不变）
_SANITIZE_CASES: list[tuple[str, str, str, bool]] = [
    # (raw, original, expected_query, expected_degraded)
    ("高血压 党参", "原问题", "高血压 党参", False),
    ("", "原问题", "原问题", True),
    ("   ", "原问题", "原问题", True),
    ("```\n高血压 党参\n```", "原问题", "高血压 党参", False),
    ("改写后的查询：\n高血压 党参", "原问题", "高血压 党参", False),
    ("高血压 党参\n这是一句解释", "原问题", "高血压 党参", False),
    ("「高血压 党参」", "原问题", "高血压 党参", False),
    ("长" * 121, "原问题", "原问题", True),
    ("无法判断用户意图", "原问题", "原问题", True),
    ("过于简短，无法改写", "原问题", "原问题", True),
    ("请提供更多信息", "原问题", "原问题", True),
]


def case_10_sanitize_regression():
    print("== 10. _sanitize_rewrite 回归（11 例）==")
    for i, (raw, original, exp_q, exp_deg) in enumerate(_SANITIZE_CASES, start=1):
        r = _sanitize_rewrite(raw, original)
        check(f"#{i} query", r.query == exp_q, f"{r.query!r} != {exp_q!r}")
        check(f"#{i} degraded", r.degraded == exp_deg, f"{r.degraded} != {exp_deg}")
        if exp_deg:
            check(f"#{i} 退化带 reason", bool(r.reason), r.reason)


# _parse_rewrite_output 11 例回归（阶段一的既有行为）
_PARSE_CASES: list[tuple[str, str, str, str]] = [
    # (raw, original, expected_act, expected_query)
    ("行为：followup\n查询：党参 副作用", "它有什么副作用", "followup", "党参 副作用"),
    ("高血压 党参", "原问题", "new_question", "高血压 党参"),
    ("行为：xyz\n查询：高血压", "原问题", "new_question", "高血压"),
    ("查询：高血压\n行为：followup", "原问题", "followup", "高血压"),
    ("行为异常 儿童", "原问题", "new_question", "行为异常 儿童"),
    ("**行为**：ack\n查询：好的", "好的", "ack", "好的"),
    ("行为：指代追问\n查询：党参 副作用", "它有什么副作用", "followup", "党参 副作用"),
    ("行为：ack\n行为：followup\n查询：好的", "好的", "ack", "好的"),
    ("问题：高血压 党参", "原问题", "new_question", "高血压 党参"),
    ("**查询**：高血压 党参", "原问题", "new_question", "高血压 党参"),
    ("对话行为：chitchat\n查询：你好", "你好", "chitchat", "你好"),
]


def case_11_parse_regression():
    print("== 11. _parse_rewrite_output 回归（11 例）==")
    for i, (raw, original, exp_act, exp_q) in enumerate(_PARSE_CASES, start=1):
        r = _parse_rewrite_output(raw, original)
        check(f"#{i} act", r.dialogue_act == exp_act, f"{r.dialogue_act!r} != {exp_act!r}")
        check(f"#{i} query", r.query == exp_q, f"{r.query!r} != {exp_q!r}")


# ============================================================================
# B. 图级兜底（§3.2 用例 12~17）
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
    page_content="高血压患者用药需注意与党参等中药的相互作用…",
    metadata={"department": "内科", "title": "高血压", "source": "t"},
)]


def fake_rerank(query, docs, top_k=5):
    scored = [Document(page_content=d.page_content, metadata={
        **(d.metadata or {}), "rerank_score": 0.95}) for d in docs]
    return scored[:top_k]


def make_rewrite(act, query, degraded=False, focus="", reason=""):
    """伪造一次改写调用的产物（含阶段三的 focus_entity）。"""
    def _fake(question, history=None):
        return RewriteResult(query, degraded, reason, False, act, focus)
    return _fake


def run_subgraph(rewrite_fake, state):
    """跑检索子图（改写/路由/检索/精排全 mock），返回终态。"""
    with patch.multiple(
        "app.graph",
        rewrite_for_retrieval=rewrite_fake,
        route_knowledge_source=lambda q: FakeRouting(),
        get_retriever=lambda k=20: FakeRetriever(HIT_DOCS),
        rerank_documents=fake_rerank,
    ):
        return g.build_retrieval_graph().invoke(state)


def case_12_followup_degraded_with_focus():
    print("== 12. followup + 退化 + 显式焦点 → 焦点补全 ==")
    out = run_subgraph(
        make_rewrite("followup", "它有什么副作用", True, "高血压 党参", "元话语「无法判断」"),
        {"question": "它有什么副作用",
         "history": [{"role": "user", "content": "高血压患者能吃党参吗"}],
         "trace": []},
    )
    check("enhanced_query = 焦点 + 原问题",
          out.get("enhanced_query") == "高血压 党参 它有什么副作用",
          out.get("enhanced_query"))
    check("focus_entity 落进状态", out.get("focus_entity") == "高血压 党参",
          out.get("focus_entity"))


def case_13_followup_degraded_history_fallback():
    print("== 13. followup + 退化 + 无显式焦点 → 历史派生兜底 ==")
    out = run_subgraph(
        make_rewrite("followup", "它有什么副作用", True, "", "元话语「无法判断」"),
        {"question": "它有什么副作用",
         "history": [{"role": "user", "content": "高血压患者能吃党参吗"}],
         "trace": []},
    )
    check("enhanced_query 用历史派生焦点拼出",
          out.get("enhanced_query") == "高血压患者能吃党参吗 它有什么副作用",
          out.get("enhanced_query"))


def case_14_followup_degraded_history_ack_only():
    print("== 14. followup + 退化 + 历史只有「好的」→ 不兜底（防回归）==")
    out = run_subgraph(
        make_rewrite("followup", "它有什么副作用", True, "", "元话语「无法判断」"),
        {"question": "它有什么副作用",
         "history": [{"role": "user", "content": "好的"}],
         "trace": []},
    )
    check("enhanced_query = 原问题（不拼会话语）",
          out.get("enhanced_query") == "它有什么副作用",
          out.get("enhanced_query"))


def case_15_followup_not_degraded():
    print("== 15. followup + 未退化 → 用改写产物，不用焦点 ==")
    out = run_subgraph(
        make_rewrite("followup", "党参 副作用 禁忌", False, "高血压 党参"),
        {"question": "它有什么副作用",
         "history": [{"role": "user", "content": "高血压患者能吃党参吗"}],
         "trace": []},
    )
    check("enhanced_query = 改写产物",
          out.get("enhanced_query") == "党参 副作用 禁忌",
          out.get("enhanced_query"))


def case_16_new_question_degraded():
    print("== 16. new_question + 退化 → 原句检索，不吃焦点 ==")
    out = run_subgraph(
        make_rewrite("new_question", "高血压患者能吃党参吗", True, "高血压 党参", "元话语"),
        {"question": "高血压患者能吃党参吗",
         "history": [{"role": "user", "content": "孩子发烧怎么办"}],
         "trace": []},
    )
    check("enhanced_query = 原句",
          out.get("enhanced_query") == "高血压患者能吃党参吗",
          out.get("enhanced_query"))


def case_17_ack_keeps_focus():
    print("== 17. ack → 掉头、不检索、不覆盖焦点 ==")
    out = run_subgraph(
        make_rewrite("ack", "好的", False, ""),
        {"question": "好的",
         "history": [{"role": "user", "content": "高血压患者能吃党参吗"}],
         "trace": [],
         "focus_entity": "高血压 党参"},
    )
    check("act=ack", out.get("dialogue_act") == "ack", out.get("dialogue_act"))
    check("未产出 enhanced_query（没检索）", not out.get("enhanced_query"),
          out.get("enhanced_query"))
    check("未产出 docs（没检索）", not out.get("docs"))
    check("焦点保持原值（不被污染）", out.get("focus_entity") == "高血压 党参",
          out.get("focus_entity"))


def main():
    case_1_three_lines()
    case_2_bold_labels()
    case_3_missing_focus_line()
    case_4_focus_empty_values()
    case_5_focus_too_long()
    case_6_focus_without_colon()
    case_7_degraded_with_focus()
    case_8_should_update_focus()
    case_9_cache_roundtrip()
    case_10_sanitize_regression()
    case_11_parse_regression()
    case_12_followup_degraded_with_focus()
    case_13_followup_degraded_history_fallback()
    case_14_followup_degraded_history_ack_only()
    case_15_followup_not_degraded()
    case_16_new_question_degraded()
    case_17_ack_keeps_focus()

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项: {_FAILED}")
        return 1
    print("全绿：focus_entity 解析层 + 图级兜底均通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())