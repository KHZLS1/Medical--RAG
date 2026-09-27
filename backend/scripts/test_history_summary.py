"""长对话历史摘要层（T61）的离线自测。

`format_history` 的返回值是**改写缓存的键**，所以这里有两件事必须钉死：
  1. 未发生截断时输出**逐字不变**（否则跨运行的可复现性会断，指标不可比）；
  2. 发生截断时旧消息被压成一行要点，且要点本身有界（条数 + 长度）。

零外部依赖（不需要 Milvus / LLM / MySQL）：
  python scripts/test_history_summary.py
退出码 0 = 全绿。
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.query_rewriter import (
    _OLDER_SUMMARY_ITEM_CHARS,
    _OLDER_SUMMARY_MAX_ITEMS,
    _format_recent,
    _summarize_older,
    format_history,
)

_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return bool(cond)


def _turn(q, a="好的。"):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def case_1_short_history_unchanged():
    print("== 1. 未截断 → 与旧实现逐字一致（缓存键不能变）==")
    hist = _turn("高血压患者能吃党参吗") + _turn("它有什么副作用")
    # 旧实现的等价形态：逐条 "角色: 内容"
    expected = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in hist
    )
    got = format_history(hist)
    check("4 条 ≤ 6，输出与旧实现逐字相同", got == expected, repr(got))
    check("恰好 6 条也不变", format_history(hist + _turn("那还要注意什么")) ==
          "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                    for m in hist + _turn("那还要注意什么")))
    check("空历史 → 「（无历史对话）」", format_history([]) == "（无历史对话）",
          repr(format_history([])))


def case_2_truncation_adds_summary():
    print("== 2. 超窗口 → 补一行要点，且最近 6 条原样保留 ==")
    hist = []
    for i in range(1, 8):                       # 7 轮 = 14 条
        hist += _turn(f"第{i}轮用户问题")
    got = format_history(hist)
    check("含「更早的对话要点」标记", "（更早的对话要点）" in got, got[:80])
    check("含最早期的问题（第1轮）", "第1轮用户问题" in got, got[:160])
    check("最近 6 条原样在尾部", got.rstrip().endswith(
        "助手: 好的。"), repr(got[-40:]))
    check("窗口内条数正确（6 条）",
          sum(1 for ln in got.split("\n") if ln.startswith(("用户:", "助手:"))) == 6,
          str(sum(1 for ln in got.split("\n") if ln.startswith(("用户:", "助手:")))))


def case_3_summary_is_users_only_and_bounded():
    print("== 3. 要点只取用户发言，且条数 / 长度有界 ==")
    hist = []
    for i in range(1, 10):                      # 9 轮，窗口外有 12 条
        hist += _turn(f"用户第{i}个问题" + "补" * 60, a="助手的长回答" * 30)
    summary = _summarize_older(hist[:-6])
    parts = summary.split("；")
    check(f"条数 ≤ {_OLDER_SUMMARY_MAX_ITEMS}", len(parts) <= _OLDER_SUMMARY_MAX_ITEMS,
          str(len(parts)))
    check("每条长度有界",
          all(len(p) <= _OLDER_SUMMARY_ITEM_CHARS for p in parts),
          str([len(p) for p in parts]))
    check("不含助手内容", "助手的长回答" not in summary, summary[:80])
    # hist[:-6] 是第 1~6 轮；从近往前取 4 条 ⇒ 留下第 3、4、5、6 轮
    check("保留的是较近的（含第6轮，不含第1轮）",
          "用户第6个问题" in summary and "用户第1个问题" not in summary,
          summary[:120])
    check("时间正序（第3轮在第6轮之前）",
          summary.index("用户第3个问题") < summary.index("用户第6个问题"), summary[:120])


def case_4_dedupe_and_empty():
    print("== 4. 重复提问去重 / 无用户发言时不产出要点 ==")
    hist = _turn("高血压能吃党参吗") * 4        # 同一句重复 4 次
    check("去重后只剩 1 条", _summarize_older(hist) == "高血压能吃党参吗",
          repr(_summarize_older(hist)))
    assistant_only = [{"role": "assistant", "content": "只有助手在说话"}]
    check("只有助手发言 → 空要点", _summarize_older(assistant_only) == "",
          repr(_summarize_older(assistant_only)))
    # 要点为空时不应多出一行空标题
    only_assistant_long = [
        {"role": "assistant", "content": f"回答{i}"} for i in range(9)
    ] + _turn("最后一句是用户说的")
    out = format_history(only_assistant_long)
    check("要点为空时不加标题行", "（更早的对话要点）" not in out, out[:80])


def case_5_switch_off():
    print("== 5. 开关关掉 → 退回「只取最近 N 条」==")
    hist = []
    for i in range(1, 8):
        hist += _turn(f"第{i}轮用户问题")
    with patch.object(settings, "history_summary_enabled", False):
        got = format_history(hist)
    check("无要点标记", "更早的对话要点" not in got, got[:80])
    check("等于旧实现（最近 6 条）", got == _format_recent(hist[-6:]), repr(got[:80]))


def case_6_custom_window():
    print("== 6. max_turns 可调，边界不差一 ==")
    hist = _turn("问题1") + _turn("问题2") + _turn("问题3")   # 6 条
    check("max_turns=6 不截断", "要点" not in format_history(hist, max_turns=6))
    out = format_history(hist, max_turns=4)
    check("max_turns=4 截断（窗口外 2 条进要点）", "（更早的对话要点）" in out, out[:120])
    check("窗口内恰好 4 条",
          sum(1 for ln in out.split("\n") if ln.startswith(("用户:", "助手:"))) == 4,
          out)


def main():
    case_1_short_history_unchanged()
    case_2_truncation_adds_summary()
    case_3_summary_is_users_only_and_bounded()
    case_4_dedupe_and_empty()
    case_5_switch_off()
    case_6_custom_window()

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项: {_FAILED}")
        return 1
    print("全绿：历史摘要只在截断时生效，未截断输出逐字不变。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
