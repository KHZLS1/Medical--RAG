"""改写缓存（app/rewrite_cache.py）的离线自测，重点覆盖**冻结模式**。

冻结模式是「评估可复现」这条线上的最后一环：它让 A/B 两边吃同一份改写结果
（跳过指纹校验 + 只读）。一旦它失效，程序**不会报错**，只会静默给出不可比的
数字 —— 这类"沉默的错"必须用硬断言守住。

零外部依赖（不需要 Milvus / LLM / MySQL），秒级：
  python scripts/test_rewrite_cache.py
退出码 0 = 全绿。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rewrite_cache import RewriteCache

_Q = "高血压患者能吃党参吗"
_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return bool(cond)


def _write_cache(path: Path, fingerprint: str, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": {"fingerprint": fingerprint, "count": len(entries)},
                    "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def _entry(query="高血压 党参 禁忌", act="new_question", focus="高血压 党参") -> dict:
    return {"q": _Q, "query": query, "degraded": False, "reason": "",
            "act": act, "focus": focus}


def case_1_fingerprint_mismatch():
    print("== 1. 指纹不符 → 整表作废（既有行为，别被冻结模式带偏）==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        _write_cache(p, "OLD_FP", {RewriteCache.make_key("single", "", _Q): _entry()})
        c = RewriteCache(p, "NEW_FP")
        check("未命中", c.get("single", "", _Q) is None)
        check("stats.frozen=False", c.stats["frozen"] is False, str(c.stats))


def case_2_fingerprint_match():
    print("== 2. 指纹相符 → 正常命中 ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        _write_cache(p, "FP", {RewriteCache.make_key("single", "", _Q): _entry()})
        c = RewriteCache(p, "FP")
        hit = c.get("single", "", _Q)
        check("命中", hit is not None)
        check("query 取回", (hit or {}).get("query") == "高血压 党参 禁忌", str(hit))
        check("stats.hits=1", c.stats["hits"] == 1, str(c.stats))


def case_3_frozen_bypasses_fingerprint():
    print("== 3. 冻结模式：指纹不符也照样命中（绕过指纹是它的全部意义）==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "frozen.json"
        _write_cache(p, "FP_AT_FREEZE_TIME",
                     {RewriteCache.make_key("context", "hist", _Q): _entry()})
        c = RewriteCache(p, "FP_NOW_AFTER_PROMPT_CHANGE", frozen=True)
        hit = c.get("context", "hist", _Q)
        check("命中", hit is not None)
        check("act/focus 一起取回",
              (hit or {}).get("act") == "new_question" and (hit or {}).get("focus") == "高血压 党参",
              str(hit))
        check("stats.frozen=True", c.stats["frozen"] is True, str(c.stats))


def case_4_frozen_is_read_only():
    print("== 4. 冻结模式只读：put 绝不落盘（否则冻结集会被悄悄改写）==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "frozen.json"
        _write_cache(p, "FP", {RewriteCache.make_key("single", "", _Q): _entry()})
        before = p.read_text(encoding="utf-8")
        c = RewriteCache(p, "FP", frozen=True)
        c.put("context", "hist", "那还要做哪些检查", "小儿发热 检查",
              False, "", "followup", "小儿发热")
        after = p.read_text(encoding="utf-8")
        check("文件内容未变", before == after)
        check("内存里仍可命中（进程内自洽）",
              c.get("context", "hist", "那还要做哪些检查") is not None)


def case_5_frozen_missing_file():
    print("== 5. 冻结集不存在 → 不崩、也不创建（提示在日志里）==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nope.json"
        c = RewriteCache(p, "FP", frozen=True)
        check("构造不抛异常", True)
        c.put("single", "", "随便一个问题", "随便")
        check("put 之后文件仍不存在（只读）", not p.exists())


def case_6_max_entries_eviction():
    print("== 6. max_entries：超出按插入顺序淘汰最旧的 ==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        c = RewriteCache(p, "FP", max_entries=2)
        for i in (1, 2, 3):
            c.put("single", "", f"问题{i}", f"查询{i}")
        check("size 收敛到 2", c.stats["size"] == 2, str(c.stats))
        check("最旧的被淘汰", c.get("single", "", "问题1") is None)
        check("最新的还在", c.get("single", "", "问题3") is not None)


def case_7_empty_meta_frozen():
    print("== 7. 冻结集没有 meta 字段 → 不崩（半损坏文件也要能读）==")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"entries": {RewriteCache.make_key("single", "", _Q): _entry()}}),
                     encoding="utf-8")
        c = RewriteCache(p, "FP", frozen=True)
        check("命中（meta 缺失不影响）", c.get("single", "", _Q) is not None)


def main():
    case_1_fingerprint_mismatch()
    case_2_fingerprint_match()
    case_3_frozen_bypasses_fingerprint()
    case_4_frozen_is_read_only()
    case_5_frozen_missing_file()
    case_6_max_entries_eviction()
    case_7_empty_meta_frozen()

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项: {_FAILED}")
        return 1
    print("全绿：改写缓存指纹作废 / 冻结绕过与只读 / 淘汰策略 均通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
