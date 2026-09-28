"""离线自测：跨会话长期记忆（T62-⑦，LangGraph SqliteStore）

验证四件事：
  1. store 读写 / key 幂等去重（同事实写两次只留一条）
  2. 抽取层吃畸形输入（解析失败一律当"这轮没记的"）
  3. 图级接入（node_recall / node_remember / 注入【用户情况】）与开关关闭时的零行为
  4. 跨轮重置（long_term_memory / memory_written 必须按轮清零）

全程不连 Milvus / MySQL / LLM（LLM 用假对象）。sqlite 落在临时目录，跑完即删。

跑法：python scripts/test_long_term_memory.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.graph as g
import app.long_term_memory as ltm
from app.long_term_memory import (
    extract_memories, learn_from_turn, recall, recall_items,
    recall_text, remember,
)

_fails: list[str] = []


def check(name: str, cond, detail: str = "") -> None:
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        _fails.append(name)


_TMP = tempfile.mkdtemp(prefix="ltm_test_")
_DB = os.path.join(_TMP, "mem.sqlite")


def _reset_store() -> None:
    """重置单例并换一个新的 sqlite 文件（每个用例从干净库起步）。"""
    if ltm._conn is not None:
        try:
            ltm._conn.close()
        except Exception:
            pass
    ltm._store = None
    ltm._conn = None


class _FakeLLM:
    def __init__(self, content=None, boom=False):
        self._content, self._boom = content, boom

    def invoke(self, _prompt):
        if self._boom:
            raise RuntimeError("provider 挂了")
        return type("R", (), {"content": self._content})()


_ORIG_GET_LLM = ltm.get_llm

with patch.object(ltm.settings, "long_term_memory_db_path", _DB):
    # ------------------------------------------------------------ 1. 读写
    print("== 1. 读写与幂等 ==")
    _reset_store()
    check("空库返回空", recall_items() == [])
    check("空库 recall_text 为空串", recall_text() == "")
    remember("allergy", "对青霉素过敏")
    remember("history", "高血压 10 年")
    items = recall_items()
    check("写入两条", len(items) == 2, str(items))
    check("按 kind 排序（history 在 allergy 之后，因定义顺序）",
          [i["kind"] for i in items] == ["history", "allergy"], str([i["kind"] for i in items]))
    remember("allergy", "对青霉素过敏")          # 同 kind 同文本
    check("同事实写两次只留一条（key 幂等）", len(recall_items()) == 2, str(recall_items()))
    remember("history", "高血压十年")            # 文本不同 ⇒ 新条目
    check("文本不同算新条目", len(recall_items()) == 3)
    check("非法 kind 被拒", remember("不存在的类型", "x") is None)
    check("空文本被拒", remember("history", "   ") is None)
    check("文本截断到 120 字", len(remember("history", "糖" * 500) or "") == 120)

    items, text = recall()
    check("recall 返回 (条目, 文本)", len(items) == 4 and "【用户长期病史（跨会话记忆）】" in text)
    check("文本里带中文标签", "过敏史：对青霉素过敏" in text, text)

    print("== 2. 持久化：换连接后仍在（这是与 InMemoryStore 的关键差别）==")
    _reset_store()                                # 丢掉单例，相当于进程重启
    check("重开后记忆还在", len(recall_items()) == 4, str(recall_items()))

    # ------------------------------------------------------------ 3. 抽取层
    print("== 3. 抽取层：吃畸形输入，绝不抛异常 ==")
    ltm.get_llm = lambda *a, **k: _FakeLLM(
        '[{"kind": "allergy", "text": "对阿司匹林过敏"}]')
    got = extract_memories("我吃阿司匹林过敏", "……")
    check("正常数组能解析", got == [{"kind": "allergy", "text": "对阿司匹林过敏"}], str(got))

    ltm.get_llm = lambda *a, **k: _FakeLLM(
        '```json\n[{"kind": "history", "text": "2 型糖尿病"}]\n```')
    check("``` 围栏能剥掉", extract_memories("q", "a")[0]["kind"] == "history")

    ltm.get_llm = lambda *a, **k: _FakeLLM('好的：[{"kind": "history", "text": "哮喘"}] 以上')
    check("前后有杂话也能截出数组", extract_memories("q", "a")[0]["text"] == "哮喘")

    ltm.get_llm = lambda *a, **k: _FakeLLM("[]")
    check("空数组 → 无记忆", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM('[{"kind": "不存在的类型", "text": "x"}]')
    check("非法 kind 被过滤", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM('[{"kind": "history"}]')
    check("缺 text 被过滤", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM('{"kind": "history"}')
    check("对象而非数组 → 空", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM("不是 JSON")
    check("非 JSON → 空", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM(None)
    check("None → 空", extract_memories("q", "a") == [])
    ltm.get_llm = lambda *a, **k: _FakeLLM(boom=True)
    check("provider 抛异常 → 空（不影响本轮问答）", extract_memories("q", "a") == [])

    # ------------------------------------------------------------ 4. learn_from_turn
    print("== 4. learn_from_turn 只报「新写入」的 ==")
    ltm.get_llm = lambda *a, **k: _FakeLLM(
        '[{"kind": "history", "text": "高血压 10 年"}, {"kind": "medication", "text": "长期服用氯沙坦"}]')
    written = learn_from_turn("我有高血压", "……")
    check("第一条已存在 ⇒ 只报新增的 1 条",
          len(written) == 1 and written[0]["text"] == "长期服用氯沙坦", str(written))
    ltm.get_llm = lambda *a, **k: _FakeLLM("[]")
    check("没抽到 → 空列表", learn_from_turn("q", "a") == [])

    # ------------------------------------------------------------ 5. 图级接入
    print("== 5. node_recall / node_remember ==")
    with patch.object(g.settings, "long_term_memory_enabled", False):
        check("开关关 → node_recall 空 update", g.node_recall({"dialogue_act": "new_question"}) == {})
        check("开关关 → node_remember 空 update",
              g.node_remember({"question": "q", "answer": "a"}) == {})
    with patch.object(g.settings, "long_term_memory_enabled", True):
        out = g.node_recall({"dialogue_act": "new_question"})
        check("开关开 → 注入 long_term_memory", out.get("long_term_memory", "").startswith("【用户长期病史"))
        check("trace 记了「长期记忆」并报条数",
              any(t.get("step") == "长期记忆" and t.get("count") == len(recall_items())
                  for t in out.get("trace", [])), str(out.get("trace")))
        check("ack 轮不读记忆", g.node_recall({"dialogue_act": "ack"}) == {})
        ltm.get_llm = lambda *a, **k: _FakeLLM('[{"kind": "allergy", "text": "对磺胺过敏"}]')
        out2 = g.node_remember({"question": "我磺胺过敏", "answer": "……"})
        check("开关开 → 写记忆并报 trace",
              out2.get("memory_written") and any(t.get("step") == "记忆写入" for t in out2.get("trace", [])),
              str(out2))

    print("== 6. 注入【用户情况】的形态 ==")
    plain = g._build_user_statement([{"role": "user", "content": "我头疼"}])
    check("无记忆时形态不变（逐字）", plain == "我头疼", plain)
    with_mem = g._build_user_statement([{"role": "user", "content": "我头疼"}], "【用户长期病史（跨会话记忆）】\n- 过敏史：对青霉素过敏")
    check("有记忆时病史在前", with_mem.startswith("【用户长期病史"))
    check("有记忆时本次主诉仍在末尾", with_mem.endswith("我头疼"), with_mem)
    check("带「不要提及」的约束（防把病史当主诉搬运，规则 2 的前提）",
          "除非与本次提问直接相关，否则不要提及" in with_mem)

    print("== 7. 跨轮重置 ==")
    reset = g.node_classify_intent({"question": "全新的问题"})
    check("重置 long_term_memory", reset.get("long_term_memory") == "")
    check("重置 memory_written", reset.get("memory_written") == [])

    ltm.get_llm = _ORIG_GET_LLM
    _reset_store()

shutil.rmtree(_TMP, ignore_errors=True)

print()
if _fails:
    print(f"❌ {len(_fails)} 项未通过：{_fails}")
    sys.exit(1)
print("全绿：跨会话长期记忆（读写 / 持久化 / 抽取 / 图接入 / 跨轮重置）均通过。")
sys.exit(0)
