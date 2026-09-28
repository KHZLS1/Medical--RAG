"""跨会话长期记忆（T62-⑦）—— LangGraph BaseStore

与 focus_entity 的分工（**别把这两件事混起来**）
    `focus_entity` 是 **thread 内**的：靠 checkpointer 保留，换会话就没了，
    它记住的是"这一轮在聊哪个病"。
    本模块是 **跨会话**的：记住"这个用户身上长期成立的事"—— 慢病史、过敏史、长期用药。
    两者的**生命周期不同**，塞进同一个字段必然互相污染，所以刻意分成两套存储。

为什么用 langgraph 的 SqliteStore，而不是再写一张 MySQL 表
    ① 它是官方的 `BaseStore` 实现，接口稳定，与 checkpointer 同源（都是 sqlite）；
    ② 记忆是"图的上下文"，不是业务实体 —— 放进业务库会让"删会话"与"删记忆"的
       语义纠缠（会话删了，病史该不该删？答案是不该，所以它本就不该挂在会话上）；
    ③ 零新依赖，不用为了一个 KV 引入 Postgres。

⚠️ 为什么用**同步** `SqliteStore` 而不是 `AsyncSqliteStore`
    本项目所有图节点都是 sync（`def`），而 AsyncSqliteStore 只有 async 接口，
    sync 节点里没法 await。记忆读写是几条极小的 sqlite 语句（KB 级），
    在事件循环线程里同步执行的开销可以忽略；换成 async 就得把整条节点链改 async，
    那是另一个量级的改动（还会撞上 ⑧ 那条"节点级超时只支持 async"的坑）。
    **这是有意的取舍，不是漏考虑。**

⚠️ 建连接必须 `sqlite3.connect(path, isolation_level=None)`
    默认的隐式事务会让 store 内部 `BEGIN` 报
    `sqlite3.OperationalError: cannot start a transaction within a transaction`
    （实测：不传 isolation_level 时 `put()` 直接炸）。autocommit 模式才是它的预期用法。

只收三类事实
    慢病史 / 过敏史 / 长期用药。刻意**不收**"这次哪里疼"这类当轮信息 ——
    那是 focus_entity 的活。收进来会让记忆变成"什么都往里塞"，噪音盖过信号。

无鉴权说明
    当前没有多用户鉴权（T57 未做），所以 namespace 固定为 `("user", "default")`。
    T57 落地后按 user_id 分 namespace 即可，**存储结构不用动**。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from langgraph.store.sqlite import SqliteStore

from .config import settings
from .llm import get_llm

# 记忆只认这三类；kind 的取值与中文标签
KIND_LABELS: dict[str, str] = {
    "history": "慢病史",
    "allergy": "过敏史",
    "medication": "长期用药",
}

_NAMESPACE: tuple[str, ...] = ("user", "default")

_store: SqliteStore | None = None
_conn: sqlite3.Connection | None = None

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def get_store() -> SqliteStore:
    """SqliteStore 单例（同步）。首次调用建库、建表。

    ⚠️ `isolation_level=None` 不能省，理由见模块 docstring。
    """
    global _store, _conn
    if _store is not None:
        return _store
    db_path = Path(__file__).resolve().parent.parent / settings.long_term_memory_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(db_path, isolation_level=None)
    _store = SqliteStore(_conn)
    _store.setup()
    return _store


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------

def _make_key(kind: str, text: str) -> str:
    """同 kind + 同文本 ⇒ 同 key ⇒ put 幂等（天然去重，不用先查后写）。"""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def remember(kind: str, text: str) -> str | None:
    """写入一条记忆。kind 非法或文本为空则跳过（返回 None）。"""
    kind = (kind or "").strip().lower()
    text = (text or "").strip()
    if kind not in KIND_LABELS or not text:
        return None
    text = text[:120]
    get_store().put(_NAMESPACE, _make_key(kind, text),
                    {"kind": kind, "text": text})
    return text


def recall_items() -> list[dict]:
    """读出全部记忆，按 kind 排序（顺序稳定，便于断言与展示）。"""
    items = get_store().search(_NAMESPACE, limit=100)
    out = [{"kind": (it.value or {}).get("kind", ""),
            "text": (it.value or {}).get("text", "")} for it in items]
    return sorted([i for i in out if i["kind"] in KIND_LABELS and i["text"]],
                  key=lambda i: (list(KIND_LABELS).index(i["kind"]), i["text"]))


def recall_text() -> str:
    """把记忆拼成一段可注入 prompt 的文本；没有记忆就返回空串。"""
    return recall()[1]


def recall() -> tuple[list[dict], str]:
    """一次读全：(条目列表, 注入用文本)。分开返回是为了让 trace 能报出条数。"""
    items = recall_items()
    if not items:
        return [], ""
    lines = [f"- {KIND_LABELS[i['kind']]}：{i['text']}" for i in items]
    return items, "【用户长期病史（跨会话记忆）】\n" + "\n".join(lines)


def clear() -> None:
    """清空全部记忆（测试与"忘记我"类需求用）。"""
    store = get_store()
    for it in store.search(_NAMESPACE, limit=1000):
        store.delete(_NAMESPACE, it.key)


# --------------------------------------------------------------------------
# 抽取（唯一一次 LLM 调用；解析失败一律当"这轮没有可记的"）
# --------------------------------------------------------------------------

_EXTRACT_PROMPT = """从下面这一轮医患对话中，提取**长期成立、且对以后问诊有影响**的事实。

只提取这三类，其他一律不要：
  1. history     慢病史（长期存在的疾病，如"高血压 10 年""2 型糖尿病"）
  2. allergy     过敏史（如"对青霉素过敏"）
  3. medication  长期用药（如"长期服用华法林"）

**不要**提取：这次的症状、这次问的疾病、医生的建议、用户没说过的推断。
宁可不提，也不要把当轮信息写成长期事实。

【输出】只输出一个 JSON 数组，不要解释、不要代码块。元素格式：
  {{"kind": "history|allergy|medication", "text": "一句话事实"}}
没有任何可记的就输出 []

示例：
  用户：我妈有高血压，我最近头晕
  => [{{"kind": "history", "text": "家族史：母亲有高血压"}}]
  （"我最近头晕"是当轮症状，不收）

用户问题：{question}
助手回答：{answer}
"""


def extract_memories(question: str, answer: str) -> list[dict]:
    """调一次 LLM 抽取长期事实。**任何失败都返回 []**（记忆层绝不成为故障点）。"""
    try:
        prompt = _EXTRACT_PROMPT.format(question=question, answer=(answer or "")[:1500])
        raw = str(get_llm().invoke(prompt).content)
        text = _FENCE_RE.sub("", raw).strip()
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        arr = json.loads(text[start:end + 1])
        if not isinstance(arr, list):
            return []
        out = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).strip().lower()
            fact = str(item.get("text", "")).strip()
            if kind in KIND_LABELS and fact:
                out.append({"kind": kind, "text": fact[:120]})
        return out
    except Exception as e:
        print(f"[长期记忆] 抽取失败，跳过本轮记忆: {type(e).__name__}: {e}")
        return []


def learn_from_turn(question: str, answer: str) -> list[dict]:
    """抽一轮 → 写库，返回**新写入**的条目（已存在的会被 key 去重掉，不重复计入）。"""
    facts = extract_memories(question, answer)
    if not facts:
        return []
    before = {(i["kind"], i["text"]) for i in recall_items()}
    written = []
    for f in facts:
        remember(f["kind"], f["text"])
        if (f["kind"], f["text"]) not in before:
            written.append(f)
    return written
