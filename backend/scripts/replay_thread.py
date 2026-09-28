"""time-travel 回放：把某个会话的图状态历史摊开（T62-⑨）

用途（badcase 复盘）
    线上答错一轮，事后想弄清"当时检索到了什么、走了哪条分支、为什么判成有资料" ——
    光看日志不够，因为 trace 只保留了**最终**那一份。而 checkpointer 其实把
    **每一步之后**的完整状态都存下来了，`aget_state_history` 能把它们按时间倒序取出。
    这就是 LangGraph 的 time-travel 能力：不改代码、不重跑（重跑还未必复现，
    见 README「评估不可复现」），直接读当时那份状态。

为什么是脚本而不是 API
    它面向开发者、不面向终端用户，也不需要鉴权与前端。放 `scripts/` 下与
    `eval_dialogue.py` 同类。

⚠️ 必须走 `aget_state_history`（异步）：checkpointer 是 `AsyncSqliteSaver`，
    它的同步方法在事件循环线程里会抛 `InvalidStateError`（见 app/checkpointer.py）。
⚠️ 需要 `GRAPH_CHECKPOINTER_ENABLED=true`（否则图没带 checkpointer，没有历史可读）。

跑法：
    python scripts/replay_thread.py                 # 列出最近的会话（拿 id）
    python scripts/replay_thread.py 6               # 回放会话 6 的全部步骤
    python scripts/replay_thread.py 6 --step 2      # 只看第 2 步（时间倒序，0 = 最新）
    python scripts/replay_thread.py 6 --full        # 每一步都打印完整状态
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings


def list_conversations(limit: int = 20) -> list[tuple]:
    """列出最近更新的会话（id / 标题 / 更新时间），供选择要回放的 thread。"""
    from app.database import SessionLocal
    from app.models import Conversation

    db = SessionLocal()
    try:
        rows = (db.query(Conversation)
                .order_by(Conversation.updated_at.desc())
                .limit(limit).all())
        return [(r.id, r.title or "", r.updated_at) for r in rows]
    finally:
        db.close()


def brief(values: dict) -> str:
    """把一步的状态压成一行摘要（只挑诊断真正要看的字段）。"""
    bits: list[str] = []
    if values.get("intent"):
        bits.append(f"intent={values['intent']}")
    if values.get("dialogue_act"):
        bits.append(f"act={values['dialogue_act']}")
    if values.get("evidence_state"):
        bits.append(f"证据={values['evidence_state']}")
    ts = values.get("top_score")
    if isinstance(ts, (int, float)):
        bits.append(f"top={ts:.3f}")
    if values.get("grade"):
        bits.append(f"分级={values['grade'].get('verdict')}")
    if values.get("tool_result"):
        bits.append(f"工具={values['tool_result'].get('tool')}")
    if values.get("memory_written"):
        bits.append(f"记了{len(values['memory_written'])}条")
    if values.get("answer_degraded"):
        bits.append("⚠️降级回答")
    return "  ".join(bits) or "—"


def steps_of(values: dict) -> list[str]:
    return [t.get("step") for t in (values.get("trace") or []) if isinstance(t, dict)]


def print_step(idx: int, snap, full: bool) -> None:
    meta = snap.metadata or {}
    values = snap.values or {}
    nxt = snap.next or ()
    print(f"[{idx}] step={meta.get('step')}  source={meta.get('source')}  "
          f"下一跳={nxt or '(END)'}")
    q = (values.get("question") or "").strip()
    if q:
        print(f"    问题: {q[:80]}")
    print(f"    状态: {brief(values)}")
    steps = steps_of(values)
    if steps:
        print(f"    步骤: {' → '.join(steps)}")
    ans = (values.get("answer") or "").strip()
    if ans:
        print(f"    回答: {ans[:80].replace(chr(10), ' ')}")
    if full:
        # 完整状态可能很大（docs 全文），逐字段截断打印
        print("    ---- 完整状态（截断）----")
        for k in sorted(values):
            v = values[k]
            s = json.dumps(v, ensure_ascii=False, default=str)
            print(f"      {k}: {s[:300]}")
    print()


async def replay(conv_id: int, step: int | None, full: bool) -> int:
    if not settings.graph_checkpointer_enabled:
        print("❌ 当前 GRAPH_CHECKPOINTER_ENABLED=false ⇒ 图没有 checkpointer，"
              "没有历史可读。")
        print("   （它是启动期配置，改 .env 后需重启后端。）")
        return 2
    from app.graph import get_graph, thread_config

    graph = await get_graph()
    snaps = [s async for s in graph.aget_state_history(thread_config(conv_id))]
    if not snaps:
        print(f"会话 {conv_id} 没有图状态历史（可能从未走过图，或快照已被清理）。")
        return 1

    # 时间**倒序**：数组第 0 项是最近一次。
    print("=" * 78)
    print(f"会话 {conv_id} 的图状态历史：{len(snaps)} 个快照（时间倒序，0 = 最新）")
    print("=" * 78)
    if step is not None:
        if not 0 <= step < len(snaps):
            print(f"❌ --step 超出范围（0~{len(snaps) - 1}）")
            return 2
        print_step(step, snaps[step], full=True)
        return 0
    for i, snap in enumerate(snaps):
        print_step(i, snap, full)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="回放某会话的 LangGraph 状态历史（time-travel）")
    ap.add_argument("conversation_id", nargs="?", type=int,
                    help="会话 id；不传则只列出最近会话")
    ap.add_argument("--step", type=int, default=None,
                    help="只看第 N 个快照（时间倒序，0 = 最新），并打印完整状态")
    ap.add_argument("--full", action="store_true", help="每个快照都打印完整状态")
    ap.add_argument("--limit", type=int, default=20, help="列出会话时的条数上限")
    args = ap.parse_args()

    if args.conversation_id is None:
        rows = list_conversations(args.limit)
        if not rows:
            print("没有会话。")
            return 1
        print(f"最近 {len(rows)} 个会话（用 id 回放：python scripts/replay_thread.py <id>）")
        for cid, title, updated in rows:
            stamp = updated.strftime("%m-%d %H:%M") if updated else "—"
            print(f"  {cid:>5}  {stamp}  {title[:40]}")
        return 0

    return asyncio.run(replay(args.conversation_id, args.step, args.full))


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    # ⚠️ 用 os._exit 而不是 sys.exit：`get_graph()` 会拉起 Milvus 客户端、torch、
    # SQLAlchemy 连接池等一批**非守护线程**，正常退出时进程会挂住不返回
    # （实测要等 120s 被 timeout 杀掉）。诊断脚本不需要优雅清理 —— 数据早已落盘，
    # 这里就是全部要展示的输出。
    os._exit(_code)
