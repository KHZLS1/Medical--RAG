"""badcase 回流：把反馈表里的差评导出成可用的评估素材（JSONL）。

为什么需要它
------------
反馈数据现在是**死水**：用户点了 👎、填了纠错文本，它们只躺在 `feedback` 表里
给看板做个计数，从不回到评估链路（T60 记的就是这笔账）。没有回流，"评估分高"
就永远是自说自话 —— 真实用户判错的那些题，才是下一轮该盯的地方。

导出的每一行是一个**自足的样本**：提问 + 当时给的回答 + 用户写的纠正 + 备注。
需要人工过一遍再决定进哪个评估集：

  · 检索类差评（答非所问、没找到资料）→ 补进 `data/eval/eval_testset.json` 的思路
  · 生成类差评（编内容、引用错）      → 补进对抗集，或作为 `GROUNDING_HEDGE` /
                                        `detect_unanswerable` 的标定素材
  · 纯口味问题（"回答太长"）          → 直接丢弃，别污染指标

刻意**不做自动入库**：反馈是用户的主观判断，直接进评估集会把"用户当时的口味"
固化成"正确答案"，指标会失真。脚本只负责把料备好 + 给出分组统计。

用法（backend 目录）
--------------------
  python scripts/export_badcases.py                     # 差评 → data/eval/badcases.jsonl
  python scripts/export_badcases.py --include-up        # 连好评一起导（做对照用）
  python scripts/export_badcases.py --out /tmp/bc.jsonl
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql  # noqa: E402

from app.config import settings  # noqa: E402

_QUERY = """
SELECT f.id            AS feedback_id,
       f.message_id    AS message_id,
       f.thumbs        AS thumbs,
       f.corrected_answer AS corrected_answer,
       f.comment       AS comment,
       f.created_at    AS created_at,
       m.conversation_id AS conversation_id,
       m.content       AS answer,
       (SELECT m2.content FROM chat_messages m2
         WHERE m2.conversation_id = m.conversation_id
           AND m2.role = 'user' AND m2.id < m.id
         ORDER BY m2.id DESC LIMIT 1) AS question
  FROM feedback f
  JOIN chat_messages m ON m.id = f.message_id
 WHERE f.thumbs IN ({thumbs})
 ORDER BY f.created_at DESC
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="导出反馈差评，形成 badcase 素材")
    ap.add_argument("--out", default="data/eval/badcases.jsonl",
                    help="输出路径（相对 backend）")
    ap.add_argument("--include-up", action="store_true",
                    help="连好评一起导出（做对照）")
    args = ap.parse_args()

    if not settings.database_url:
        print("[ERROR] 未配置 DATABASE_URL（backend/.env）")
        return 1

    u = urlparse(settings.database_url)
    db_name = (u.path or "/medical_rag").lstrip("/")
    try:
        conn = pymysql.connect(host=u.hostname or "127.0.0.1", port=u.port or 3306,
                               user=u.username or "root", password=u.password or "",
                               database=db_name, charset="utf8mb4",
                               cursorclass=pymysql.cursors.DictCursor,
                               connect_timeout=5)
    except Exception as e:
        print(f"[ERROR] 连不上 MySQL（{db_name}）: {type(e).__name__}: {e}")
        return 1

    thumbs = ("'down'", "'up'") if args.include_up else ("'down'",)
    cur = conn.cursor()
    try:
        cur.execute(_QUERY.format(thumbs=",".join(thumbs)))
        rows = cur.fetchall()
    except Exception as e:
        print(f"[ERROR] 查询失败（feedback 表结构变更过？）: {type(e).__name__}: {e}")
        conn.close()
        return 1
    conn.close()

    out_path = Path(__file__).resolve().parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with_correction = 0
    no_question = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            question = r.get("question")
            if not question:
                # 反馈挂在了会话首条助手消息上（前面没有用户发言）—— 无提问可回溯，
                # 这样的样本没法用来评估，单独计数提示
                no_question += 1
            if (r.get("corrected_answer") or "").strip():
                with_correction += 1
            item = {
                "question": question,
                "answer": r.get("answer"),
                "corrected_answer": r.get("corrected_answer"),
                "comment": r.get("comment"),
                "thumbs": r.get("thumbs"),
                "message_id": r.get("message_id"),
                "conversation_id": r.get("conversation_id"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"库：{db_name}")
    print(f"导出：{len(rows)} 条 → {out_path}")
    print(f"  · 带用户纠错（corrected_answer）：{with_correction} 条")
    print(f"  · 无对应提问（首条消息，不可用）：{no_question} 条")
    print()
    if not rows:
        print("提示：库里的反馈还不足以形成回流素材。先去工作台点几条 👎 再回来跑。")
        return 0
    print("下一步（人工，别跳过）：")
    print("  1. 逐条判断属于「检索错 / 生成错 / 口味问题」")
    print("  2. 检索错 → 补进 eval_testset；生成错 → 补进对抗集或做阈值标定素材")
    print("  3. 口味问题直接丢弃 —— 别把它固化成「正确答案」，会让指标失真")
    return 0


if __name__ == "__main__":
    sys.exit(main())
