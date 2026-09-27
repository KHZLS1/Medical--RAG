"""审计孤儿消息：`conversation_id` 指向已不存在会话的 chat_messages 行。

现状（2026-09-27 之后）
----------------------
外键 `fk_chat_messages_conversation_id (ON DELETE CASCADE)` 已由 alembic revision
`0f6e3b0d0396` 在真库建起，**孤儿消息从此不会再产生**（删会话即级联删消息）。
所以本脚本现在的定位是**审计 + 备份工具**，而不是常规清理手段：
  · 日常用途 —— 跑一次确认孤儿为 0，作为「级联真的在生效」的证据；
  · 应急用途 —— 若在别的环境（比如某台没跑过 0f6e3b0d0396 的旧库）又出现孤儿，
    用 `--apply` 清理后再 `alembic upgrade head` 收敛。此时它仍是必需的，
    因为带孤儿数据建外键这条 DDL 必然失败。

为什么当初会有孤儿
------------------
`chat_messages.conversation_id` 在 models 里声明了 `ForeignKey(..., ondelete="CASCADE")`，
但**真库里这个外键从来没建起来**（表建在前、FK 声明加在后，`create_all` 不会补结构）。
于是 MySQL 侧没有级联保护，会话被删时消息留下来 —— 实测曾积到 21 条。

安全性
------
孤儿消息在业务上**不可达**：前端按 `conversation_id` 拉消息，孤儿永远不会被显示。
但删之前仍然：① 导出完整 JSON 备份；② 打印每一行；③ 默认只做 dry-run。

用法（backend 目录）
--------------------
  python scripts/cleanup_orphan_messages.py            # 只报告 + 备份（默认）
  python scripts/cleanup_orphan_messages.py --apply    # 真删（会先写备份）
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

BACKUP_DIR = Path(__file__).resolve().parent.parent / "data" / "backup"


def _connect(db_name: str):
    u = urlparse(settings.database_url)
    return pymysql.connect(
        host=u.hostname or "127.0.0.1", port=u.port or 3306,
        user=u.username or "root", password=u.password or "",
        database=db_name, charset="utf8mb4",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="清理指向已删会话的孤儿消息")
    ap.add_argument("--apply", action="store_true",
                    help="真的删除（默认只报告并备份）")
    args = ap.parse_args()

    if not settings.database_url:
        print("[ERROR] 未配置 DATABASE_URL")
        return 1

    db_name = (urlparse(settings.database_url).path or "/medical_rag").lstrip("/")
    conn = _connect(db_name)
    cur = conn.cursor(pymysql.cursors.DictCursor)

    cur.execute(
        "SELECT m.id, m.conversation_id, m.role, m.content, m.created_at "
        "FROM chat_messages m LEFT JOIN conversations c ON m.conversation_id = c.id "
        "WHERE c.id IS NULL ORDER BY m.id"
    )
    orphans = cur.fetchall()

    print(f"库：{db_name}")
    print(f"孤儿消息：{len(orphans)} 条")
    for r in orphans[:20]:
        preview = (r["content"] or "").replace("\n", " ")[:40]
        print(f"  id={r['id']:>6}  conversation_id={r['conversation_id']:>6}  "
              f"{r['role']:<9} {preview}")
    if len(orphans) > 20:
        print(f"  … 其余 {len(orphans) - 20} 条见备份文件")

    if not orphans:
        print("没有孤儿消息，无需清理。")
        conn.close()
        return 0

    # 备份（无论 dry-run 与否都写，方便用户自己核对）
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"orphan_messages_{stamp}.json"
    backup.write_text(json.dumps(
        [dict(r, created_at=r["created_at"].isoformat() if r["created_at"] else None)
         for r in orphans], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[备份] {backup}")

    if not args.apply:
        print("\n[dry-run] 未做任何删除。确认无误后加 --apply 执行。")
        print("         注意：这些 id 的反馈（feedback 表）会随外键级联一并删除。")
        conn.close()
        return 0

    ids = [r["id"] for r in orphans]
    placeholders = ",".join(["%s"] * len(ids))
    cur.execute(f"DELETE FROM chat_messages WHERE id IN ({placeholders})", ids)
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"\n[已删除] {deleted} 条孤儿消息（备份见上）。")
    print("下一步：python -m alembic upgrade head（revision 0f6e3b0d0396 会补上外键）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
