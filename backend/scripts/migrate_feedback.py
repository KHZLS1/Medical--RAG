"""执行 feedback 幂等迁移：清理历史重复反馈 + 加唯一约束 + 加分页索引。

连接信息从 backend/.env 的 DATABASE_URL 解析（与 init_db.py 一致）。
SQL 语句来自同目录的 migrate_feedback_unique.sql，本脚本逐条执行并校验。

用法:
  cd backend
  python scripts/migrate_feedback.py

说明: 会先删除每个 message_id 里除最新外重复的反馈行（不可回退），
      再加 UNIQUE(message_id)。执行前可先备份 feedback 表。
"""
import sys
import pathlib
from urllib.parse import urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pymysql

from app.config import settings

CONSTRAINT_NAME = "uq_feedback_message_id"


def _read_statements() -> list[str]:
    """读取同目录 .sql，剔除注释/空行后按分号切分为可执行语句。"""
    sql_path = pathlib.Path(__file__).resolve().parent / "migrate_feedback_unique.sql"
    lines = [
        ln.strip()
        for ln in sql_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("--")
    ]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def main():
    if not settings.database_url:
        print("[ERROR] 未配置 DATABASE_URL，请在 backend/.env 中填写")
        return

    u = urlparse(settings.database_url)
    db_name = (u.path or "/medical_rag").lstrip("/")

    conn = pymysql.connect(
        host=u.hostname or "127.0.0.1",
        port=u.port or 3306,
        user=u.username or "root",
        password=u.password or "",
        database=db_name,   # 目标库必须已存在（init_db.py 创建）
        charset="utf8mb4",
    )
    cur = conn.cursor()

    for stmt in _read_statements():
        print(f"[运行] {stmt.splitlines()[0][:80]} ...")
        cur.execute(stmt)
        conn.commit()

    # 校验唯一约束是否生效
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS "
        "WHERE CONSTRAINT_SCHEMA=%s AND TABLE_NAME='feedback' "
        "AND CONSTRAINT_NAME=%s",
        (db_name, CONSTRAINT_NAME),
    )
    exists = cur.fetchone()[0] > 0
    cur.execute(
        "SELECT COUNT(DISTINCT message_id), COUNT(*) FROM feedback"
    )
    distinct, total = cur.fetchone()
    print(f"[校验] 唯一约束 {CONSTRAINT_NAME} 生效: {exists}")
    print(f"[校验] feedback 行数: {total}（distinct message_id: {distinct}）")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()