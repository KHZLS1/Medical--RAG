"""创建 medical_rag 数据库（如果不存在）

连接信息从 backend/.env 的 DATABASE_URL 解析，避免硬编码密码。

用法:
  cd backend
  python scripts/init_db.py
"""
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql

from app.config import settings


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
    )
    cur = conn.cursor()

    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()

    cur.execute(f"SHOW DATABASES LIKE '{db_name}'")
    result = cur.fetchall()
    print(f"database '{db_name}' exists: {len(result) > 0}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
