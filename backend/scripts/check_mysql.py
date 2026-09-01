"""快速检查 MySQL 连接和 medical_rag 数据库是否存在

连接信息从 backend/.env 的 DATABASE_URL 解析，避免硬编码密码。

用法:
  cd backend
  python scripts/check_mysql.py
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

    try:
        conn = pymysql.connect(
            host=u.hostname or "127.0.0.1",
            port=u.port or 3306,
            user=u.username or "root",
            password=u.password or "",
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute(f"SHOW DATABASES LIKE '{db_name}'")
        result = cur.fetchall()
        if result:
            print(f"[OK] database '{db_name}' exists")
        else:
            print(f"[MISSING] database '{db_name}' NOT exists - run: python scripts/init_db.py")
        conn.close()
    except Exception as e:
        print(f"[ERROR] MySQL connection failed: {e}")


if __name__ == "__main__":
    main()
