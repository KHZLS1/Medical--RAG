"""Alembic 环境配置：直接复用 app 的 settings 与 Base.metadata。

为什么连接串不写进 alembic.ini
------------------------------
写进去就有两份真相（alembic.ini 与 .env），且密码会进版本库。这里直接 import
`app.config.settings`，与后端跑的是同一个 `DATABASE_URL` —— 谁也不用记得同步。

⚠️ `set_main_option` 的值会经过 configparser 插值，密码里若有 `%` 必须写 `%%`，
   否则会在建 Engine 时抛 InterpolationSyntaxError。下面统一做了转义。

常用命令（backend 目录）
------------------------
  python -m alembic current                      # 当前库处于哪个版本
  python -m alembic revision --autogenerate -m "说明"   # 按 models 变更生成迁移
  python -m alembic upgrade head                 # 应用迁移
  python -m alembic stamp head                   # 只记版本号、不执行 DDL（接管已有库用）
  python -m alembic downgrade -1                 # 回退一步
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# backend/ 进 sys.path，否则 import app.* 会失败（alembic 不会自动加项目根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings       # noqa: E402
from app.database import Base         # noqa: E402
from app import models                # noqa: E402,F401  必须 import：表定义注册到 Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

if not settings.database_url:
    raise RuntimeError(
        "未配置 DATABASE_URL（backend/.env）。Alembic 与后端共用同一份配置，"
        "请先按 .env.example 建好 .env。"
    )

# `%` 转义：configparser 会把单个 % 当插值符号
config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))

# autogenerate 的比对基准：models.py 里的全部表
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文本，不连库。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连库执行。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 打开类型比对：不然改了列类型 autogenerate 会一声不响地漏掉
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
