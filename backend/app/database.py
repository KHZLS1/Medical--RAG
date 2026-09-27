"""数据库连接与会话管理"""
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# 创建数据库引擎
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # 连接前是否有效
    pool_recycle=3600,  # 连接回收时间（秒）
    echo=False,         # 生产环境关闭 SQL 日志
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 约束命名规则（配合 Alembic）。
# 为什么需要它：不给约束起名前，MySQL 会自动命名外键为 `<表名>_ibfk_<序号>`，
# 序号取决于建表顺序 —— 于是"同一个 schema、不同机器"的外键名可能不同，
# alembic autogenerate 会把它当成"约束不存在"而反复生成 create_foreign_key/drop，
# 迁移历史里全是噪音（本项目实测踩到过）。
# 注意：规则只影响**新建**约束；已有库需要一次性改名（见 alembic/versions 的说明）。
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    # 刻意不把被引用的表名也拼进来（"fk_feedback_message_id_chat_messages" 太长）：
    # "fk_<表>_<列>" 在单列外键上不会重名，且与真库已对齐的名字一致。
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}

# ORM 基类
Base = declarative_base(metadata=MetaData(naming_convention=NAMING_CONVENTION))

def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()