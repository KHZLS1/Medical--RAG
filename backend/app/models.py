"""数据库模型：上传文档元数据、会话与消息

⚠️ 本文件是 schema 的**唯一声明**（`alembic revision --autogenerate` 以它为准）。
   改字段后请立刻生成迁移；反过来，手工在库里执行 DDL 也要同步写回这里，
   否则 autogenerate 会一直报漂移（本项目已踩过：真库里多出 content_hash 的
   唯一索引与 feedback 的复合索引，而 models 没写）。
"""
from datetime import datetime

from sqlalchemy import (
    CHAR,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base

class UploadedDocument(Base):
    """上传文档元数据表（文件本身保留在磁盘 uploads/ 目录）"""
    __tablename__ = "uploaded_documents"
    # 内容哈希唯一：同一份文件重复上传直接挡在库层面（接口层给 409）。
    # 这里保留显式名而不是交给命名的约定：该索引名已被 README 的升级 SQL 引用，
    # 改名会让文档失效，收益不值。
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_uploads_content_hash"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    filename = Column(String(255), nullable=False, comment="文件名")
    file_size = Column(Integer, nullable=False, comment="文件大小(字节)")
    file_type = Column(String(20), nullable=False, comment="文件扩展名")
    file_path = Column(String(500), nullable=False, comment="磁盘存储路径")
    uploaded_at = Column(DateTime, default=datetime.now, comment="上传时间")
    # CHAR 而非 VARCHAR：SHA-256 十六进制串定长 64，CHAR 更贴合且比较更快
    content_hash = Column(CHAR(64), nullable=True, comment="文件内容 SHA-256")

class Conversation(Base):
    """对话会话表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="会话ID")
    title = Column(String(200), nullable=False, default="新对话", comment="会话标题")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="最后更新时间")

    # 删除会话时由 ORM 级联删除其所有消息，避免遗留孤儿数据
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

class ChatMessage(Base):
    """对话消息表"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="消息ID")
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        comment="所属会话ID",
    )
    role = Column(String(20), nullable=False, comment="角色: user/assistant")
    content = Column(Text, nullable=False, comment="消息内容")
    sources = Column(Text, nullable=True, comment="引用来源(JSON字符串)")
    # 阶段二：证据不足时后端 interrupt 追问，追问话术也作为一条助手消息落库。
    # 没有这个标记，前端刷新后只能把它当普通回答渲染，「🔎 请补充信息」徽章丢失。
    is_clarification = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="本条是否为证据不足的追问话术（阶段二）",
    )
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    conversation = relationship("Conversation", back_populates="messages")

class Feedback(Base):
    """回答反馈表：用户对助手回答的客观评价与纠错"""
    __tablename__ = "feedback"
    # 一条助手消息只允许一条反馈：重复点 👍/👎 在库层面直接挡掉，
    # 业务上重复提交更新同一条记录而非再插一行（见 api/feedback.py submit_feedback）。
    # 另有一条 (thumbs, created_at) 复合索引，服务反馈看板的"按评价 + 时间"统计，
    # 之前只存在于真库、没进 models —— 那正是 autogenerate 报"要删掉它"的原因。
    __table_args__ = (
        UniqueConstraint("message_id", name="uq_feedback_message_id"),
        Index("idx_feedback_thumbs_time", "thumbs", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    message_id = Column(
        Integer,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        comment="关联助手消息ID",
    )
    thumbs = Column(String(10), nullable=False, comment="评价: up/down")
    corrected_answer = Column(Text, nullable=True, comment="用户纠错文本(可选)")
    comment = Column(String(500), nullable=True, comment="补充说明")
    created_at = Column(DateTime, default=datetime.now, comment="提交时间")