"""数据库模型：上传文档元数据、会话与消息"""
from datetime import datetime

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship

from .database import Base

class UploadedDocument(Base):
    """上传文档元数据表（文件本身保留在磁盘 uploads/ 目录）"""
    __tablename__ = "uploaded_documents"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    filename = Column(String(255), nullable=False, comment="文件名")
    file_size = Column(Integer, nullable=False, comment="文件大小(字节)")
    file_type = Column(String(20), nullable=False, comment="文件扩展名")
    file_path = Column(String(500), nullable=False, comment="磁盘存储路径")
    uploaded_at = Column(DateTime, default=datetime.now, comment="上传时间")
    content_hash = Column(String(64), nullable=True, comment="文件内容 SHA-256")

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
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    conversation = relationship("Conversation", back_populates="messages")

class Feedback(Base):
    """回答反馈表：用户对助手回答的客观评价与纠错"""
    __tablename__ = "feedback"

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