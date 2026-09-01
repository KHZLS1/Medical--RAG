"""数据库模型：上传文档元数据"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime

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