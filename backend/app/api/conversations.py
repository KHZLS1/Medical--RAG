"""会话管理路由：创建、列表、详情、删除、更新标题"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database import get_db
from ..models import Conversation, ChatMessage

router = APIRouter(prefix="/api/conversations", tags=["会话管理"])

# ===== 请求模型 =====
class CreateConversationRequest(BaseModel):
    title: str | None = None

class UpdateConversationRequest(BaseModel):
    title: str

# ===== 工具函数 =====
def _message_to_dict(msg: ChatMessage) -> dict:
    import json
    return {
        "id": msg.id,
        "conversation_id": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "sources": json.loads(msg.sources) if msg.sources else None,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }

def _conversation_to_dict(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }

# ===== 接口 =====
@router.get("", summary="获取会话列表")
async def list_conversations(db: Session = Depends(get_db)):
    """获取所有会话，按最后更新时间倒序排列"""
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return {
        "conversations": [_conversation_to_dict(conv) for conv in convs],
    }

@router.post("", summary="创建新会话")
async def create_conversation(
    req: CreateConversationRequest,
    db: Session = Depends(get_db),
):
    """创建一个新的对话会话"""
    conv = Conversation(title=req.title or "新对话")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _conversation_to_dict(conv)

@router.get("/{conv_id}", summary="获取会话详情")
async def get_conversation(conv_id: int, db: Session = Depends(get_db)):
    """获取单个会话的基本信息"""
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _conversation_to_dict(conv)

@router.put("/{conv_id}", summary="更新会话标题")
async def update_conversation(
    conv_id: int,
    req: UpdateConversationRequest,
    db: Session = Depends(get_db),
):
    """更新会话标题"""
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    conv.title = req.title
    db.commit()
    db.refresh(conv)
    return _conversation_to_dict(conv)

@router.delete("/{conv_id}", summary="删除会话")
async def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    """删除会话及其所有消息（消息由 relationship 的 cascade 级联删除）"""
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    db.delete(conv)   # cascade="all, delete-orphan" 会一并删除关联消息
    db.commit()

    return {"deleted": conv_id}

@router.get("/{conv_id}/messages", summary="获取会话消息列表")
async def list_messages(conv_id: int, db: Session = Depends(get_db)):
    """获取指定会话的所有消息，按时间正序排列"""
    # 先检查会话是否存在
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conv_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return {
        "messages": [_message_to_dict(m) for m in messages]
    }