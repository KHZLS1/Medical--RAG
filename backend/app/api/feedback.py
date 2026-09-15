from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Feedback, ChatMessage

router = APIRouter(prefix="/api/feedback", tags=["回答反馈"])

class FeedbackRequest(BaseModel):
    message_id: int
    thumbs: str               # "up" | "down"
    corrected_answer: str | None = None
    comment: str | None = None

@router.post("")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.thumbs not in ("up", "down"):
        raise HTTPException(status_code=400, detail="thumbs 仅支持 up/down")
    msg = db.query(ChatMessage).filter(ChatMessage.id == req.message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="消息不存在")
    if msg.role != "assistant":
        raise HTTPException(status_code=400, detail="只能对助手回答评价")

    db.add(Feedback(
        message_id=req.message_id,
        thumbs=req.thumbs,
        corrected_answer=req.corrected_answer,
        comment=req.comment,
    ))
    db.commit()
    return {"ok": True}