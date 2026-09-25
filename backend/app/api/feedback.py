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


@router.get("/stats")
async def feedback_stats(limit: int = 50, db: Session = Depends(get_db)):
    """反馈汇总 —— 「人工介入」这条闭环的入口。

    为什么必须有它
    --------------
    只写不读的反馈表等于没做反馈：前端点了 👎、corrected_answer 也存进库了，
    但**没有任何出口**把它捞出来，没人会去翻数据库。这个接口把差评连同
    「用户当时问的是什么」一起列出来，闭环才合上：

        用户点 👎 → 落库 → GET /api/feedback/stats → 人工看 → 改 Prompt / 补语料

    `corrected_answer` 是用户自己写的正确答案，价值最高（现成的标注数据），
    所以单列一个 `with_correction` 计数。只读接口，不写库。
    """
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    up = sum(1 for r in rows if r.thumbs == "up")
    down_rows = [r for r in rows if r.thumbs == "down"]
    total = up + len(down_rows)

    items = []
    for r in down_rows[: max(1, min(limit, 200))]:
        msg = db.query(ChatMessage).filter(ChatMessage.id == r.message_id).first()
        # 反查这条回答对应的问题是哪个：同一会话里 id 紧邻在前的 user 消息
        question = None
        if msg:
            q = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.conversation_id == msg.conversation_id,
                    ChatMessage.id < msg.id,
                    ChatMessage.role == "user",
                )
                .order_by(ChatMessage.id.desc())
                .first()
            )
            question = q.content if q else None
        items.append({
            "feedback_id": r.id,
            "message_id": r.message_id,
            "question": question,
            "answer_head": (msg.content[:120] if msg else None),
            "corrected_answer": r.corrected_answer,
            "comment": r.comment,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {
        "total": total,
        "up": up,
        "down": len(down_rows),
        "down_rate": round(len(down_rows) / total, 4) if total else 0.0,
        "with_correction": sum(1 for r in rows if r.corrected_answer),
        "down_items": items,
    }