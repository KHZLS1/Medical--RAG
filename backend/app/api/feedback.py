from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Feedback, ChatMessage

router = APIRouter(prefix="/api/feedback", tags=["回答反馈"])

class FeedbackRequest(BaseModel):
    message_id: int
    thumbs: str               # "up" | "down"
    corrected_answer: str | None = None
    comment: str | None = None

def _get_valid_thumbs(thumbs: str) -> str:
    if thumbs not in ("up", "down"):
        raise HTTPException(status_code=400, detail="thumbs 仅支持 up/down")
    return thumbs

@router.post("")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """提交/更新反馈。幂等：同一 message_id 只保留一条，重复提交视为更新。"""
    _get_valid_thumbs(req.thumbs)
    msg = db.query(ChatMessage).filter(ChatMessage.id == req.message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="消息不存在")
    if msg.role != "assistant":
        raise HTTPException(status_code=400, detail="只能对助手回答评价")

    existing = (
        db.query(Feedback)
        .filter(Feedback.message_id == req.message_id)
        .first()
    )
    if existing:
        existing.thumbs = req.thumbs
        existing.corrected_answer = req.corrected_answer or None
        existing.comment = req.comment or None
        fb = existing
    else:
        fb = Feedback(
            message_id=req.message_id,
            thumbs=req.thumbs,
            corrected_answer=req.corrected_answer or None,
            comment=req.comment or None,
        )
        db.add(fb)
    db.commit()
    db.refresh(fb)
    return {"ok": True, "feedback_id": fb.id, "thumbs": fb.thumbs}


@router.delete("/{message_id}")
async def withdraw_feedback(message_id: int, db: Session = Depends(get_db)):
    """撤回反馈：撤销对某条助手消息的评价。幂等，删除不存在的反馈也返回 ok。"""
    fb = db.query(Feedback).filter(Feedback.message_id == message_id).first()
    if fb:
        db.delete(fb)
        db.commit()
    return {"ok": True, "message_id": message_id}


@router.get("/stats")
async def feedback_stats(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """反馈汇总 —— 「人工介入」这条闭环的入口。

    只写不读的反馈表等于没做反馈：这个接口把差评连同「用户当时问的是什么」
    一起列出来，`corrected_answer`（用户自己写的正确答案，价值最高的标注数据）
    单独计数，闭环才合上。

       用户点 👎 → 落库 → GET /api/feedback/stats → 人工看 → 改 Prompt / 补语料

    查询策略：统计用一次 GROUP BY 聚合、清单分页只取当前页，回答消息用一次 IN 查回，
    question 逐条反查。相比原先把全部反馈 .all() 拉进内存再逐条 3 次查询，随反馈量
    增长的只有「当前页」的构建，避免全表加载。
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    counts = (
        db.query(Feedback.thumbs, func.count(Feedback.id))
        .group_by(Feedback.thumbs)
        .all()
    )
    by_thumbs = {t: c for t, c in counts}
    up = by_thumbs.get("up", 0)
    down_total = by_thumbs.get("down", 0)
    total = up + down_total
    with_correction = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.corrected_answer.isnot(None))
        .scalar()
        or 0
    )

    down_rows = (
        db.query(Feedback)
        .filter(Feedback.thumbs == "down")
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())  # 同秒稳定排序
        .offset(offset)
        .limit(limit + 1)  # 多取 1 条判断是否有下一页
        .all()
    )
    has_more = len(down_rows) > limit
    down_rows = down_rows[:limit]

    if not down_rows:
        return {
            "total": total, "up": up, "down": down_total,
            "down_rate": round(down_total / total, 4) if total else 0.0,
            "with_correction": with_correction,
            "down_items": [], "offset": offset, "has_more": has_more,
        }

    # 一次 IN 查回当前页涉及的回答消息
    msg_ids = [r.message_id for r in down_rows]
    msg_map = {
        m.id: m
        for m in db.query(ChatMessage).filter(ChatMessage.id.in_(msg_ids)).all()
    }

    items = []
    for r in down_rows:
        msg = msg_map.get(r.message_id)
        # 反查这条回答对应的问题：同一会话里 id 紧邻在前的 user 消息
        question = None
        if msg:
            q = (
                db.query(ChatMessage.content)
                .filter(
                    ChatMessage.conversation_id == msg.conversation_id,
                    ChatMessage.id < msg.id,
                    ChatMessage.role == "user",
                )
                .order_by(ChatMessage.id.desc())
                .first()
            )
            question = q[0] if q else None
        items.append({
            "feedback_id": r.id,
            "message_id": r.message_id,
            "question": question,
            "answer_head": (msg.content[:120] if msg else None),
            "corrected_answer": r.corrected_answer,
            "comment": r.comment,
            "thumbs": r.thumbs,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {
        "total": total, "up": up, "down": down_total,
        "down_rate": round(down_total / total, 4) if total else 0.0,
        "with_correction": with_correction,
        "down_items": items, "offset": offset, "has_more": has_more,
    }