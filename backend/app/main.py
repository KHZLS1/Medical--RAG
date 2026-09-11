"""FastAPI 主程序

提供接口:
  GET  /api/health              健康检查
  POST /api/ingest              触发数据入库 (返回任务概览)
  POST /api/chat                流式问答 (SSE)
"""
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.database import Base, engine, get_db, SessionLocal
from sqlalchemy.orm import Session
from .config import settings
from .rag_chain import stream_answer
from .query_rewriter import generate_title
from .vectorstore import add_documents
from .api.documents import router as documents_router
from .api.conversations import router as conversations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    #应用启动时自动建表
    Base.metadata.create_all(bind=engine)
    print(f"[启动] 数据库表已就绪: {settings.database_url}")

    """应用启动/关闭钩子"""
    print(f"[启动] 数据目录: {settings.data_dir_resolved}")
    print(f"[启动] Milvus: {settings.milvus_uri}, collection={settings.milvus_collection}")

    # 检索层就绪检查：载入 collection（BM25 索引由 Milvus 服务端维护，无需客户端预热）
    print("[启动] 正在载入 Milvus collection ...")
    try:
        from .vectorstore import get_milvus_client
        client = get_milvus_client()
        stats = client.get_collection_stats(settings.milvus_collection)
        rows = int(stats.get("row_count", 0)) if stats else 0
        print(f"[启动] Milvus 就绪: {settings.milvus_collection}, 共 {rows:,} 条")
    except Exception as e:
        print(f"[启动][警告] Milvus 不可用，问答会失败: {type(e).__name__}: {e}")

    yield
    print("[关闭] 服务退出")


app = FastAPI(
    title="医疗 RAG 问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - 允许前端 Vite 开发服务器
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(conversations_router)

# ===== 请求/响应 模型 =====
class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None

class IngestRequest(BaseModel):
    limit_per_file: int | None = 500   # 默认每个科室入库 500 条 (Demo 模式)


# ===== 健康检查 =====
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "milvus_uri": settings.milvus_uri,
        "collection": settings.milvus_collection,
        "data_dir": str(settings.data_dir_resolved),
    }


# ===== 数据入库 =====
@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """同步入库（小批量可用，大批量请用 scripts/ingest.py 后台跑）"""
    from .data_loader import load_medical_documents

    try:
        # 读取 CSV + 向量化都是同步阻塞调用，放进线程池避免卡住事件循环
        docs = await asyncio.to_thread(
            load_medical_documents, limit_per_file=req.limit_per_file
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not docs:
        raise HTTPException(status_code=400, detail="未加载到任何文档，请检查数据目录")

    await asyncio.to_thread(add_documents, docs)

    # 按科室统计
    from collections import Counter
    dept_count = Counter(d.metadata.get("department", "未知") for d in docs)

    return {
        "ingested": len(docs),
        "by_department": dict(dept_count),
        "collection": settings.milvus_collection,
    }


# ===== 流式入库 (SSE) =====
class IngestStreamRequest(BaseModel):
    limit_per_file: int | None = 500
    use_cleaned: bool = True

@app.post("/api/ingest-stream")
async def ingest_stream(req: IngestStreamRequest):
    """SSE 流式入库，实时推送入库进度

    返回 SSE 事件流:
      {"type": "start", "total": N}
      {"type": "progress", "done": X, "total": N, "progress": P, "department": "..."}
      {"type": "done", "total": N, "by_department": {...}}
      {"type": "error", "data": "..."}
    """
    import json as json_mod
    from .data_loader import iter_cleaned_documents, iter_medical_documents
    from collections import Counter

    async def event_generator():
        try:
            if req.use_cleaned:
                doc_iter = iter_cleaned_documents(limit_per_file=req.limit_per_file)
            else:
                doc_iter = iter_medical_documents(limit_per_file=req.limit_per_file)

            # 解析全部 CSV/JSON 属同步 IO，放线程池执行
            all_docs = await asyncio.to_thread(list, doc_iter)
            total = len(all_docs)

            if total == 0:
                yield {
                    "event": "message",
                    "data": json_mod.dumps(
                        {"type": "error", "data": "未加载到任何文档"}, ensure_ascii=False
                    ),
                }
                return

            yield {
                "event": "message",
                "data": json_mod.dumps(
                    {"type": "start", "total": total}, ensure_ascii=False
                ),
            }

            batch_size = 500
            done = 0
            dept_count = Counter()

            for i in range(0, total, batch_size):
                batch = all_docs[i:i + batch_size]
                current_dept = batch[0].metadata.get("department", "未知") if batch else ""

                await asyncio.to_thread(add_documents, batch)

                for d in batch:
                    dept_count[d.metadata.get("department", "未知")] += 1

                done += len(batch)
                progress = round(done / total * 100, 1)

                yield {
                    "event": "message",
                    "data": json_mod.dumps(
                        {
                            "type": "progress",
                            "done": done,
                            "total": total,
                            "progress": progress,
                            "department": current_dept,
                        },
                        ensure_ascii=False,
                    ),
                }

            yield {
                "event": "message",
                "data": json_mod.dumps(
                    {
                        "type": "done",
                        "total": total,
                        "by_department": dict(dept_count),
                    },
                    ensure_ascii=False,
                ),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json_mod.dumps(
                    {"type": "error", "data": f"入库异常: {e}"}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator())


# ===== 助手消息落库（供 SSE 结束时调用）=====
def _persist_assistant_message(conversation_id: int, answer: str, sources) -> None:
    """用独立会话保存助手回答。

    为什么不能复用请求作用域的 db: 客户端中途断开时, generator 被取消,
    FastAPI 的 get_db 依赖会在 finally 里把该会话 close 掉, 此时再操作会抛
    "Session is closed"; 且取消期间 await 可能被二次打断。因此这里单开一个
    SessionLocal, 并保持同步调用(单条 INSERT 仅毫秒级, 不构成事件循环瓶颈)。
    """
    # 一个字都没生成(例如 LLM 报错或断开得极早)就不落库, 免得前端多出一条空气泡
    if not answer:
        print(f"[chat][警告] 回答为空, 跳过落库 (conversation_id={conversation_id})")
        return

    from .models import Conversation, ChatMessage

    db = SessionLocal()
    try:
        db.add(
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=json.dumps(sources, ensure_ascii=False) if sources else None,
            )
        )
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = datetime.now()
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[chat][警告] 保存助手消息失败: {type(e).__name__}: {e}")
    finally:
        db.close()


# ===== 流式问答 (SSE) =====
@app.post("/api/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """流式问答接口（支持多轮上下文）

    返回 SSE 事件流，每条事件 data 为 JSON:
      {"type": "token", "data": "..."}      答案增量文本
      {"type": "sources", "data": [...]}    引用来源 (最后一条)
      {"type": "conversation_id", "data": N} 会话ID (第一条消息)
    """
    from .models import Conversation, ChatMessage
    import json as json_mod

    # 1. 处理会话：有ID则用已有，无ID则新建
    conversation = None
    is_new_conversation = False

    if req.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == req.conversation_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="会话不存在")

    if not conversation:
        # 用 LLM 生成会话标题（同步 HTTP 调用，放线程池避免阻塞事件循环）
        title = await asyncio.to_thread(generate_title, req.question)
        conversation = Conversation(title=title)
        db.add(conversation)
        db.flush()  # 获取 ID
        is_new_conversation = True

    # 2. 读取历史消息（用于上下文改写）
    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in history_msgs
    ]

    # 3. 保存用户消息
    user_msg = ChatMessage(
        conversation_id=conversation.id,
        role="user",
        content=req.question,
    )
    db.add(user_msg)
    db.commit()  # 立即提交，确保会话和用户消息持久化（不依赖 SSE 流完成）

    # commit 后 conversation 对象过期，记录 ID 供后续使用
    conversation_id = conversation.id

    # 4. 流式生成回答
    full_answer = ""
    sources_data = None

    async def event_generator():
        nonlocal full_answer, sources_data

        # 新会话先推送会话ID
        if is_new_conversation:
            yield {
                "event": "message",
                "data": json_mod.dumps(
                    {"type": "conversation_id", "data": conversation_id},
                    ensure_ascii=False,
                ),
            }

        try:
            async for payload in stream_answer(req.question, history):
                # 解析 payload 收集完整回答和 sources
                try:
                    evt = json_mod.loads(payload)
                    if evt.get("type") == "token":
                        full_answer += evt["data"]
                    elif evt.get("type") == "sources":
                        sources_data = evt["data"]
                except Exception:
                    pass

                yield {"event": "message", "data": payload}

        except asyncio.CancelledError:
            # 客户端中途断开：已生成的部分回答同样要落库，随后向上抛出以正常关闭流
            print("[chat] 客户端断开连接，保存已生成的部分回答")
            raise
        except Exception as e:
            yield {
                "event": "error",
                "data": json_mod.dumps(
                    {"type": "error", "data": f"服务异常: {e}"}, ensure_ascii=False
                ),
            }
        finally:
            # 无论正常结束、报错还是客户端断开，都用独立会话保存助手回答。
            # 这里刻意用同步调用：generator 被取消时 await 可能再次被打断，
            # 而单独的 INSERT 只有毫秒级，不会明显拖慢事件循环。
            _persist_assistant_message(conversation_id, full_answer, sources_data)

    return EventSourceResponse(event_generator())


# 入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
