"""FastAPI 主程序

提供接口:
  GET  /api/health              健康检查
  POST /api/ingest              触发数据入库 (返回任务概览)
  POST /api/chat                流式问答 (SSE)
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.database import Base, engine
from .config import settings
from .rag_chain import stream_answer
from .vectorstore import get_vectorstore
from .api.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    #应用启动时自动建表
    Base.metadata.create_all(bind=engine)
    print(f"[启动] 数据库表已就绪: {settings.database_url}")

    """应用启动/关闭钩子"""
    print(f"[启动] 数据目录: {settings.data_dir_resolved}")
    print(f"[启动] Milvus: {settings.milvus_uri}, collection={settings.milvus_collection}")
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

# ===== 请求/响应 模型 =====
class ChatRequest(BaseModel):
    question: str


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
        docs = load_medical_documents(limit_per_file=req.limit_per_file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not docs:
        raise HTTPException(status_code=400, detail="未加载到任何文档，请检查数据目录")

    vs = get_vectorstore()
    vs.add_documents(docs)

    # 按科室统计
    from collections import Counter
    dept_count = Counter(d.metadata.get("department", "未知") for d in docs)

    return {
        "ingested": len(docs),
        "by_department": dict(dept_count),
        "collection": settings.milvus_collection,
    }


# ===== 流式问答 (SSE) =====
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """流式问答接口

    返回 SSE 事件流，每条事件 data 为 JSON:
      {"type": "token", "data": "..."}      答案增量文本
      {"type": "sources", "data": [...]}    引用来源 (最后一条)
    """
    async def event_generator():
        try:
            async for payload in stream_answer(req.question):
                yield {"event": "message", "data": payload}
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"type": "error", "data": f"服务异常: {e}"}, ensure_ascii=False
                ),
            }

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
