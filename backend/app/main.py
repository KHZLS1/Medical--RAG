"""FastAPI 主程序

提供接口:
  GET  /api/health              健康检查
  POST /api/ingest              触发数据入库 (返回任务概览)
  POST /api/chat                流式问答 (SSE)
"""
import asyncio
import json
from contextlib import asynccontextmanager, suppress
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
from .api.feedback import router as feedback_router
from .api.stats import router as stats_router


# ---- 流空闲超时的兜底文案（见 config.sse_idle_timeout_sec）----
# 与 graph.DEGRADED_LLM_ANSWER 同款口径：先说清发生了什么，再指向急诊。
# 分两条是因为"一个 token 都没到"与"吐了一半卡住"对用户是两种不同处境。
IDLE_TIMEOUT_NO_CONTENT = (
    "抱歉，本次回答没有成功生成（服务端响应超时）。请稍后重试；"
    "若情况紧急，请立即拨打 120 或前往急诊。"
)
IDLE_TIMEOUT_PARTIAL = (
    "\n\n⚠️ 服务端响应超时，已停止等待。以上内容可能不完整，建议重试；"
    "若情况紧急，请立即拨打 120 或前往急诊。"
)


async def _watchdog_iter(stream, idle: float):
    """按**空闲**超时逐条产出 `(payload, timed_out)`。

    抽成独立函数只为一个理由：**可离线测试**。`chat()` 里那个生成器绑着数据库
    与图，没法单测；而这段逻辑恰恰最容易写错 —— 写成"整轮计时"就会把慢但在吐字
    的流一起杀掉（前端就是这么错了 120s 那一版）。
    所以这里刻意让 idle 在**每取到一条之后重新计时**。

    `timed_out=True` 是收尾标记，payload 为 None：调用方据此补一段确定性文案。
    `idle <= 0` 表示不设超时（保留原行为）。
    """
    stream_iter = stream.__aiter__()
    while True:
        try:
            if idle and idle > 0:
                payload = await asyncio.wait_for(stream_iter.__anext__(), idle)
            else:
                payload = await stream_iter.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError:
            # 别让底层生成器悬着（wait_for 已经把这次 __anext__ 取消了）。
            with suppress(Exception):
                await stream.aclose()
            yield None, True
            return
        yield payload, False


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

    # 展示大屏的语料统计要扫 232 个 JSON（约 500MB，几十秒）。放后台线程预热，
    # 不阻塞启动；等有人打开大屏时缓存早就好了。
    from .stats import warm_corpus_cache

    asyncio.create_task(asyncio.to_thread(warm_corpus_cache))

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
app.include_router(feedback_router)
app.include_router(stats_router)

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
def _persist_assistant_message(
    conversation_id: int,
    answer: str,
    sources,
    is_clarification: bool = False,
) -> tuple[int, "datetime"] | None:
    """用独立会话保存助手回答，返回 (新消息 id, 落库时间)；没落库则 None。

    为什么要返回 id
    ---------------
    前端要提交反馈只能按 `message_id`（见 `POST /api/feedback`），而流式回答的
    id 是**这一刻**才生成的。原先这里把 id 丢掉，前端于是永远拿不到它，
    `Chat.tsx` 里 `m.messageId && m.messageId > 0` 那道门恒假 ——
    结果是**刚生成的回答根本没有 👍/👎 按钮**，只能刷新页面重新加载会话
    才点得到。反馈闭环在最后一步断了：接口、落库、看板都齐了，用户却点不到。
    所以把这个 id 一路传回 SSE（见 event_generator 里的 `message_id` 事件）。

    为什么要返回 created_at
    ----------------------
    前端要显示"系统回复时间"。流式结束时是客户端自己取 `new Date()`、还是用
    这里落库的 `created_at`，单看一次刷新区别不出来 —— 但两者并不相等：客户端
    时钟可能与服务端有偏差，而且客户端取的是"流结束那一刻"，服务端记的是
    "INSERT 那一刻"。一旦用了客户端时间，**同一个回答的时间会在刷新前后跳变**
    （刷新后从 `GET /messages` 读的是库里的值）。所以统一以库为准，一路带回前端。

    为什么不能复用请求作用域的 db: 客户端中途断开时, generator 被取消,
    FastAPI 的 get_db 依赖会在 finally 里把该会话 close 掉, 此时再操作会抛
    "Session is closed"; 且取消期间 await 可能被二次打断。因此这里单开一个
    SessionLocal, 并保持同步调用(单条 INSERT 仅毫秒级, 不构成事件循环瓶颈)。
    """
    # 一个字都没生成(例如 LLM 报错或断开得极早)就不落库, 免得前端多出一条空气泡
    if not answer:
        print(f"[chat][警告] 回答为空, 跳过落库 (conversation_id={conversation_id})")
        return None

    from .models import Conversation, ChatMessage

    db = SessionLocal()
    try:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            sources=json.dumps(sources, ensure_ascii=False) if sources else None,
            is_clarification=is_clarification,
        )
        db.add(msg)
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = datetime.now()
        db.commit()
        # commit 会让实例属性过期，取值会触发一次 SELECT；必须在 session 关闭前取。
        # created_at 也一并返回：前端要显示"系统回复时间"，用服务端这一刻的值才能
        # 保证「刚生成」与「刷新后从库里读」显示的是同一个时间（客户端本地时钟
        # 与服务端可能有时差，那会让同一个回答的时间在刷新前后跳变）。
        return msg.id, msg.created_at
    except Exception as e:
        db.rollback()
        print(f"[chat][警告] 保存助手消息失败: {type(e).__name__}: {e}")
        return None
    finally:
        db.close()


def _message_id_event(msg_id: int, created_at) -> dict:
    """打包 SSE 的 `message_id` 事件。

    data 是对象而不是裸 id：前端除了要 id 提交反馈，还要显示"系统回复时间"。
    时间用库里那条记录的值，保证「刚生成」与「刷新后重读」显示同一个时间点。
    created_at 理论上不会为 None（列有 default），真为 None 时前端会退回不显示，
    所以这里不做特殊处理，如实透传。
    """
    import json as json_mod

    return {
        "event": "message",
        "data": json_mod.dumps(
            {
                "type": "message_id",
                "data": {
                    "id": msg_id,
                    "created_at": created_at.isoformat() if created_at else None,
                },
            },
            ensure_ascii=False,
        ),
    }


# ===== 流式问答 (SSE) =====
@app.post("/api/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """流式问答接口（支持多轮上下文）

    返回 SSE 事件流，每条事件 data 为 JSON:
      {"type": "token", "data": "..."}      答案增量文本
      {"type": "sources", "data": [...]}    引用来源 (最后一条)
      {"type": "conversation_id", "data": N} 会话ID (第一条消息)
      {"type": "message_id", "data": {"id": N, "created_at": "..."}}
                                            本条助手消息落库后的 id 与落库时间 (最后一条)
                                            —— 前端拿 id 提交 👍/👎，拿 created_at 显示
                                            "系统回复时间"；缺了就只能刷新页面才点得到
                                            /才看得到时间
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

    # 4. 流式生成回答（阶段二：有挂起的澄清中断 → 本条消息作为 resume 值恢复图）
    from .graph import get_graph, thread_config, astream_resume

    config = thread_config(conversation_id)
    resuming = False
    if settings.graph_checkpointer_enabled:
        # 暂停中的图 next 非空（只有 interrupt 会造成这种状态）。
        # ⚠️ 必须走 aget_state：AsyncSqliteSaver 的同步方法在事件循环线程
        # 调用会抛 InvalidStateError，而本端点整体跑在事件循环上。
        snapshot = await (await get_graph()).aget_state(config)
        resuming = bool(snapshot and snapshot.next)

    if resuming:
        stream = astream_resume(req.question, history, config)
    else:
        stream = stream_answer(req.question, history, config=config)

    # 5. 收集完整回答与 sources
    full_answer = ""
    sources_data = None

    async def event_generator():
        nonlocal full_answer, sources_data

        # 落库是否已在正常路径完成 —— 正常路径要先落库、再用 message_id 推事件，
        # 所以不能把落库一律塞在 finally 里（finally 里 yield 在被取消时会炸）。
        # finally 只兜「异常 / 客户端断开」这两条没走到落库的路。
        saved = False

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
            # 流空闲看门狗（理由见 config.sse_idle_timeout_sec 与 _watchdog_iter）：
            # `llm_timeout_sec` 对流式响应不是墙钟上限，卡住的流能拖到几十分钟。
            # 前端 `Chat.tsx` 另有一层 105s 的对称兜底（阈值更大，让后端先收尾）。
            idle = settings.sse_idle_timeout_sec
            async for payload, timed_out in _watchdog_iter(stream, idle):
                if timed_out:
                    print(f"[chat] 流空闲超过 {idle:.0f}s，主动收尾")
                    full_answer = (
                        full_answer + IDLE_TIMEOUT_PARTIAL
                        if full_answer else IDLE_TIMEOUT_NO_CONTENT
                    )
                    # 复用 correction 出口（前端整段替换、落库用修正后的文本），
                    # 这是已经打通的唯一"覆盖而不是追加"的通道。
                    yield {
                        "event": "message",
                        "data": json_mod.dumps(
                            {"type": "correction",
                             "data": {"answer": full_answer,
                                      "invalid_citations": [],
                                      "verdict": "stream_idle_timeout"}},
                            ensure_ascii=False,
                        ),
                    }
                    break

                # 解析 payload 收集完整回答和 sources
                try:
                    evt = json_mod.loads(payload)
                    if evt.get("type") == "token":
                        full_answer += evt["data"]
                    elif evt.get("type") == "correction":
                        # 引用编号被程序剥除 / 追加了忠实性提示（阶段四）→ 落库必须用
                        # 修正后的文本。否则页面显示修正版、库里存原始版，刷新后假编号
                        # 复活（页面与库不一致是最糟的形态，排查时会被当成"随机复现"）。
                        full_answer = (evt.get("data") or {}).get("answer") or full_answer
                    elif evt.get("type") == "sources":
                        sources_data = evt["data"]
                    elif evt.get("type") == "clarification_request":
                        # 中断不是错误：把追问话术作为助手消息落库（刷新页面后
                        # 仍可见，后续回复才有上下文）；message_id 一并回传。
                        msg_text = (evt.get("data") or {}).get("message", "")
                        saved_msg = _persist_assistant_message(
                            conversation_id, msg_text, [], is_clarification=True
                        )
                        saved = True   # 落库在此完成，别让 finally 再补一条
                        if saved_msg:
                            yield _message_id_event(*saved_msg)
                except Exception:
                    pass

                yield {"event": "message", "data": payload}

            # 正常收尾：先落库拿到 message_id，再把它推给前端。
            # 顺序不能反 —— 前端拿到 id 就会把它挂到最后一条助手消息上，
            # 用来渲染 👍/👎。推早了 id 还没生成，推晚了自己这条流已经关了。
            saved_msg = _persist_assistant_message(conversation_id, full_answer, sources_data)
            saved = True
            if saved_msg:
                yield _message_id_event(*saved_msg)

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
            # 兜底：报错或取消时还没落库，用独立会话补一条，避免这段回答凭空消失。
            # 这里刻意用同步调用：generator 被取消时 await 可能再次被打断，
            # 而单独的 INSERT 只有毫秒级，不会明显拖慢事件循环。
            if not saved:
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
