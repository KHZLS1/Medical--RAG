"""展示大屏的数据聚合层（`GET /api/stats/overview` 的数据来源）

为什么单独一层而不是塞进路由
----------------------------
语料统计要扫 232 个 JSON（合计约 500MB），一次几十秒。挂在请求里现算，
大屏每次轮询都要等半分钟，且展厅网络一抖就超时。而这份统计**一小时内几乎不变**，
所以这里做了两件事：

  1. 带 TTL 的进程内缓存，并由 lifespan 在启动时后台预热 —— 请求侧永远拿现成的；
  2. 明确区分「语料库规模」与「已入库向量」两个数字。

第 2 点是**诚实性问题，不是技术问题**：
  - `corpus.total_records` 扫的是 `data/by_department/` 磁盘语料，代表"语料库有多大"；
  - `corpus.ingested` 问的是 Milvus 实际条数，代表"真正能被检索到多少"。
入库时可能带了 `--limit`（小批量验证），两者会差很远。把前者当后者展示，
就是在展厅里报一个自己都知道不成立的数字，所以两个都返回、由前端分别标注。
Milvus 连不上时 `ingested` 返回 None，前端显示"向量库未连接"，而不是假装有数据。
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import settings

# 「调用方没预传这个值」与「算出来就是 None（如 Milvus 连不上）」是两回事，
# 用哨兵区分，否则 None 会被当成"没传"而重复去连一次 Milvus。
_UNSET = object()

# 语料扫描结果的缓存时长：语料是离线产物，一小时够新了
CORPUS_TTL = 3600.0
# 评估指标：跑一次 eval 才会变，缓存短一些以便新跑完能及时反映
EVAL_TTL = 600.0

_CACHE: dict[str, tuple[float, object]] = {}
_LOCK = threading.Lock()


def _cached(key: str, ttl: float, producer):
    """极简 TTL 缓存。

    锁覆盖整个 producer 调用（含耗时几十秒的语料扫描）：并发请求宁可排队，
    也不要同时起多个扫描把内存和磁盘 IO 打满。扫描本身跑在线程池里，
    不会卡住事件循环。
    """
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        value = producer()
        _CACHE[key] = (now, value)
        return value


# ============================================================================
# 语料库：扫描清洗后的按科室拆分文件
# ============================================================================
def _scan_corpus() -> dict:
    """扫 `data/by_department/*.json`，按文件统计条数。

    文件名即科室名（`clean_data.py` 按科室拆分落盘，形如 `cleaned_急诊科.json`）。
    注意：清洗产物里**不含** department 字段（清洗时没开 --keep-dept），
    所以科室只能从文件名取；这也意味着本统计反映的是语料库，不是 Milvus 内容。
    """
    base: Path = settings.cleaned_data_dir_resolved
    files = sorted(base.glob("*.json")) if base.is_dir() else []

    by_dept: dict[str, int] = {}
    failed = 0
    for path in files:
        try:
            with path.open(encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            # 单个文件坏了不该让整块大屏没数据，跳过并计数
            failed += 1
            print(f"[stats][警告] 读取语料文件失败 {path.name}: {type(e).__name__}: {e}")
            continue
        if not isinstance(rows, list) or not rows:
            continue
        dept = path.stem
        if dept.startswith("cleaned_"):
            dept = dept[len("cleaned_"):]
        by_dept[dept] = by_dept.get(dept, 0) + len(rows)

    ranked = sorted(by_dept.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(by_dept.values())
    return {
        "total_records": total,
        "departments": len(by_dept),
        "files": len(files),
        "failed_files": failed,
        "ranked": [{"department": d, "count": c} for d, c in ranked],
    }


def corpus_stats() -> dict:
    return _cached("corpus", CORPUS_TTL, _scan_corpus)


def warm_corpus_cache() -> None:
    """启动时预热（由 lifespan 放进线程池调用），让首个请求不必等扫描"""
    try:
        data = corpus_stats()
        print(
            f"[启动] 语料统计就绪: {data['total_records']:,} 条 / "
            f"{data['departments']} 个科室 / {data['files']} 个文件"
        )
    except Exception as e:
        print(f"[启动][警告] 语料统计预热失败: {type(e).__name__}: {e}")


# ============================================================================
# 离线评估指标
# ============================================================================
def _read_latest_evaluation() -> dict | None:
    """挑 `data/eval/` 下 meta.timestamp 最新的一份评估结果。

    按文件内记录的 timestamp 排序，而不是文件 mtime —— 复制、同步、重新 checkout
    都会改 mtime，而 timestamp 是跑评估那一刻写进去的，才是真正的时间口径。
    只认同时含 meta 与 summary 的文件，其它（测试集、调参中间产物）自动跳过。
    """
    base: Path = settings.eval_dir_resolved
    if not base.is_dir():
        return None

    best: tuple[str, str, dict, dict] | None = None
    for path in base.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        meta, summary = payload.get("meta"), payload.get("summary")
        if not isinstance(meta, dict) or not isinstance(summary, dict):
            continue
        ts = str(meta.get("timestamp") or "")
        if best is None or ts > best[0]:
            best = (ts, path.name, meta, summary)

    if best is None:
        return None

    ts, name, meta, summary = best
    return {
        "file": name,
        "timestamp": ts or None,
        "mode": meta.get("mode"),
        "recall_k": meta.get("k"),
        "bm25_weight": meta.get("bm25_weight"),
        "vector_weight": meta.get("vector_weight"),
        "reranker_top_k": meta.get("reranker_top_k"),
        "testset": meta.get("testset"),
        "questions": summary.get("count"),
        "hit_rate": summary.get("avg_hit_rate"),
        "mrr": summary.get("avg_mrr"),
        "coverage": summary.get("avg_coverage"),
        "similarity": summary.get("avg_similarity"),
        "latency_s": summary.get("avg_latency"),
    }


def evaluation_stats() -> dict | None:
    return _cached("evaluation", EVAL_TTL, _read_latest_evaluation)


# ============================================================================
# 检索配置（大屏「检索链路」页展示的就是这几个真实值）
# ============================================================================
def retrieval_stats() -> dict:
    """检索权重与召回参数。

    延迟导入 vectorstore：它会把 langchain_huggingface / pymilvus 一起拉进来，
    路由模块导入期不该付这个成本；更重要的是**取权重必须走 vectorstore**，
    否则 tuned 文件、.env、默认值这三层优先级会被复制成第二份实现，迟早不一致。

    导入失败时退化为只用 settings 的值：大屏会少一个"权重来源"标注，
    但不该因为一个可选依赖缺失就整页没数据 —— 与"各板块独立失败"的约定一致。
    """
    try:
        from .vectorstore import DEFAULT_RECALL_K, _TUNED_WEIGHTS
    except Exception as e:
        print(f"[stats][警告] 读取检索配置失败，退化为 .env 配置值: {type(e).__name__}: {e}")
        return {
            "bm25_weight": settings.bm25_weight,
            "vector_weight": settings.vector_weight,
            "reranker_top_k": settings.reranker_top_k,
            "recall_k": None,
            "weights_source": "config",
        }

    tuned_has_weights = "bm25_weight" in _TUNED_WEIGHTS or "vector_weight" in _TUNED_WEIGHTS
    return {
        "bm25_weight": _TUNED_WEIGHTS.get("bm25_weight", settings.bm25_weight),
        "vector_weight": _TUNED_WEIGHTS.get("vector_weight", settings.vector_weight),
        "reranker_top_k": settings.reranker_top_k,
        "recall_k": DEFAULT_RECALL_K,
        # 让看板能说清这个权重是哪来的：调参产物 / .env 配置
        "weights_source": "tuned" if tuned_has_weights else "config",
    }


# ============================================================================
# Milvus 实际入库条数
# ============================================================================
def ingested_count() -> int | None:
    """返回 Milvus collection 的实体数；连不上返回 None（前端据此显示未连接）"""
    try:
        from .vectorstore import get_milvus_client

        client = get_milvus_client()
        stats = client.get_collection_stats(settings.milvus_collection)
        return int(stats.get("row_count", 0)) if stats else 0
    except Exception as e:
        print(f"[stats][警告] Milvus 统计失败: {type(e).__name__}: {e}")
        return None


# ============================================================================
# 使用量（DB）
# ============================================================================
def usage_stats(db: Session | None) -> dict | None:
    """会话 / 问答量 / 澄清追问次数。DB 不可用返回 None，不阻断其它板块。"""
    from .models import ChatMessage, Conversation

    if db is None:
        return None

    try:
        conversations = db.query(func.count(Conversation.id)).scalar() or 0
        questions = (
            db.query(func.count(ChatMessage.id))
            .filter(ChatMessage.role == "user")
            .scalar()
            or 0
        )
        answers = (
            db.query(func.count(ChatMessage.id))
            .filter(ChatMessage.role == "assistant")
            .scalar()
            or 0
        )
        clarifications = (
            db.query(func.count(ChatMessage.id))
            .filter(ChatMessage.is_clarification.is_(True))
            .scalar()
            or 0
        )
    except Exception as e:
        print(f"[stats][警告] 使用量统计失败: {type(e).__name__}: {e}")
        return None

    return {
        "conversations": conversations,
        "questions": questions,
        "answers": answers,
        # 「证据不足 → 追问」真实发生了多少次：这是产品机制的活证据，值得单独露出来
        "clarifications": clarifications,
    }


# ============================================================================
# 反馈汇总
# ============================================================================
def feedback_stats(db: Session | None) -> dict | None:
    """反馈量 / 差评率 / 带人工纠错条数。与 /api/feedback/stats 同口径。"""
    from .models import Feedback

    if db is None:
        return None

    try:
        by_thumbs = {
            t: c
            for t, c in db.query(Feedback.thumbs, func.count(Feedback.id))
            .group_by(Feedback.thumbs)
            .all()
        }
        with_correction = (
            db.query(func.count(Feedback.id))
            .filter(Feedback.corrected_answer.isnot(None))
            .scalar()
            or 0
        )
    except Exception as e:
        print(f"[stats][警告] 反馈统计失败: {type(e).__name__}: {e}")
        return None

    up = by_thumbs.get("up", 0)
    down = by_thumbs.get("down", 0)
    total = up + down
    return {
        "total": total,
        "up": up,
        "down": down,
        "down_rate": round(down / total, 4) if total else 0.0,
        "with_correction": with_correction,
    }


# ============================================================================
# 总览组装
# ============================================================================
def build_overview(
    db: Session | None,
    top: int = 10,
    *,
    corpus=_UNSET,
    ingested=_UNSET,
    evaluation=_UNSET,
) -> dict:
    """组装大屏总览 payload —— 接口与「兜底快照」脚本共用的**唯一**形状定义。

    为什么不各写一份：快照是给前端在接口挂掉时用的，形状一旦和接口不一致，
    前端会静默渲染出空白面板，而这种故障只在后端挂掉时才出现 —— 最难被发现。
    所以形状只在这里定义一次。

    重量级项（语料扫描、Milvus）可由调用方预先算好传进来：路由在线程池里算，
    避免阻塞事件循环；脚本则什么都不传，让本函数同步算（离线生成快照用）。

    Args:
        top: 「覆盖分布」返回的科室数量，其余合并进 other_records。
    """
    if corpus is _UNSET:
        corpus = corpus_stats()
    if ingested is _UNSET:
        ingested = ingested_count()
    if evaluation is _UNSET:
        evaluation = evaluation_stats()

    top_n = max(1, min(top, 50))
    ranked = corpus["ranked"]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": {
            # 语料库规模（磁盘语料）与已入库向量（Milvus）是两个口径，分开返回
            "total_records": corpus["total_records"],
            "departments": corpus["departments"],
            "ingested": ingested,
            "top_departments": ranked[:top_n],
            "other_records": sum(d["count"] for d in ranked[top_n:]),
            "failed_files": corpus["failed_files"],
        },
        "retrieval": retrieval_stats(),
        "evaluation": evaluation,
        "usage": usage_stats(db),
        "feedback": feedback_stats(db),
    }