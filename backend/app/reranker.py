"""检索后处理：Reranker 重排序 + 去重 + 元数据过滤

Reranker 与向量检索的区别：
  - 向量检索（Embedding）：把 query 和 doc 分别编码成向量算余弦相似度，快但粗
  - Reranker（Cross-Encoder）：把 query 和 doc 拼一起送进模型做精细打分，慢但准

推荐流程：向量检索召回 Top-20 → Reranker 精排 → 取 Top-5
"""
import logging
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """返回 Reranker 单例（首次加载约 2GB 模型）

    ⚠️ 下面两个参数是踩过坑才这么写的，别改回看起来更"标准"的写法：

    1. model_name_or_path 传**本地快照绝对路径**，而不是 "BAAI/bge-reranker-v2-m3"。
       FlagReranker.__init__ 虽然声明了 **kwargs，但构造 tokenizer/model 时只下传
       trust_remote_code / cache_dir，其余 kwargs 一律静默丢弃 —— 所以
       local_files_only=True 传进去等于没传，进程仍会向 huggingface.co 发 HEAD
       校验版本；国内直连会被 SSL 阻断（SSLEOFError: UNEXPECTED_EOF_WHILE_READING）。
       改传绝对路径后 from_pretrained 走 os.path.isdir 分支，零网络请求。

    2. 参数名是 devices（复数），不是 device。写成 device= 同样会被 kwargs 吞掉，
       静默失效、模型悄悄退回默认设备。
    """
    model_path = settings.reranker_model_path
    device = settings.reranker_device

    if not Path(model_path).is_dir():
        logger.warning(
            "未找到 reranker 本地快照（%s），将回退在线加载；"
            "网络受限时会以 SSLEOFError 失败", model_path
        )

    return FlagReranker(
        model_path,
        use_fp16=device.startswith("cuda"),   # fp16 只在 CUDA 上才有意义
        devices=device,
    )


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int | None = None,
    department_filter: str | None = None,
) -> list[Document]:
    """对检索结果重排序 + 去重 + 过滤

    Args:
        query: 用户问题
        docs: 检索召回的文档列表
        top_k: 最终保留的文档数；None 时取配置 RERANKER_TOP_K（默认 5）
        department_filter: 按科室过滤（如 "儿科"），None 表示不过滤
    """
    if not docs:
        return []

    if top_k is None:
        top_k = settings.reranker_top_k

    # 1. 元数据过滤
    if department_filter:
        docs = [d for d in docs if d.metadata.get("department") == department_filter]

    # 2. 去重（按 page_content 前 100 字符判重）
    seen = set()
    unique_docs = []
    for d in docs:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    docs = unique_docs

    # 3. Reranker 精排
    reranker = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.compute_score(pairs, normalize=True)

    # 单条结果时 compute_score 返回 float，统一为 list
    if isinstance(scores, (int, float)):
        scores = [scores]

    # 按分数降序排序
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4. 取 Top-K，并把分数写回 metadata（供 graph.node_rerank 的相关性闸门使用）
    #    compute_score(normalize=True) 的输出已过 sigmoid，落在 (0,1)，
    #    可直接当"相关概率"理解。
    #    注意：_format_docs_with_sources 只取 department/title/source 三个键，
    #    rerank_score 不会泄漏到给前端的 sources 载荷里。
    result = []
    for score, doc in scored[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        result.append(doc)
    return result
