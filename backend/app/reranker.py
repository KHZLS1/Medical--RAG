"""检索后处理：Reranker 重排序 + 去重 + 元数据过滤

Reranker 与向量检索的区别：
  - 向量检索（Embedding）：把 query 和 doc 分别编码成向量算余弦相似度，快但粗
  - Reranker（Cross-Encoder）：把 query 和 doc 拼一起送进模型做精细打分，慢但准

推荐流程：向量检索召回 Top-20 → Reranker 精排 → 取 Top-5
"""
from functools import lru_cache

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker


@lru_cache(maxsize=1)
def get_reranker():
    """返回 Reranker 单例（首次加载约 2GB 模型）"""
    return FlagReranker(
        "BAAI/bge-reranker-v2-m3",
        use_fp16=True,
    )


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 5,
    department_filter: str | None = None,
) -> list[Document]:
    """对检索结果重排序 + 去重 + 过滤

    Args:
        query: 用户问题
        docs: 检索召回的文档列表
        top_k: 最终保留的文档数
        department_filter: 按科室过滤（如 "儿科"），None 表示不过滤
    """
    if not docs:
        return []

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

    # 4. 取 Top-K
    return [doc for _, doc in scored[:top_k]]
