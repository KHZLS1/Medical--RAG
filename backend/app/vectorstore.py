"""Milvus 检索层：服务端 BM25 + 稠密向量 + 混合融合

collection schema（由 scripts/ingest.py 建立）:
  id       INT64   auto_id
  embedding FLOAT_VECTOR(dim=1024)   bge-large-zh-v1.5 稠密向量
  content  VARCHAR(65535)            enable_analyzer + chinese analyzer
  sparse   SPARSE_FLOAT_VECTOR       由 BM25 Function 服务端自动生成，客户端不写
  metadata JSON                      department / title / source

检索统一走 pymilvus 原生 hybrid_search：
  - 稀疏：query 原文直接传给 sparse 字段，Milvus 服务端按 BM25 打分
  - 稠密：query 由 bge 编码后检索 embedding 字段（metric=IP）
  - 融合：WeightedRanker(BM25 权重, 向量权重) 分数归一化；也可切 RRFRanker

为什么全部绕开 langchain_milvus 的 Milvus 封装：
  该封装默认按 text / vector / pk 命名字段，检索时还会用 output_fields=["*"]
  把不可回传的 sparse 字段一起取回，与本 collection 的 schema 不兼容；
  写入同理。检索与写入现在都直接走 pymilvus
  （检索 hybrid_search / 写入 add_documents）。
"""
import asyncio
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from .config import settings

# ===== 字段名（必须与 scripts/ingest.py 建库时一致） =====
TEXT_FIELD = "content"
VECTOR_FIELD = "embedding"
SPARSE_FIELD = "sparse"
METADATA_FIELD = "metadata"
OUTPUT_FIELDS = [TEXT_FIELD, METADATA_FIELD]

# ===== 召回条数（两路各召回 top_k，融合后再交给 Reranker 精排） =====
DEFAULT_RECALL_K = 20

# ===== 混合检索权重（可用 scripts/eval_rag.py tune 网格搜索调优） =====
BM25_WEIGHT = 0.3       # 关键词精确匹配（服务端 BM25）
VECTOR_WEIGHT = 0.7     # 语义相似度（bge 稠密向量）


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """返回 Embedding 单例（首次加载会下载模型，约 1.3GB）"""
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.embedding_device,
            "model_kwargs": {"dtype": "float16"},  # FP16 提速约一倍
        },
        encode_kwargs={
            "normalize_embeddings": True,   # bge 推荐归一化
            "batch_size": 128,
        },
    )
    # 截断长度 512→320：问答文本关键语义在前部，检索质量几乎无损
    emb._client.max_seq_length = 320
    return emb


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """Milvus 客户端单例（首次调用时把 collection 载入内存）"""
    client = MilvusClient(uri=settings.milvus_uri)
    client.load_collection(settings.milvus_collection)
    return client


def _hits_to_documents(hits: list[dict], mode: str) -> list[Document]:
    """把 Milvus 检索结果转成 LangChain Document（保留原始 metadata）"""
    documents = []
    for hit in hits:
        entity = hit.get("entity") or {}
        metadata = dict(entity.get(METADATA_FIELD) or {})
        metadata["retrieval_score"] = float(hit.get("distance", 0.0))
        metadata["retrieval_mode"] = mode
        documents.append(
            Document(page_content=entity.get(TEXT_FIELD, ""), metadata=metadata)
        )
    return documents


def hybrid_search(
    query: str,
    k: int = DEFAULT_RECALL_K,
    mode: str = "hybrid",
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    ranker_type: str = "weighted",
) -> list[Document]:
    """混合检索：服务端 BM25(sparse) + 稠密向量(dense)

    Args:
        query: 检索 query（建议用改写后的 query）
        k: 每路召回条数
        mode: "hybrid" (默认) / "vector" (仅语义) / "bm25" (仅关键词)
        bm25_weight: 混合模式下 BM25 权重
        vector_weight: 混合模式下向量权重
        ranker_type: "weighted" (默认，归一化加权) 或 "rrf" (倒数排名融合)

    Returns:
        按融合分数降序的 Document 列表
    """
    if not query or not query.strip():
        return []

    client = get_milvus_client()
    collection = settings.milvus_collection

    # ---- 单路检索：不需要 ranker ----
    if mode == "bm25":
        results = client.search(
            collection_name=collection,
            data=[query],                       # 原文交给服务端做 BM25 分词打分
            anns_field=SPARSE_FIELD,
            search_params={"metric_type": "BM25"},
            limit=k,
            output_fields=OUTPUT_FIELDS,
        )
        return _hits_to_documents(results[0] if results else [], mode)

    if mode == "vector":
        results = client.search(
            collection_name=collection,
            data=[get_embedder().embed_query(query)],
            anns_field=VECTOR_FIELD,
            search_params={"metric_type": "IP"},
            limit=k,
            output_fields=OUTPUT_FIELDS,
        )
        return _hits_to_documents(results[0] if results else [], mode)

    if mode != "hybrid":
        raise ValueError(f"未知检索模式: {mode}（可选 hybrid / vector / bm25）")

    # ---- 混合检索：两路召回 + 融合排序 ----
    # 注意：ranker 的权重顺序必须与 reqs 的顺序一致
    reqs = [
        AnnSearchRequest(
            data=[query],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=k,
        ),
        AnnSearchRequest(
            data=[get_embedder().embed_query(query)],
            anns_field=VECTOR_FIELD,
            param={"metric_type": "IP"},
            limit=k,
        ),
    ]

    if ranker_type == "rrf":
        ranker = RRFRanker(60)
    else:
        ranker = WeightedRanker(bm25_weight, vector_weight, norm_score=True)

    # 融合后多留一些候选给 Reranker 精排（两路检索各召回 k 条，去重前最多 2k）
    fusion_limit = min(k * 2, 50)

    results = client.hybrid_search(
        collection_name=collection,
        reqs=reqs,
        ranker=ranker,
        limit=fusion_limit,
        output_fields=OUTPUT_FIELDS,
    )
    return _hits_to_documents(results[0] if results else [], mode)


class MilvusHybridRetriever(BaseRetriever):
    """混合检索器：把原生 hybrid_search 包装成 LangChain Retriever

    用法与普通 retriever 一致：retriever.invoke(query) -> list[Document]
    上层（rag_chain.retrieve）随后用 Reranker 对这批召回结果做精排。
    """

    k: int = DEFAULT_RECALL_K
    mode: str = "hybrid"
    bm25_weight: float = BM25_WEIGHT
    vector_weight: float = VECTOR_WEIGHT

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return hybrid_search(
            query,
            k=self.k,
            mode=self.mode,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # 检索是 CPU/GPU 密集的同步调用，放进线程池避免阻塞事件循环
        return await asyncio.to_thread(self._get_relevant_documents, query)


@lru_cache(maxsize=1)
def get_retriever(k: int = DEFAULT_RECALL_K) -> MilvusHybridRetriever:
    """返回混合检索器（服务端 BM25 0.3 + 稠密向量 0.7）

    BM25 擅长精确匹配药名 / 疾病名 / 检查指标等关键词，
    向量检索擅长语义理解和口语化问题，两者互补。
    """
    return MilvusHybridRetriever(k=k)


# ============================================================================
# 写入路径：pymilvus 原生 insert（与 ingest.py 建库时的 schema 完全一致）
#   - 客户端只需提供 content / embedding / metadata
#   - sparse 由 Milvus 服务端 BM25 Function 依据 content 自动生成，客户端不写
#   不再使用 langchain_milvus 封装：它的默认字段名（text/vector/pk）与本
#   collection 的 schema（content/embedding/id）不匹配，写入必然失败。
# ============================================================================
def _clean_metadata(metadata: dict | None) -> dict:
    """把 metadata 清洗成 Milvus JSON 字段可接受的类型"""
    from datetime import date, datetime

    cleaned = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
        elif isinstance(value, (datetime, date)):
            cleaned[str(key)] = value.isoformat()
        else:
            cleaned[str(key)] = str(value)   # Path 等其它类型统一转字符串
    return cleaned


def add_documents(documents: list[Document], batch_size: int = 200) -> int:
    """批量写入文档到 Milvus（本地编码 embedding + 原生 insert）

    Args:
        documents: 待写入的 LangChain Document 列表
        batch_size: 每批条数（编码 + 写入）

    Returns:
        实际写入条数
    """
    if not documents:
        return 0

    client = get_milvus_client()
    embedder = get_embedder()
    collection = settings.milvus_collection
    total = 0

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        vectors = embedder.embed_documents([d.page_content for d in batch])
        rows = [
            {
                TEXT_FIELD: d.page_content,
                VECTOR_FIELD: vector,
                METADATA_FIELD: _clean_metadata(d.metadata),
            }
            for d, vector in zip(batch, vectors)
        ]
        client.insert(collection_name=collection, data=rows)
        total += len(rows)

    # 落盘，保证写入后立即可被检索到
    client.flush(collection)
    return total


def _escape_filter_value(value: str) -> str:
    """转义 Milvus JSON 过滤表达式中的字符串值"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def delete_documents_by_filename(filename: str, flush: bool = True) -> int:
    """按文件名删除该文档已入库的所有向量块

    上传文档的每个 chunk 都带 metadata["filename"]（见 api/documents.py），
    删除文档时一并清理向量，避免回答继续引用已删除的资料。

    Returns:
        删除的实体数量（Milvus 返回的 delete_count）
    """
    if not filename:
        return 0

    client = get_milvus_client()
    collection = settings.milvus_collection
    expr = f'metadata["filename"] == "{_escape_filter_value(filename)}"'
    result = client.delete(collection_name=collection, filter=expr)
    if flush:
        client.flush(collection)
    if isinstance(result, dict):
        return int(result.get("delete_count", 0) or 0)
    return 0
