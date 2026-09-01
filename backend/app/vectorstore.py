"""Milvus 向量库 + bge-large-zh Embedding 封装

提供两个核心函数:
  - get_embedder(): 返回单例 Embedding 实例
  - get_vectorstore(): 返回 Milvus 向量库实例
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .config import settings


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """返回 Embedding 单例（首次加载会下载模型，约 1.3GB）"""
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.embedding_device,
            "model_kwargs": {"torch_dtype": "float16"},  # FP16 提速约一倍
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
def get_vectorstore() -> Milvus:
    """返回 Milvus 向量库单例"""
    return Milvus(
        embedding_function=get_embedder(),
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        auto_id=True,
        drop_old=False,
    )


def get_retriever(k: int = 5):
    """返回混合检索器（当前为向量检索，可扩展为 EnsembleRetriever）"""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
