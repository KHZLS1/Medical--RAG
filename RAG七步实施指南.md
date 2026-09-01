# 医疗 RAG 检索增强 — 7 步完整实施指南

> 对照项目 `backend/app/` 现有代码，每一步给出：目标、现状、要做什么、代码示例、验证方法。

---

## 步骤 ① 文档加载与解析模块（数据接入）

### 目标
读取不同格式的原始文档，统一转成 LangChain `Document` 对象（`page_content` + `metadata`）。

### 现状
- `data_loader.py` 已实现 CSV(GBK) 加载，支持 6 个科室 79 万条问答
- 每行 CSV → 一个 Document，格式：`【科室】... 【问题】... 【解答】...`
- metadata 已保存 `department`、`title`、`source`

### 要做什么

**基础（已有，跑通即可）：** CSV 加载已完成，不用改。

**进阶（后续接入多格式文档时加）：** 在 `data_loader.py` 中新增多格式 Loader：

```python
# data_loader.py 新增

from langchain_community.document_loaders import (
    PyMuPDFLoader,      # PDF
    Docx2txtLoader,     # Word
    TextLoader,         # TXT
    UnstructuredMarkdownLoader,  # Markdown
)

def load_file(file_path: str) -> list[Document]:
    """根据扩展名自动选择 Loader"""
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".pdf":  PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader,
        ".md":   UnstructuredMarkdownLoader,
        ".csv":  None,  # CSV 走已有的 iter_medical_documents
    }

    if ext not in loaders:
        raise ValueError(f"不支持的格式: {ext}")

    if ext == ".csv":
        return load_medical_documents()

    loader = loaders[ext](str(path))
    return loader.load()
```

### 依赖
```bash
pip install pymupdf docx2txt unstructured
```

### 验证
```python
from app.data_loader import load_medical_documents
docs = load_medical_documents(limit_per_file=10)
print(f"加载 {len(docs)} 条")
print(docs[0].page_content[:200])
print(docs[0].metadata)
```

---

## 步骤 ② 文本切分模块（Chunk 分块）

### 目标
把长文档切成大小合适的文本块，兼顾「检索精度」和「上下文完整性」。

### 现状
- 问答数据每条本身就是短的 Q&A，**当前没有切分，一行一个 Document**
- 对问答数据集来说这是合理的，不需要改

### 要做什么

**基础（问答数据不用切分）：** 跳过。

**进阶（接入长文档时必须加）：** 在 `data_loader.py` 中新增切分逻辑：

```python
# data_loader.py 新增

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """将长文档切分为 chunk

    Args:
        chunk_size: 每个块最大字符数（中文建议 300-800）
        chunk_overlap: 相邻块重叠字符数（建议 chunk_size 的 10%-20%）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 中文优先按段落→句子切
    )
    return splitter.split_documents(documents)
```

### 切分参数建议

| 文档类型 | chunk_size | chunk_overlap | 说明 |
|---|---|---|---|
| 问答数据（当前） | 不切分 | - | 每条 Q&A 本身就是一个 chunk |
| 医学指南/教科书 | 500 | 50 | 按段落+句子切，保留完整语义 |
| 病历报告 | 300 | 30 | 较短，避免跨病例混淆 |

### 验证
```python
from app.data_loader import load_medical_documents, split_documents
docs = load_medical_documents(limit_per_file=10)
chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
print(f"原始 {len(docs)} 条 → 切分后 {len(chunks)} 条")
```

---

## 步骤 ③ 向量化与向量库模块（知识库构建）

### 目标
把文本块通过 Embedding 模型转成向量，存入向量数据库，构建可检索的知识库。

### 现状
- `vectorstore.py` 已封装 bge-large-zh-v1.5 + Milvus
- `get_embedder()` 返回单例 Embedding（已设置归一化）
- `get_vectorstore()` 返回 Milvus 单例
- `scripts/ingest.py` 已有批量入库脚本

### 要做什么

**基础（已有，跑通即可）：** 不用改代码，直接跑入库脚本。

**进阶（优化 metadata 索引）：** 让 Milvus 对 `department` 字段建索引，支持过滤检索：

```python
# vectorstore.py 修改 get_vectorstore，增加索引字段

@lru_cache(maxsize=1)
def get_vectorstore() -> Milvus:
    return Milvus(
        embedding_function=get_embedder(),
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        auto_id=True,
        drop_empty=True,
        # 新增：声明 metadata 字段，便于过滤检索
        # Milvus 2.4+ 支持分区键，可按 department 做分区
    )
```

### 执行入库
```bash
# 在 backend/ 目录下
# 先跑小批量验证
python scripts/ingest.py --limit 500

# 确认无误后跑全量（耗时较长）
python scripts/ingest.py
```

### 验证
```python
from app.vectorstore import get_vectorstore
vs = get_vectorstore()
results = vs.similarity_search("高血压能吃什么", k=3)
for r in results:
    print(r.page_content[:100])
    print(r.metadata)
    print("---")
```

---

## 步骤 ④ 检索模块（用户提问阶段）

### 目标
用户提问后，从向量库中召回最相关的 Top-K 文档。

### 现状
- `get_retriever(k=5)` 返回纯向量检索器
- `rag_chain.py` 中调用 `retriever.invoke(question)` 获取 5 条相关文档

### 要做什么

**基础（已有）：** 纯向量检索能跑通。

**进阶（混合检索，效果提升明显）：** 向量检索 + BM25 关键词检索联合召回：

```python
# vectorstore.py 新增

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def get_hybrid_retriever(k: int = 5):
    """混合检索：向量检索 + BM25 关键词检索

    向量检索擅长语义匹配（"高血压" ≈ "血压偏高"）
    BM25 擅长关键词匹配（专有名词、药名精确命中）
    两者互补，召回率更高
    """
    # 1. 向量检索
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    # 2. BM25 检索（需要先把所有文档加载到内存，适合中小规模）
    # 大规模数据可用 Elasticsearch 替代
    from app.data_loader import load_medical_documents
    all_docs = load_medical_documents(limit_per_file=500)  # 根据内存调整
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k * 2

    # 3. 融合：权重可调，向量 0.5 + BM25 0.5
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble
```

### 依赖
```bash
pip install rank-bm25
```

### 验证
```python
from app.vectorstore import get_retriever
retriever = get_retriever(k=5)
docs = retriever.invoke("高血压患者能吃党参吗")
for d in docs:
    print(d.metadata.get("department"), d.page_content[:80])
```

---

## 步骤 ⑤ Prompt 组装与大模型生成模块

### 目标
把检索到的文档拼接成 context，配合用户问题组装 Prompt，调用大模型生成回答。

### 现状
- `rag_chain.py` 已有完整的医疗 Prompt（7 条规则：仅基于资料、拒答、结构化、禁处方、急症提示、引用编号、免责声明）
- `_format_docs_with_sources()` 已实现带编号的 context 格式化
- `stream_answer()` 已实现 SSE 流式输出 + 引用来源返回
- `llm.py` 已封装 DeepSeek 客户端

### 要做什么

**基础（已有，跑通即可）：** Prompt 写得很好，不用改。

**进阶（多轮对话上下文）：** 当前只支持单轮问答，如需多轮对话，增加历史消息拼接：

```python
# rag_chain.py 新增多轮对话支持

MULTI_TURN_PROMPT = """你是一名严谨、专业的医学助手。请结合对话历史，仅基于【医学资料】回答用户的医疗问题。

【对话历史】
{history}

【医学资料】
{context}

【回答规则】
1. 只能基于上述资料作答，不得使用资料外的知识，更不能编造。
2. 如果资料不足以回答问题，请直接说明"现有医学资料无法回答该问题，建议咨询执业医师"。
3. 回答应结构化、专业且通俗。
4. 严禁给出具体药物剂量或处方建议；如涉及用药，提示"请遵医嘱"。
5. 若问题涉及急症，优先提示"请立即拨打120或前往急诊"。
6. 回答末尾用 [1] [2] 等标注引用的资料编号。
7. 末尾必须附加免责声明："⚠️ 本回答仅基于公开医学资料供参考，不能替代执业医师诊断。"

【用户问题】
{question}
"""

def build_multi_turn_chain(history: list[dict]):
    """构建多轮对话 RAG 链

    Args:
        history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    retriever = get_retriever(k=5)
    llm = get_llm()

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        context, sources = _format_docs_with_sources(docs)
        # 拼接历史对话
        history_text = "\n".join(
            f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
            for m in history[-6:]  # 最近 3 轮
        )
        return {
            "context": context,
            "question": question,
            "history": history_text,
            "_sources": sources,
        }

    prompt = ChatPromptTemplate.from_template(MULTI_TURN_PROMPT)
    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### 验证
```python
# 单轮
from app.rag_chain import stream_answer
import asyncio
async def test():
    async for chunk in stream_answer("高血压患者能吃党参吗"):
        print(chunk)
asyncio.run(test())
```

---

## 步骤 ⑥ 检索后处理（Reranker 重排序 + 过滤）

### 目标
检索召回的 Top-K 文档经过重排序和过滤，只把最相关的喂给 LLM，提升回答质量、降低 token 消耗。

### 现状
- **当前没有这一步**，检索结果直接丢给 LLM
- 这是效果提升投入产出比最高的一步

### 要做什么

**新建 `reranker.py`：**

```python
"""检索后处理：Reranker 重排序 + 去重 + 元数据过滤

Reranker 与向量检索的区别：
  - 向量检索（Embedding）：把 query 和 doc 分别编码成向量算余弦相似度，快但粗
  - Reranker（Cross-Encoder）：把 query 和 doc 拼一起送进模型做精细打分，慢但准

推荐流程：向量检索召回 Top-20 → Reranker 精排 → 取 Top-5
"""
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker


@lru_cache(maxsize=1)
def get_reranker():
    """返回 Reranker 单例（首次加载约 2GB 模型）"""
    return FlagReranker(
        "BAAI/bge-reranker-v2-m3",
        use_fp16=True,  # 半精度加速
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

    # 按分数降序排序
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4. 取 Top-K
    return [doc for _, doc in scored[:top_k]]
```

**修改 `rag_chain.py` 调用 Reranker：**

```python
# rag_chain.py 修改 retrieve_and_format

from .reranker import rerank_documents

def build_rag_chain():
    retriever = get_retriever(k=20)  # 召回扩大到 20
    llm = get_llm()

    def retrieve_and_format(question: str) -> dict:
        # 先粗召 20 条
        docs = retriever.invoke(question)
        # 再精排取 5 条
        docs = rerank_documents(question, docs, top_k=5)
        context, sources = _format_docs_with_sources(docs)
        return {"context": context, "question": question, "_sources": sources}

    prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT)
    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### 依赖
```bash
pip install FlagEmbedding
```

### 验证
```python
from app.vectorstore import get_retriever
from app.reranker import rerank_documents

retriever = get_retriever(k=20)
docs = retriever.invoke("高血压能吃党参吗")
print(f"召回 {len(docs)} 条，rerank 前：")
for d in docs[:5]:
    print(f"  - {d.page_content[:60]}")

reranked = rerank_documents("高血压能吃党参吗", docs, top_k=5)
print(f"\nrerank 后：")
for d in reranked:
    print(f"  - {d.page_content[:60]}")
```

---

## 步骤 ⑦ 查询改写（Query Rewriting）

### 目标
用户提问往往口语化、信息不全，先用 LLM 改写 query 再去检索，提升召回率。

### 三种改写策略

| 策略 | 原理 | 适合场景 | 复杂度 |
|---|---|---|---|
| **Query Expansion** | LLM 把口语化问题改写成关键词增强版 | 口语化提问 | 低 |
| **Multi-Query** | LLM 生成 3-5 个变体 query，分别检索后合并 | 召回率不够 | 中 |
| **HyDE** | LLM 先生成假设答案，用假设答案去检索 | query 和 doc 表述差异大 | 高 |

### 要做什么

**新建 `query_rewriter.py`：**

```python
"""查询改写模块

在检索前对用户 query 做增强，提升召回率。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .llm import get_llm


REWRITE_PROMPT = """你是一个医学搜索查询改写助手。
请将用户的口语化问题改写为更适合向量检索的查询，要求：
1. 提取核心医学关键词
2. 补充相关医学术语
3. 去除无意义的口语词汇
4. 保持原意不变
5. 只输出改写后的查询，不要解释

用户问题：{question}
改写后的查询："""


def rewrite_query(question: str) -> str:
    """单次改写：口语化 → 关键词增强版"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def multi_query(question: str, n: int = 3) -> list[str]:
    """多路改写：生成 n 个变体 query，用于多路召回

    示例：
      原始："高血压患者能吃党参吗"
      变体1："党参 高血压 禁忌 相互作用"
      变体2："高血压中药调理 党参安全性"
      变体3："高血压患者 人参 党参 用药注意"
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是医学搜索助手。请为以下问题生成 {n} 个不同角度的检索查询变体，每行一个，不要编号。"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "n": n})
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    queries.insert(0, question)  # 原始 query 也保留
    return queries[:n + 1]
```

**修改 `rag_chain.py` 集成查询改写：**

```python
# rag_chain.py 修改 build_rag_chain，加入查询改写

from .query_rewriter import rewrite_query
from .reranker import rerank_documents

def build_rag_chain():
    retriever = get_retriever(k=20)
    llm = get_llm()

    def retrieve_and_format(question: str) -> dict:
        # ⑦ 查询改写
        enhanced_query = rewrite_query(question)

        # ④ 检索（用改写后的 query）
        docs = retriever.invoke(enhanced_query)

        # ⑥ Reranker 重排序
        docs = rerank_documents(question, docs, top_k=5)  # 注意：reranker 用原始 question

        # ⑤ 组装 context
        context, sources = _format_docs_with_sources(docs)
        return {"context": context, "question": question, "_sources": sources}

    prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT)
    chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### 验证
```python
from app.query_rewriter import rewrite_query, multi_query

print(rewrite_query("高血压患者能吃党参吗"))
# 输出: "党参 高血压 用药禁忌 相互作用 安全性"

print(multi_query("高血压患者能吃党参吗", n=3))
# 输出: ["高血压患者能吃党参吗", "党参 高血压 禁忌...", "高血压中药调理..."]
```

---

## 完整流程串联

最终 `rag_chain.py` 的 `build_rag_chain()` 包含全部 7 步：

```
用户提问
  │
  ├─⑦ 查询改写（query_rewriter.py）
  │    "高血压患者能吃党参吗" → "党参 高血压 用药禁忌 相互作用"
  │
  ├─④ 检索（vectorstore.py）
  │    用改写后的 query 向量检索 → Top-20
  │
  ├─⑥ 检索后处理（reranker.py）
  │    去重 → Reranker 精排 → 按 department 过滤 → Top-5
  │
  ├─⑤ Prompt 组装（rag_chain.py）
  │    带编号 context + 医疗 Prompt 规则
  │
  └─⑤ DeepSeek 流式生成 + 引用溯源
       SSE 输出回答 + sources 列表
```

---

## 实施优先级

| 优先级 | 步骤 | 投入 | 效果 | 说明 |
|---|---|---|---|---|
| P0 必做 | ①②③④⑤ | 低 | 能跑通 | 项目已有代码，跑通即可 |
| P1 强烈推荐 | ⑥ Reranker | 中 | ⭐⭐⭐ | 投入产出比最高，回答质量明显提升 |
| P2 推荐 | ⑦ 查询改写 | 中 | ⭐⭐ | 召回率提升，口语化问题改善明显 |
| P3 可选 | ④ 混合检索 | 高 | ⭐⭐ | 需要额外维护 BM25 索引，大规模数据才值得 |

**建议路径：先跑通 P0 → 验证能回答 → 加 P1 Reranker → 对比效果 → 再考虑 P2/P3。**

---

## 依赖安装汇总

```bash
# 基础（P0，项目已有）
pip install -r backend/requirements.txt

# Reranker（P1）
pip install FlagEmbedding

# 混合检索（P3）
pip install rank-bm25

# 多格式文档加载（可选）
pip install pymupdf docx2txt unstructured
```
