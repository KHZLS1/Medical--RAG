# 医疗智能问答系统 (Medical RAG)

基于 **React + FastAPI + Milvus + DeepSeek** 的医疗问答系统，知识库来自开源中文医疗问答数据集（清洗后 60.5 万条）。

**技术栈**

| 层 | 选型 |
|---|---|
| 前端 | React 18 + Vite + TypeScript（SSE 流式打字机） |
| 后端 | FastAPI + SQLAlchemy |
| 向量库 | Milvus 2.5+（服务端 BM25 Function + 稠密向量混合检索） |
| Embedding | BAAI/bge-large-zh-v1.5（1024 维，FP16） |
| Reranker | BAAI/bge-reranker-v2-m3（Cross-Encoder 精排） |
| 生成 | DeepSeek-V3（deepseek-chat） |
| 会话存储 | MySQL（会话、消息、上传文档元数据） |

## 🏗️ 检索链路

```
用户提问
  └─ 查询改写（LLM，多轮时结合历史补全省略主语）
       └─ 混合检索（pymilvus hybrid_search，各召回 20 条）
            ├─ 稀疏：query 原文 → Milvus 服务端 BM25 打分（sparse 字段）
            └─ 稠密：bge 编码 → embedding 字段（metric=IP）
            └─ 融合：WeightedRanker(BM25 0.3, 向量 0.7)，分数归一化
       └─ Reranker 精排（bge-reranker-v2-m3，取 Top-5）
            └─ DeepSeek 流式生成（严格基于检索资料，附引用编号与免责声明）
```

**Milvus collection schema**（由 `scripts/ingest.py` 建立，字段名与代码强绑定）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64, auto_id | 主键 |
| `embedding` | FLOAT_VECTOR(1024) | bge 稠密向量，AUTOINDEX / IP |
| `content` | VARCHAR(65535) | 原文，enable_analyzer + chinese 分词 |
| `sparse` | SPARSE_FLOAT_VECTOR | **由 BM25 Function 服务端生成，客户端不写** |
| `metadata` | JSON | `department` / `title` / `source` / `filename` |

可调参数集中在 `backend/app/vectorstore.py`：`BM25_WEIGHT`、`VECTOR_WEIGHT`、召回 `k`（默认 20，融合后给 Reranker 的候选数为 `min(2k, 50)`）。

## 📁 项目结构

```
医疗RAG/
├── Chinese-medical-dialogue-data-master/   # 原始数据集（GBK，不入库）
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 入口：/api/health /chat /ingest /ingest-stream
│   │   ├── config.py          # 配置（读取 .env）
│   │   ├── vectorstore.py     # ★ 检索与写入核心（pymilvus 原生混合检索）
│   │   ├── reranker.py        # bge-reranker-v2-m3 精排 + 去重 + 科室过滤
│   │   ├── query_rewriter.py  # 查询改写（单次/多路/结合上下文）+ 会话标题生成
│   │   ├── rag_chain.py       # RAG 链组装 + 医疗 Prompt
│   │   ├── llm.py             # DeepSeek 客户端
│   │   ├── data_loader.py     # CSV/JSON 数据集加载 + 上传文件读写
│   │   ├── text_split.py      # 中文滑窗切分
│   │   ├── database.py        # MySQL 连接
│   │   ├── models.py          # 会话 / 消息 / 上传文档表
│   │   └── api/
│   │       ├── documents.py     # 上传、列表、删除（联动清理向量）
│   │       └── conversations.py # 会话增删改查
│   ├── scripts/
│   │   ├── ingest.py            # ★ 建 collection + 批量入库（内置 BM25）
│   │   ├── preflight_check.py   # 入库前置体检（环境/schema/依赖/资源）
│   │   ├── clean_data.py        # 原始 CSV → 按科室清洗 JSON
│   │   ├── check_data.py        # 查看库内数据 / 检索测试
│   │   ├── eval_rag.py          # 评估 + 权重网格搜索
│   │   ├── init_db.py           # 创建 MySQL 库
│   │   └── gen_bm25_cache.py    # 【已废弃】旧的客户端 BM25 缓存脚本
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # React + Vite + TS
└── docker-compose.yml           # Milvus standalone + etcd + minio
```

## 🚀 快速开始

### 0) 环境要求

- Docker（跑 Milvus）、MySQL 8（跑会话存储）
- Python 3.11，CUDA 可选（`EMBEDDING_DEVICE=cuda` 可提速 20~40 倍）
- 首次运行会下载模型：bge-large-zh-v1.5 约 1.3GB、bge-reranker-v2-m3 约 2GB

### 1) 启动 Milvus（需 2.5+，内置 BM25 Function）

```bash
docker compose up -d
# 等待 30s+，访问 http://localhost:9091 查看健康状态
```

### 2) 配置后端

```bash
cd backend
python -m venv .venv && .\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

cp .env.example .env
# 必填：DEEPSEEK_API_KEY、DATABASE_URL（MySQL 连接串）
python scripts/init_db.py        # 创建 medical_rag 数据库（表在后端启动时自动建）
```

### 3) 数据入库

先做前置体检（只读，不改任何数据）：

```bash
python scripts/preflight_check.py
```

再入库：

```bash
python scripts/ingest.py --cleaned --limit 500    # 冒烟：每科 500 条
python scripts/ingest.py --cleaned                # 全量：60.5 万条（CPU 耗时较长）
python scripts/ingest.py --cleaned --dept 内科    # 只入指定科室
```

> ⚠️ **`ingest.py` 默认会先删除并重建 collection（清空已有数据）**，追加写入请加 `--no-recreate`。
> 清洗数据在 `backend/data/by_department/*.json`，由 `scripts/clean_data.py` 从原始 CSV 生成。

### 4) 启动前后端

```bash
# 终端 1：后端
cd backend && uvicorn app.main:app --reload --port 8000

# 终端 2：前端
cd frontend && npm install && npm run dev
```

打开 http://localhost:5173 即可提问（前端已配好 `/api` 代理到 8000）。
首次提问会加载 Embedding 与 Reranker 模型，需要等几十秒属于正常。

## 🔌 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（Milvus 地址、collection） |
| POST | `/api/chat` | 流式问答（SSE：`token` / `sources` / `conversation_id` / `error`） |
| POST | `/api/ingest` | 同步入库（小批量，参数 `limit_per_file`） |
| POST | `/api/ingest-stream` | 流式入库（SSE 进度） |
| POST | `/api/upload` | 上传文档（.pdf/.docx/.txt/.md/.csv，≤100MB，同名自动改名） |
| GET | `/api/uploads` | 已上传文档列表 |
| DELETE | `/api/uploads/{doc_id}` | **按 id 删除**：同时删磁盘文件、数据库记录与已入库向量 |
| GET/POST | `/api/conversations` | 会话列表 / 新建会话 |
| GET/PUT/DELETE | `/api/conversations/{id}` | 会话详情 / 改标题 / 删除（消息级联删除） |
| GET | `/api/conversations/{id}/messages` | 会话消息列表 |

## 🧪 评估与调优

```bash
python scripts/eval_rag.py build --num 100                    # 从清洗数据抽样建测试集
python scripts/eval_rag.py eval --mode hybrid_rerank          # vector/bm25/hybrid/hybrid_rerank
python scripts/eval_rag.py eval --mode hybrid_rerank --judge  # 额外跑 LLM 裁判打分
python scripts/eval_rag.py tune                               # BM25/向量权重网格搜索
python scripts/eval_rag.py summary                            # 汇总对比各模式
```

评估与生产走**同一条** `hybrid_search` 链路，因此 tune 出来的最优权重可直接填回 `vectorstore.py`。

## 🛠️ 运维与排错

**首次请求慢 / 卡在模型下载**
本机若无法访问 huggingface.co，会反复重试导致等待数分钟。模型已在本地缓存时可开启离线模式：
```bash
set HF_HUB_OFFLINE=1          # PowerShell: $env:HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

**chat 报错排查顺序**
1. `GET /api/health` 是否返回 200；
2. `python scripts/check_data.py` 能否看到 60 万条数据；
3. 看后端日志里的异常类型：
   - `cannot create index on non-exist field: vector`，或 `not allowed to retrieve raw data of field sparse`
     → 说明有代码又用回了 `langchain-milvus` 的 `Milvus` 封装。该封装默认按 `text/vector/pk` 命名字段，
     检索时还会用 `output_fields=["*"]` 取回不可回传的 sparse 字段，与本 schema 不兼容。
     本项目检索与写入都走 pymilvus 原生 API（`hybrid_search` / `insert`），不要把封装写法加回来；
   - `未配置 DEEPSEEK_API_KEY` → 检查 `backend/.env`。

**重新入库 / 清空数据**
- 重建：`python scripts/ingest.py --cleaned`（先 drop 再建）
- 彻底清空（含数据卷）：`docker compose down -v`
- 删除单条/单科室：按 `metadata["department"]` / `metadata["filename"]` 过滤删除

**查看与检索自测**
```bash
python scripts/check_data.py                                  # 概况 + 科室分布 + 样例
python scripts/check_data.py --query "头痛怎么办" --mode hybrid --k 5
```

## ⚠️ 医疗安全提示

- 回答**仅供参考**，不能替代执业医师诊断，严禁用于实际诊疗决策
- 数据来自开源医疗问答数据集，可能存在错漏
- 已限制低温度生成、拒答处方剂量，并对急症提示拨打 120
- 接口目前**没有鉴权**，仅适合本机/内网 demo，不要直接暴露到公网

## 📌 后续方向

- [ ] 接口鉴权与多用户隔离（会话归属）
- [ ] 检索权重的自动化调优（把 eval 结果接入配置）
- [ ] 文档去重与增量索引（同一文件重复上传会重复入库）
- [ ] 会话历史压缩（当前上下文改写只取最近 6 条消息）
- [ ] Ragas / LangSmith 评估体系接入
- [ ] 医疗同义词词典扩展召回、LangGraph Agent 主动反问
